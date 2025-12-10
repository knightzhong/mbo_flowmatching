import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import Config
from data_loader import build_paired_dataloader
from models import VectorFieldNet, ScoreProxy
from solver import ConditionalFlowMatching

def main():
    cfg = Config()
    print(f"Experiment: {cfg.TASK_NAME} | Device: {cfg.DEVICE}")
    
    # 1. 准备数据
    # 接收 mean_x, std_x
    train_loader, task, test_candidates, test_scores, mean_x, std_x = build_paired_dataloader(cfg)
    
    # 将统计量移动到设备上
    mean_x = mean_x.to(cfg.DEVICE)
    std_x = std_x.to(cfg.DEVICE)
    
    raw_x = task.x
    if isinstance(raw_x, np.ndarray):
        raw_x = torch.tensor(raw_x)
    if raw_x.ndim == 3:
        raw_x = raw_x.squeeze(-1)
    seq_len = raw_x.shape[1] if raw_x.ndim >= 2 else test_candidates.shape[1]
    vocab_size = test_candidates.shape[1] // seq_len
    input_dim = test_candidates.shape[1]
    print(f"Model Input Dimension: {input_dim}")

    # ==========================================
    # Part A: 训练打分代理模型 (Score Proxy)
    # ==========================================
    print("\nTraining Score Proxy (for Guidance)...")
    
    proxy = ScoreProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.Adam(proxy.parameters(), lr=1e-3)
    
    # === 修改：使用与 Flow Model 完全一致的数据处理 ===
    if task.is_discrete:
        # 1. 拿原始数据
        raw_x = task.x
        # 2. 转 Logits
        all_x_logits = task.to_logits(raw_x)
        all_x_logits = all_x_logits.reshape(all_x_logits.shape[0], -1)
        all_x = torch.tensor(all_x_logits, dtype=torch.float32).to(cfg.DEVICE)
    else:
        all_x = torch.tensor(task.x, dtype=torch.float32).to(cfg.DEVICE)
        if all_x.dim() > 2: all_x = all_x.view(all_x.shape[0], -1)

    # 3. !!! 关键：应用同样的标准化 !!!
    all_x = (all_x - mean_x) / std_x

    all_y = torch.tensor(task.y, dtype=torch.float32).to(cfg.DEVICE).view(-1, 1)
    
    print(f"Proxy Training Data Shape: {all_x.shape}") # 确认是 [32898, 32]

    # 简单的训练循环 (50 epochs 足够了)
    for i in range(50):
        proxy_opt.zero_grad()
        pred_y = proxy(all_x)
        loss = nn.MSELoss()(pred_y, all_y)
        loss.backward()
        proxy_opt.step()
        if (i + 1) % 10 == 0:
            print(f"Proxy Epoch {i+1}/50 | Loss: {loss.item():.4f}")
    
    # ==========================================
    # Part B: 训练 Flow Matching 模型
    # ==========================================
    print("\nStart Training Flow Model...")
    net = VectorFieldNet(input_dim=input_dim, hidden_dim=cfg.LATENT_DIM)
    cfm = ConditionalFlowMatching(net, cfg.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    
    for epoch in range(cfg.EPOCHS):
        net.train()
        total_loss = 0
        for x0, x1 in train_loader:
            optimizer.zero_grad()
            loss = cfm.compute_loss(x0, x1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Flow Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {avg_loss:.6f}")
            
    # ==========================================
    # Part C: 评估 (Evaluation + Guidance)
    # ==========================================
    print("\nRunning Evaluation with Guidance...")
    _, worst_indices = torch.topk(test_scores, k=cfg.NUM_SAMPLES, largest=False)
    x_start = test_candidates[worst_indices]
    original_y = test_scores[worst_indices]
    # 选取分数排序最低的 50% 中最高的 N 个样本作为起始点
    # bottom_half_k = max(1, test_scores.shape[0] // 2)
    # bottom_half_indices = torch.topk(test_scores, k=bottom_half_k, largest=False).indices
    # bottom_half_scores = test_scores[bottom_half_indices]
    # select_k = min(cfg.NUM_SAMPLES, bottom_half_scores.shape[0])
    # top_in_bottom = torch.topk(bottom_half_scores, k=select_k, largest=True).indices
    # best_indices = bottom_half_indices[top_in_bottom]
    # x_start = test_candidates[best_indices]
    # original_y = test_scores[best_indices]
    
    # 定义引导函数
    def guidance_func(x):
        return proxy(x)
    
    # === 调参区域 ===
    # 试着改这个值：1.0, 10.0, 50.0
    GUIDANCE_SCALE = 0.0 
    print(f"Using Guidance Scale: {GUIDANCE_SCALE}")
    
    # 带引导的采样
    x_optimized_continuous = cfm.sample(
        x_start, 
        steps=cfg.ODE_STEPS,
        guidance_fn=guidance_func,
        guidance_scale=GUIDANCE_SCALE
    )
    
    # 后处理与打分
    if task.is_discrete:
        # === 核心修改：反标准化 (Denormalize) ===
        # 先恢复成原始的 Logits 分布，再做 Argmax
        x_denorm = x_optimized_continuous * std_x + mean_x
        
        # 动态推断 vocab/seq_len，避免与 logits 维度不符
        seq_len = seq_len
        vocab_size = x_denorm.shape[1] // seq_len
        
        # 还原形状
        x_opt_reshaped = x_denorm.view(cfg.NUM_SAMPLES, seq_len, vocab_size)
        
        # Argmax 转回索引
        x_opt_indices = torch.argmax(x_opt_reshaped, dim=2).cpu().numpy()
        
        # 预测
        final_scores = task.predict(x_opt_indices)
    else:
        # 连续任务也要反标准化
        x_denorm = x_optimized_continuous * std_x + mean_x
        final_scores = task.predict(x_denorm.cpu().numpy())
    
    # 结果展示
    print("-" * 30)
    print(f"Optimization Results (Top {cfg.NUM_SAMPLES} worst samples optimized):")
    print(f"Original Score Mean: {original_y.mean().item():.4f}")
    
    final_scores_flat = np.asarray(final_scores).reshape(-1)
    finite_mask = np.isfinite(final_scores_flat)
    num_bad = final_scores_flat.shape[0] - finite_mask.sum()
    if num_bad > 0:
        print(f"Warning: found {num_bad} non-finite scores (inf/-inf/nan); excluded from stats.")
    valid_scores = final_scores_flat[finite_mask]
    
    if valid_scores.size == 0:
        print("No valid scores to summarize.")
    else:
        print(f"Optimized Score Mean (valid only): {valid_scores.mean():.4f}")
        print(f"Optimized Score Max (valid only):  {valid_scores.max():.4f}")
        
        # 按得分升序排序后取 50% / 80% / 100% 位置（1-based 排名）
        sorted_scores = np.sort(valid_scores)  # 小 -> 大
        total = sorted_scores.shape[0]
        def pick_rank_fraction(f):
            idx = max(0, min(total - 1, int(np.ceil(f * total) - 1)))
            return sorted_scores[idx]
        def to_scalar(x):
            return float(np.asarray(x).squeeze())
        p50 = to_scalar(pick_rank_fraction(0.50))
        p80 = to_scalar(pick_rank_fraction(0.80))
        p100 = to_scalar(pick_rank_fraction(1.00))  # 最后一个
        print(f"Percentile Scores (50th / 80th / 100th, asc, valid only): {p50:.4f} | {p80:.4f} | {p100:.4f}")
        
        improvement = valid_scores.mean() - original_y.mean().item()
        print(f"Average Improvement (valid only):  {improvement:.4f}")
    print("-" * 30)

def seed_everything(seed=42):
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    for seed in range(8):
        seed_everything(seed)
        main()
        print(f"Seed: {seed} completed")