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
    # train_loader 里的数据已经是 One-Hot (B, 32) 了
    train_loader, task, test_candidates, test_scores = build_paired_dataloader(cfg)
    
    # 获取正确的输入维度 (应该为 32)
    input_dim = test_candidates.shape[1]
    print(f"Model Input Dimension: {input_dim}")

    # ==========================================
    # Part A: 训练打分代理模型 (Score Proxy)
    # ==========================================
    print("\nTraining Score Proxy (for Guidance)...")
    
    # 初始化 Proxy
    proxy = ScoreProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.Adam(proxy.parameters(), lr=1e-3)
    
    # 关键修复：准备全量训练数据，必须手动转 One-Hot ！！！
    # 之前报错就是因为直接用了 task.x (N, 8) 而没有转成 (N, 32)
    if task.is_discrete:
        # 1. 拿到原始索引 (N, 8)
        raw_x = torch.tensor(task.x, dtype=torch.long).to(cfg.DEVICE)
        if raw_x.ndim == 3: raw_x = raw_x.squeeze(-1) # 防御性 reshape
        
        # 2. 强制转 One-Hot (N, 8, 4)
        vocab_size = 4
        all_x_onehot = F.one_hot(raw_x, num_classes=vocab_size).float()
        
        # 3. 展平 (N, 32) -> 这才是 Proxy 能吃的格式
        all_x = all_x_onehot.view(all_x_onehot.shape[0], -1)
    else:
        # 连续任务直接用
        all_x = torch.tensor(task.x, dtype=torch.float32).to(cfg.DEVICE)
        if all_x.dim() > 2: all_x = all_x.view(all_x.shape[0], -1)

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
        vocab_size = 4
        seq_len = x_optimized_continuous.shape[1] // vocab_size
        
        # 还原形状 (B, 8, 4)
        x_opt_reshaped = x_optimized_continuous.view(cfg.NUM_SAMPLES, seq_len, vocab_size)
        
        # Argmax 转回索引
        x_opt_indices = torch.argmax(x_opt_reshaped, dim=2).cpu().numpy()
        
        # 预测
        final_scores = task.predict(x_opt_indices)
    else:
        final_scores = task.predict(x_optimized_continuous.cpu().numpy())
    
    # 结果展示
    print("-" * 30)
    print(f"Optimization Results (Top {cfg.NUM_SAMPLES} worst samples optimized):")
    print(f"Original Score Mean: {original_y.mean().item():.4f}")
    
    if np.isneginf(final_scores).any():
        print("Warning: Still found -inf in scores! (Check data validity)")
    else:
        print(f"Optimized Score Mean: {final_scores.mean():.4f}")
        print(f"Optimized Score Max:  {final_scores.max():.4f}")
        
        # 按得分升序排序后取 50% / 80% / 100% 位置（1-based 排名）
        sorted_scores = np.sort(np.asarray(final_scores).reshape(-1))  # 小 -> 大
        total = sorted_scores.shape[0]
        def pick_rank_fraction(f):
            idx = max(0, min(total - 1, int(np.ceil(f * total) - 1)))
            return sorted_scores[idx]
        def to_scalar(x):
            return float(np.asarray(x).squeeze())
        p50 = to_scalar(pick_rank_fraction(0.50))
        p80 = to_scalar(pick_rank_fraction(0.80))
        p100 = to_scalar(pick_rank_fraction(1.00))  # 最后一个
        print(f"Percentile Scores (50th / 80th / 100th, asc): {p50:.4f} | {p80:.4f} | {p100:.4f}")
        
        improvement = final_scores.mean() - original_y.mean().item()
        print(f"Average Improvement:  {improvement:.4f}")
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