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
    
    # 1. 接收所有统计量（与 ROOT 一致）
    # 注意：返回整个数据集，而不是只返回 bottom 50%（与 ROOT 一致）
    train_loader, task, offline_x, offline_y, mean_x, std_x, mean_y, std_y = build_paired_dataloader(cfg)
    
    # 保存原始的 mean_y 和 std_y 值（用于计算目标分数，需要在移动到设备前提取）
    if isinstance(mean_y, torch.Tensor):
        mean_y_orig = mean_y.item() if mean_y.numel() == 1 else float(mean_y.mean().item())
    elif isinstance(mean_y, np.ndarray):
        mean_y_orig = float(mean_y.item() if mean_y.size == 1 else mean_y.mean())
    else:
        mean_y_orig = float(mean_y)
        
    if isinstance(std_y, torch.Tensor):
        std_y_orig = std_y.item() if std_y.numel() == 1 else float(std_y.mean().item())
    elif isinstance(std_y, np.ndarray):
        std_y_orig = float(std_y.item() if std_y.size == 1 else std_y.mean())
    else:
        std_y_orig = float(std_y)
    
    # 移动到设备
    mean_x, std_x = mean_x.to(cfg.DEVICE), std_x.to(cfg.DEVICE)
    mean_y, std_y = mean_y.to(cfg.DEVICE), std_y.to(cfg.DEVICE)
    
    # 获取输入维度（从 offline_x 获取，已经标准化）
    input_dim = offline_x.shape[1]
    print(f"Model Input Dimension: {input_dim}")

    # ==========================================
    # Part A: 训练打分代理模型 (Score Proxy)
    # ==========================================
    print("\nTraining Score Proxy (for Guidance)...")
    
    proxy = ScoreProxy(input_dim=input_dim).to(cfg.DEVICE)
    proxy_opt = torch.optim.Adam(proxy.parameters(), lr=1e-3)
    
    # 使用与 Flow Model 完全一致的数据处理（已经标准化）
    # 从 train_loader 中获取所有数据用于训练 proxy
    all_x_list = []
    all_y_list = []
    for x0, x1, y_high, y_low in train_loader:  # 现在接收 y_high 和 y_low
        all_x_list.append(x0)
        all_x_list.append(x1)
        all_y_list.append(y_high)  # 使用 y_high 作为目标
        all_y_list.append(y_high)
    all_x = torch.cat(all_x_list, dim=0).to(cfg.DEVICE)
    all_y = torch.cat(all_y_list, dim=0).to(cfg.DEVICE).view(-1, 1)
    
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
        for x0, x1, y_high, y_low in train_loader:  # 现在接收 y_high 和 y_low
            optimizer.zero_grad()
            loss = cfm.compute_loss(x0, x1, y_high, y_low)  # 传入 y_high 和 y_low
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
    # 回退到与训练数据一致的策略：从最低分开始优化
    # 训练时学习的是 bottom 50% -> top 50%，所以测试时也应该从低分开始
    # 这样可以保证训练和测试的一致性，避免模型从未见过的映射导致失败
    y_sorted_indices = torch.argsort(offline_y)
    x_sorted = offline_x[y_sorted_indices]
    y_sorted = offline_y[y_sorted_indices]
    
    # 从最低分的样本中采样（与训练数据一致）
    x_start = x_sorted[:cfg.NUM_SAMPLES].to(cfg.DEVICE)
    original_y = y_sorted[:cfg.NUM_SAMPLES]
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
    
    # === 修改2：基于 oracle 范围设置目标分数（与 ROOT 一致）===
    # 定义任务的最小/最大值（与 ROOT 一致）
    task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268, 
                   'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067, 
                   'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    
    oracle_y_min = task_to_min.get(cfg.TASK_NAME, 0.0)
    oracle_y_max = task_to_max.get(cfg.TASK_NAME, 1.0)
    
    # 归一化 oracle_y_max（与 ROOT 完全一致，不做任何调整）
    # 使用之前保存的原始值（未标准化的统计量）
    normalized_oracle_y_max = (oracle_y_max - mean_y_orig) / std_y_orig
    
    # 使用 alpha 系数（ROOT 中使用 0.8）
    alpha = 0.8  # 与 ROOT 一致
    target_score = normalized_oracle_y_max * alpha
    
    # 直接使用 ROOT 的方式，不做任何调整
    y_target_norm = torch.full((cfg.NUM_SAMPLES, 1), target_score, device=cfg.DEVICE)
    
    print(f"Oracle Y Range: [{oracle_y_min}, {oracle_y_max}]")
    print(f"Original Y Mean: {mean_y_orig:.4f}, Std: {std_y_orig:.4f}")
    print(f"Normalized Oracle Y Max: {normalized_oracle_y_max:.4f}")
    print(f"Target Score (normalized, alpha={alpha}): {target_score:.4f}")
    
    # 获取起始点的 y_low（低价值分数）
    y_low_start = original_y.view(-1, 1).to(cfg.DEVICE)  # (NUM_SAMPLES, 1)
    
    # 采样（现在传入 y_high 和 y_low）
    x_optimized_continuous = cfm.sample(
        x_start, 
        y_high=y_target_norm,  # 目标分数（高价值）
        y_low=y_low_start,  # 起始分数（低价值）
        steps=cfg.ODE_STEPS,
        guidance_fn=guidance_func,
        guidance_scale=GUIDANCE_SCALE
    )
    
    # ==========================================
    # 后处理：与 ROOT 保持一致
    # ==========================================
    # 1. 反标准化（与 ROOT 一致）
    x_denorm = x_optimized_continuous * std_x + mean_x
    
    # 2. 移动到 CPU 并转换为 numpy
    x_denorm = x_denorm.cpu().numpy()
    
    # 3. 如果是离散任务，需要 reshape 并调用 task.map_to_logits()
    if task.is_discrete:
        # 调用 map_to_logits（与 ROOT 一致）
        task.map_to_logits()
        
        # Reshape 为 (N, seq_len, vocab_size) 格式（与 ROOT 一致）
        denormalize_high_candidates = x_denorm.reshape(x_denorm.shape[0], task.x.shape[1], task.x.shape[2])
        
        # task.predict() 会自动处理 logits 到离散索引的转换
        final_scores = task.predict(denormalize_high_candidates)
    else:
        # 连续任务直接 predict
        final_scores = task.predict(x_denorm)
    
    # ==========================================
    # 结果评估：与 ROOT 一致（归一化百分位数分数）
    # ==========================================
    # 1. 计算归一化分数（与 ROOT 一致）
    final_scores_flat = np.asarray(final_scores).reshape(-1)
    finite_mask = np.isfinite(final_scores_flat)
    num_bad = final_scores_flat.shape[0] - finite_mask.sum()
    if num_bad > 0:
        print(f"Warning: found {num_bad} non-finite scores (inf/-inf/nan); excluded from stats.")
    valid_scores = final_scores_flat[finite_mask]
    
    if valid_scores.size == 0:
        print("No valid scores to summarize.")
        return
    
    # 2. 归一化分数（与 ROOT 一致）：(score - oracle_y_min) / (oracle_y_max - oracle_y_min)
    normalized_scores = (valid_scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
    
    # 3. 计算百分位数（与 ROOT 一致）
    percentiles = np.percentile(normalized_scores, [100, 80, 50])  # 100th, 80th, 50th
    
    # 结果展示
    print("-" * 30)
    print(f"Optimization Results (Top {cfg.NUM_SAMPLES} samples optimized):")
    print(f"Original Score Mean: {original_y.mean().item():.4f}")
    print(f"Optimized Score Mean (valid only): {valid_scores.mean():.4f}")
    print(f"Optimized Score Max (valid only):  {valid_scores.max():.4f}")
    print(f"Normalized Percentile Scores (100th / 80th / 50th): {percentiles[0]:.4f} | {percentiles[1]:.4f} | {percentiles[2]:.4f}")
    
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