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
    
    # 直接用全量 offline 数据训练 proxy，避免成对采样导致的重复
    all_x = offline_x.to(cfg.DEVICE)
    all_y = offline_y.to(cfg.DEVICE).view(-1, 1)
    
    print(f"Proxy Training Data Shape: {all_x.shape}") # 确认是 [32898, 32]

    # 简单的训练循环 (50 epochs 足够了)
    for i in range(200):
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
    
    # 定义引导函数
    def guidance_func(x):
        return proxy(x)
    # ==========================================
    # Part C: 评估 (Evaluation + Guidance)
    # ==========================================
    print("\nRunning Evaluation with Guidance...")
    
    # 1. 准备起始数据 (Bottom 样本)
    y_sorted_indices = torch.argsort(offline_y)
    x_start = offline_x[y_sorted_indices][:cfg.NUM_SAMPLES].to(cfg.DEVICE)
    original_y = offline_y[y_sorted_indices][:cfg.NUM_SAMPLES]
    
    # 2. 设定目标分数 (Target Score)
    # 因为 y 已经归一化了，所以 target=max(y_train) 是一个安全且强力的目标
    # 计算训练集中见过的最大归一化分数
    max_train_y_norm = offline_y.max().item()
    
    # 策略：稍微推高一点点 (1.1倍)，引导模型向更优处探索
    # 对于 TFBind8，这通常在 2.5 ~ 3.0 之间
    TARGET_SIGMA = max_train_y_norm * 1.1
    
    y_target_norm = torch.full((cfg.NUM_SAMPLES, 1), TARGET_SIGMA, device=cfg.DEVICE)
    y_low_start = original_y.view(-1, 1).to(cfg.DEVICE)
    
    print(f"Max Training Sigma: {max_train_y_norm:.4f}")
    print(f"Conditioning on Target Score: {TARGET_SIGMA:.4f}")
    
    # 3. 采样
    x_optimized_continuous = cfm.sample(
        x_start, 
        y_high=y_target_norm, 
        y_low=y_low_start,
        steps=cfg.ODE_STEPS,
        guidance_fn=guidance_func,
        guidance_scale=20.0 # 先设为0，验证纯 Flow 能力
    )
    
    # 4. 后处理与解码 (关键修改！)
    # 反标准化
    x_denorm = x_optimized_continuous * std_x + mean_x
    
    if task.is_discrete:
        # === 关键：手动解码 4 维 One-Hot ===
        # 此时 x_denorm 是 (N, 32)
        # 我们知道 TFBind8 是 8 个位置，每个位置 4 种可能
        vocab_size = 4
        seq_len = x_denorm.shape[1] // vocab_size # 应该是 8
        
        # Reshape 回 (N, 8, 4)
        x_reshaped = x_denorm.view(x_denorm.shape[0], seq_len, vocab_size)
        print('x_reshaped.shape') 
        print(x_reshaped.shape) 
        # Argmax: 找出每个位置概率最大的碱基索引 (0-3)
        # 这就是这一方案最稳的地方：哪怕数值有噪声，最大值通常是对的
        x_opt_indices = torch.argmax(x_reshaped, dim=2).cpu().numpy()
        
        # 直接把离散索引喂给 task.predict
        # 注意：不要调用 task.map_to_logits()，因为我们给的是 indices
        final_scores = task.predict(x_opt_indices)
    else:
        final_scores = task.predict(x_denorm.cpu().numpy())
    
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
    # oracle_y_min = offline_y.min().item()
    # oracle_y_max = offline_y.max().item() 
    # 使用 ROOT 中写死的 oracle 上下界，避免受当前 run 的 offline_y 范围影响
    task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268,
                   'AntMorphology-Exact-v0': -386.90036, 'DKittyMorphology-Exact-v0': -880.4585}
    task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067,
                   'AntMorphology-Exact-v0': 590.24445, 'DKittyMorphology-Exact-v0': 340.90985}
    oracle_y_min = task_to_min.get(cfg.TASK_NAME, offline_y.min().item())
    oracle_y_max = task_to_max.get(cfg.TASK_NAME, offline_y.max().item())
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

    # 返回当前 seed 的百分位结果，便于多 seed 汇总
    return percentiles

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
    all_percentiles = []
    for seed in range(8):
        seed_everything(seed)
        perc = main()
        if perc is not None:
            all_percentiles.append(perc)
        print(f"Seed: {seed} completed")

    if len(all_percentiles) > 0:
        all_percentiles = np.vstack(all_percentiles)  # shape (n_seed, 3)
        mean_percentiles = all_percentiles.mean(axis=0)
        std_percentiles = all_percentiles.std(axis=0)
        print("=" * 30)
        print("Average Normalized Percentile Scores across seeds (100th / 80th / 50th): "
              f"{mean_percentiles[0]:.4f} | {mean_percentiles[1]:.4f} | {mean_percentiles[2]:.4f}")
        print("Std of Normalized Percentile Scores across seeds (100th / 80th / 50th): "
              f"{std_percentiles[0]:.4f} | {std_percentiles[1]:.4f} | {std_percentiles[2]:.4f}")
        print("=" * 30)
    else:
        print("No percentile results collected across seeds.")