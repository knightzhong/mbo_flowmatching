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
    
    y_sorted_indices = torch.argsort(offline_y) 
    max_train_y_norm = offline_y.max().item()
    # ==========================================
    # Part C: 评估 (CFG + Proxy Screening / Rejection Sampling)
    # ==========================================
    print("\nRunning Evaluation with CFG + Proxy Screening...")
    
    # 1. 设置筛选参数
    # 我们要生成 1280 个候选样本，最后选出 128 个
    # 这样 Proxy 就能帮我们过滤掉那些“虽然符合条件但实际质量不高”的样本
    NUM_CANDIDATES = 1280 
    FINAL_K = 128         
    BEST_SCALE = 4.0      # 根据之前的实验，4.0 是效果最好的 Scale
    
    # 2. 准备起始数据 (随机采样策略)
    # 策略：从 Bottom 50% 的数据中随机抽取 1280 个不同的样本
    # 这样能保证起始点的多样性，避免重复计算
    
    # 确定 Source Pool 的范围 (Bottom 50%)
    # 注意：这里我们利用 y_sorted_indices 来定位低分区域
    num_total = offline_x.shape[0]
    num_source_pool = int(num_total * 0.5) # 取前 50% 最差的作为池子
    
    # 获取 Bottom 50% 的索引
    source_indices = y_sorted_indices[:num_source_pool]
    
    # 从中随机采样 1280 个索引 (无放回，如果池子够大)
    # 这种随机性是 MBO 成功的关键之一
    random_perm = torch.randperm(len(source_indices))
    selected_indices = source_indices[random_perm[:NUM_CANDIDATES]]
    
    # 提取起始样本 x_start 和对应的 y_low
    x_start_candidates = offline_x[selected_indices].to(cfg.DEVICE)
    y_low_candidates = offline_y[selected_indices].view(-1, 1).to(cfg.DEVICE)
    
    # 准备目标分数 y_target (Broadcast 到 1280 个)
    # 依然设定为训练集最大值的 1.1 倍，或者直接设定为高分
    y_target_val = max_train_y_norm * 1.1 
    y_target_candidates = torch.full((NUM_CANDIDATES, 1), y_target_val, device=cfg.DEVICE)
    
    print(f"Generating {NUM_CANDIDATES} candidates (Scale={BEST_SCALE}) from random bottom 50% samples...")
    
    # 3. 批量生成 (Generation)
    # 如果显存不够，可以分 Batch 跑，这里假设 TFBind8 (32维) 1280个样本能一次跑完
    cfm.model.eval()
    with torch.no_grad():
        x_generated = cfm.sample(
            x_start_candidates, 
            y_high=y_target_candidates, 
            y_low=y_low_candidates,
            steps=cfg.ODE_STEPS,
            guidance_scale=BEST_SCALE 
        )
        
    # 4. Proxy 打分筛选 (Screening)
    print("Screening candidates with Proxy...")
    proxy.eval()
    with torch.no_grad():
        # 让 Proxy 给这 1280 个生成样本打分
        # 注意：Proxy 输出可能是 (N, 1)，需要 view(-1)
        pred_scores = proxy(x_generated).view(-1)
        
        # 选出 Proxy 认为分数最高的 128 个 (Top K)
        # 这一步是把“运气”变成“实力”的关键
        top_scores, top_indices = torch.topk(pred_scores, k=FINAL_K)
        
        # 提取最终优胜者
        x_final = x_generated[top_indices]
        # 同时也提取出它们对应的原始 x_start，方便计算 improvement
        # x_start_final = x_start_candidates[top_indices]
        # y_original_final = offline_y[selected_indices][top_indices.cpu()] # 对应的原始真实分数
        
    # 5. Oracle 最终评估 (Oracle Evaluation)
    print(f"Evaluating Top {FINAL_K} Candidates selected by Proxy...")
    
    # 反标准化
    x_denorm = x_final * std_x + mean_x
    
    # 解码与打分
    if task.is_discrete:
        vocab_size = 4
        seq_len = x_denorm.shape[1] // vocab_size
        x_reshaped = x_denorm.view(x_denorm.shape[0], seq_len, vocab_size)
        x_opt_indices = torch.argmax(x_reshaped, dim=2).cpu().numpy()
        final_scores = task.predict(x_opt_indices)
    else:
        final_scores = task.predict(x_denorm.cpu().numpy())
        
    # ==========================================
    # 统计结果
    # ==========================================
    final_scores_flat = np.asarray(final_scores).reshape(-1)
    valid_scores = final_scores_flat[np.isfinite(final_scores_flat)]
    
    if len(valid_scores) > 0:
        # 归一化分数计算 (与之前逻辑一致)
        # oracle_y_min/max 需要你在前面定义好，或者这里重新从 task_to_min/max 获取
        task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585268}
        task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287067}
        oracle_y_min = task_to_min.get(cfg.TASK_NAME, offline_y.min().item())
        oracle_y_max = task_to_max.get(cfg.TASK_NAME, offline_y.max().item())
        
        normalized_scores = (valid_scores - oracle_y_min) / (oracle_y_max - oracle_y_min)
        percentiles = np.percentile(normalized_scores, [100, 80, 50])
        
        print("-" * 30)
        print(f"Optimization Results (Proxy Screened Top {FINAL_K}):")
        # print(f"Original Score Mean (of selected starts): {y_original_final.mean().item():.4f}") # 可选
        print(f"Optimized Score Mean (valid only): {valid_scores.mean():.4f}")
        print(f"Optimized Score Max (valid only):  {valid_scores.max():.4f}")
        print(f"Normalized Percentile Scores (100th / 80th / 50th): {percentiles[0]:.4f} | {percentiles[1]:.4f} | {percentiles[2]:.4f}")
        print("-" * 30)
        return percentiles
    else:
        print("No valid scores generated.")
        return None

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