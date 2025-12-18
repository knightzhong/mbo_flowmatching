import torch
import numpy as np
import design_bench
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
def get_design_bench_data(task_name):
    """
    最佳实践：4维 One-Hot + 去量化噪声 + Z-Score 标准化
    """
    print(f"Loading task: {task_name}...")
    if task_name != 'TFBind10-Exact-v0':
        task = design_bench.make(task_name)
    else:
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": 10000})
    
    # === 1. 处理 X (输入) ===
    if task.is_discrete:
        # 获取原始离散索引 (N, 8)
        raw_x = task.x
        # 确保形状是 (N, L)
        if raw_x.ndim == 3: raw_x = raw_x.squeeze(-1)
        
        # 转为 Tensor
        x_indices = torch.tensor(raw_x, dtype=torch.long)
        
        # 强制转为 4 维 One-Hot (N, 8, 4)
        # 这一步保证了信息的完整性，不会像 Logits 那样丢一维
        vocab_size = 4
        x_onehot = F.one_hot(x_indices, num_classes=vocab_size).float()
        
        # === 关键步骤：去量化 (Dequantization) ===
        # 加入微小的正态噪声，将离散的 0/1 变成连续值
        # 0.05 是经验值，既能让数据连续，又不至于让模型分不清类别
        noise = 0.05 * torch.randn_like(x_onehot)
        x_continuous = x_onehot + noise
        
        # 展平为 (N, 32) -> 这就是模型要处理的连续向量
        offline_x = x_continuous.view(x_continuous.shape[0], -1).numpy()
    else:
        # 连续任务保持原样
        offline_x = task.x
    
    # === 2. 计算统计量 (用于 Z-Score) ===
    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x < 1e-6, 1.0, std_x) # 防除零
    
    # 执行标准化: N(0, 1)
    offline_x_norm = (offline_x - mean_x) / std_x
    
    # === 3. 处理 Y (分数) ===
    offline_y = task.y.reshape(-1)
    mean_y = np.mean(offline_y)
    std_y = np.std(offline_y)
    if std_y == 0: std_y = 1.0
    
    # 归一化 Y
    offline_y_norm = (offline_y - mean_y) / std_y
    
    print(f"Data Processed (One-Hot+Noise). X_dim={offline_x_norm.shape[1]}. Y_norm Mean={offline_y_norm.mean():.2f}, Std={offline_y_norm.std():.2f}")
    
    return task, offline_x_norm, offline_y_norm, mean_x, std_x, mean_y, std_y



import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def global_ot_pairing_and_cleaning(
    x_all, y_all, 
    x_elite, y_elite, 
    label_weight=10.0, 
    max_cost_threshold=20.0,  # 新增：剔除阈值 (根据 Cost 分布调整)
    device='cuda'
):
    """
    全显存极速版：全局配对 + 自动剔除劣质样本
    
    参数:
    x_all: Source Pool (通常是 100% 数据)
    x_elite: Target Pool (通常是 Top 20% 数据)
    max_cost_threshold: 如果最佳配对的代价超过此值，说明该 Source 无法被有效优化，剔除。
    """
    print(f"Executing Global Pure Matrix OT Pairing...")
    print(f"Input Source: {x_all.shape[0]}, Candidate Targets: {x_elite.shape[0]}")
    
    # 1. 数据上 GPU
    x_src = torch.tensor(x_all, device=device, dtype=torch.float32)
    y_src = torch.tensor(y_all, device=device, dtype=torch.float32).view(-1, 1)
    
    x_tgt = torch.tensor(x_elite, device=device, dtype=torch.float32)
    y_tgt = torch.tensor(y_elite, device=device, dtype=torch.float32).view(-1, 1)
    
    # 2. 全局计算 Cost Matrix (无循环，极速)
    # 显存预估: 3w * 6k * 4bytes ≈ 700MB -> 轻松吃下
    
    # A. 几何距离 (Squared L2)
    # (N_src, N_tgt)
    dist_geom = torch.cdist(x_src, x_tgt, p=2) ** 2
    
    # B. 功能增益 (Gain = y_tgt - y_src)
    # Broadcast: (1, N_tgt) - (N_src, 1) -> (N_src, N_tgt)
    gain = y_tgt.T - y_src 
    
    # C. 综合 Cost = Dist - lambda * Gain
    print(f"dist_geom: {dist_geom.shape}, gain: {gain.shape}")
    print(f"dist_geom: {dist_geom}, gain: {gain}")
    epsilon= 1e-3
    cost_matrix = dist_geom / (gain + epsilon) #dist_geom - label_weight * gain
    
    # 3. 施加硬约束
    # 约束 A: 必须变好 (Gain > 0)。如果没变好，Cost 设为无限大
    mask_bad_gain = gain <= 1e-6 # 加个 epsilon 防止 0 增益
    cost_matrix[mask_bad_gain] = float('inf')
    
    # 4. 贪心匹配 (Many-to-One)
    # 对每行(Source)找最小值的列索引(Target)
    # min_values: 每个 Source 的最低代价
    # best_indices: 对应的最佳 Target 索引
    min_values, best_indices = torch.min(cost_matrix, dim=1)
    
    # 5. 数据清洗 (剔除代价过大的样本)
    # 即使是最佳配对，如果 Cost 依然是 inf (说明周围全是低分) 
    # 或者 Cost 很大 (说明距离太远，或者增益太小)，则剔除。
    
    # 定义保留掩码
    # 1. 必须找到了有效目标 (Cost != inf)
    # 2. Cost 必须小于阈值 (排除"强扭的瓜")
    valid_mask = (min_values != float('inf')) & (min_values < max_cost_threshold)
    
    num_kept = valid_mask.sum().item()
    num_dropped = x_src.shape[0] - num_kept
    print(f"Cleaning Results: Kept {num_kept}, Dropped {num_dropped} (Too expensive or no gain)")
    
    # 6. 构建最终数据集
    # 只保留 Mask 为 True 的行
    final_x_src = x_src[valid_mask].cpu() # 移回 CPU 组装 DataLoader
    final_y_src = y_src[valid_mask].cpu()
    
    # 找出对应的 Target
    valid_indices = best_indices[valid_mask]
    final_x_tgt = x_tgt[valid_indices].cpu()
    final_y_tgt = y_tgt[valid_indices].cpu()
    
    # 可选：打印一下 Cost 的统计信息，帮你确定阈值
    if num_kept > 0:
        valid_costs = min_values[valid_mask]
        print(f"Cost Stats (Kept): Mean={valid_costs.mean():.2f}, Max={valid_costs.max():.2f}")
    
    return final_x_src, final_x_tgt, final_y_tgt, final_y_src
def local_robust_pairing(x_low, x_high, y_high, y_low, k=50, max_dist_threshold=5.0):
    """
    结合了 Semantic Pairing 的局部性和 OT 的清洗机制
    """
    # 保证输入是 Tensor（有可能从 numpy 直接传进来）
    if isinstance(x_low, np.ndarray):
        x_low = torch.tensor(x_low, dtype=torch.float32)
    if isinstance(x_high, np.ndarray):
        x_high = torch.tensor(x_high, dtype=torch.float32)
    if isinstance(y_high, np.ndarray):
        y_high = torch.tensor(y_high, dtype=torch.float32)
    if isinstance(y_low, np.ndarray):
        y_low = torch.tensor(y_low, dtype=torch.float32)

    # 1. 计算距离矩阵 (和原来一样)
    dists = torch.cdist(x_low, x_high, p=2) 
    
    # 2. 只看最近的 k 个邻居 (保留局部性核心!)
    # values: (N_low, k) 距离
    # indices: (N_low, k) 邻居下标
    topk_dists, topk_indices = torch.topk(dists, k=k, dim=1, largest=False)
    
    # 3. 获取这些邻居的分数
    # (N_low, k)
    neighbor_scores = y_high[topk_indices]
    
    # 4. 在 k 个邻居中，选 "分数最高" 的 (Greedy)
    # 或者选 "性价比最高" 的 (Score - lambda * Dist) -> 推荐这个，更平滑
    # 这里我们先保持和原来类似的贪心，但加入清洗
    best_gain_in_k, best_idx_in_k = torch.max(neighbor_scores, dim=1)
    
    # 映射回全局索引
    best_target_indices = topk_indices.gather(1, best_idx_in_k.unsqueeze(1)).squeeze()
    
    # === 关键改进：清洗机制 (Rejection) ===
    # 即使是邻居里分最高的，也可能：
    # A. 分数还没我自己高 (无效优化)
    # B. 距离虽然是最近的，但依然太远 (离群点)
    
    # 计算实际增益
    actual_gains = y_high[best_target_indices] - y_low
    # 计算实际距离
    actual_dists = topk_dists.gather(1, best_idx_in_k.unsqueeze(1)).squeeze()
    
    # 定义保留条件：
    # 1. 必须有正向增益 (Gain > 0)
    # 2. 距离必须足够近 (Dist < threshold) -> 防止长距离插值
    valid_mask = (actual_gains > 0.0) & (actual_dists < max_dist_threshold)
    
    print(f"Cleaning: Kept {valid_mask.sum()} / {len(x_low)} pairs.")
    print(f"Dropped {len(x_low) - valid_mask.sum()} pairs (Negative gain or too far).")
    
    # 筛选并显式转为 CPU float32 Tensor（避免对已有 Tensor 再次 torch.tensor 引发 warning）
    x_src = x_low[valid_mask].clone().detach().to(torch.float32).cpu()
    y_src = y_low[valid_mask].clone().detach().to(torch.float32).cpu()
    x_tgt = x_high[best_target_indices][valid_mask].clone().detach().to(torch.float32).cpu()
    y_tgt = y_high[best_target_indices][valid_mask].clone().detach().to(torch.float32).cpu()
    
    return x_src, x_tgt, y_tgt, y_src
def build_paired_dataloader(config, proxy=None):
    task, x_norm, y_norm, mean_x, std_x, mean_y, std_y = get_design_bench_data(config.TASK_NAME)
    
    # === 策略定义 ===
    
    
    # Target: Top 20% 精英数据
    target_threshold = np.percentile(y_norm, 80)
    source_threshold = np.percentile(y_norm, 50)
    mask_target = y_norm >= target_threshold
    mask_source = y_norm < source_threshold

    # Source: 100% 全量数据
    x_source_pool = x_norm[mask_source]
    y_source_pool = y_norm[mask_source]
    x_elite_pool = x_norm[mask_target]
    y_elite_pool = y_norm[mask_target]
    
    # === 执行配对 + 清洗 ===
    # max_cost_threshold 是个超参。
    # 建议先设个大数 (比如 100.0) 跑一次看 log 里的 Stats，再收缩。
    # x_src, x_tgt, y_tgt, y_src = global_ot_pairing_and_cleaning(
    #     x_source_pool, y_source_pool,
    #     x_elite_pool, y_elite_pool,
    #     label_weight=10.0,        # 偏向分数的权重
    #     max_cost_threshold=50.0,  # 清洗阈值 (建议根据第一次运行的打印结果调整)
    #     device=config.DEVICE
    # )
    # 注意参数顺序：x_low / y_low 对应 Source（差的样本），x_high / y_high 对应 Elite（高分样本）
    x_src, x_tgt, y_tgt, y_src = local_robust_pairing(
        x_source_pool,  # x_low  (source / bottom)
        x_elite_pool,   # x_high (elite / top)
        y_elite_pool,   # y_high
        y_source_pool,  # y_low
        k=config.TOP_K_NEIGHBORS,
        max_dist_threshold=5.0,
    )
    
    # 封装
    dataset = TensorDataset(x_src, x_tgt, y_tgt, y_src)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 将离线全集 x / y 从 numpy 转为 CPU 上的 Tensor，方便后续直接使用 .to() 等张量操作
    offline_x = torch.tensor(x_norm, dtype=torch.float32)
    offline_y = torch.tensor(y_norm, dtype=torch.float32)
    
    return (
        loader,
        task,
        offline_x,
        offline_y,
        torch.tensor(mean_x).float(),
        torch.tensor(std_x).float(),
        torch.tensor(mean_y).float(),
        torch.tensor(std_y).float(),
    )