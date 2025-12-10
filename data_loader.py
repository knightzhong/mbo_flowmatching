import torch
import numpy as np
import design_bench
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# def get_design_bench_data(task_name):
#     """加载原始数据"""
#     print(f"Loading task: {task_name}...")
#     task = design_bench.make(task_name)
    
#     x = task.x
#     y = task.y
    
#     # 针对离散数据 (TFBind, RNA) 的处理
#     if task.is_discrete:
#         # Design-Bench 默认 x 是 indices (N, L)
#         # 我们需要将其转换为 Logits 或 One-Hot (N, L * Vocab)
#         # task.to_logits() 会返回 (N, L, Vocab)
#         x = task.to_logits(x) 
#         # Flatten 成 (N, D) 以便输入 MLP
#         x = x.reshape(x.shape[0], -1) 
        
#         # 使用 Softmax 归一化，使其像概率分布 (Continuous Relaxation)
#         # 注意：design-bench 的 logits 可能尚未归一化
#         x = torch.tensor(x, dtype=torch.float32)
#         # 这里的 x 已经是连续值了，可以直接用于 Flow Matching
#     else:
#         x = torch.tensor(x, dtype=torch.float32)
        
#     y = torch.tensor(y, dtype=torch.float32).view(-1)
#     return task, x, y
def get_design_bench_data(task_name):
    """加载原始数据 (强制 4-Channel One-Hot)"""
    print(f"Loading task: {task_name}...")
    task = design_bench.make(task_name)
    
    x = task.x
    y = task.y
    
    if task.is_discrete:
        # === 关键修改：手动 One-Hot，强制 Vocab=4 ===
        print("Processing discrete data manually...")
        
        # 1. 确保 x 是 Long 类型的索引 (N, L)
        if x.ndim == 3: 
            # 极少数情况如果是 (N, L, 1)
            x = x.squeeze(-1)
        
        x_indices = torch.tensor(x, dtype=torch.long)
        
        # 2. 强制转为 4 维 One-Hot (N, L, 4)
        # TFBind8/10 都是 DNA，Vocab 固定是 4
        vocab_size = 4
        x_onehot = F.one_hot(x_indices, num_classes=vocab_size).float()
        
        print(f"Manual One-Hot Shape: {x_onehot.shape}") # 应该是 (N, 8, 4)
        
        # 3. 展平 (N, 32)
        x = x_onehot.view(x_onehot.shape[0], -1)
        
    else:
        x = torch.tensor(x, dtype=torch.float32)
        
    y = torch.tensor(y, dtype=torch.float32).view(-1)
    
    # 注意：这里返回的 task 是干净的（处于 index 模式），
    # 它的 predict 方法期望的是离散索引 (N, L)，这正是我们要的。
    return task, x, y

def semantic_pairing(x_low, x_high, y_high, k=50):
    """
    SP-RFM 核心算法: Top-k 语义重采样
    对于每个 x_low:
    1. 在 x_high 中找到 k 个最近邻 (结构最像的)。
    2. 在这 k 个邻居中，选择分数 y 最高的那个作为目标配对。
    """
    print(f"Starting Semantic Pairing (Pools: Low={len(x_low)}, High={len(x_high)})...")
    
    # 为了计算距离，我们最好把数据归一化或者直接用 One-hot 距离
    # 对于 TFBind8 (32维), 直接欧氏距离就是 Hamming 距离的近似
    
    # 这是一个耗时操作，如果数据量大需要用 FAISS，但 TFBind8 数据量小，直接用广播即可
    # 计算距离矩阵 (Num_Low, Num_High)
    # 内存优化：分批计算
    x_source = []
    x_target = []
    
    batch_size = 100
    for i in range(0, len(x_low), batch_size):
        x_low_batch = x_low[i : i+batch_size] # (B, D)
        
        # 计算距离: ||a-b||^2
        dists = torch.cdist(x_low_batch, x_high, p=2) # (B, Num_High)
        
        # 1. 找到 Top-k 最近邻的索引
        # values: (B, k), indices: (B, k)
        _, neighbor_indices = torch.topk(dists, k=k, dim=1, largest=False)
        
        # 2. 在这些邻居中找分数最高的
        # 收集邻居的分数
        neighbor_scores = y_high[neighbor_indices] # (B, k)
        
        # 找到每个样本对应 k 个邻居中分数最高的那个的 index (0..k-1)
        best_in_k_idx = torch.argmax(neighbor_scores, dim=1) # (B,)
        
        # 映射回 x_high 的真实 index
        best_high_indices = neighbor_indices.gather(1, best_in_k_idx.unsqueeze(1)).squeeze()
        
        # 构建配对
        x_source.append(x_low_batch)
        x_target.append(x_high[best_high_indices])
        
    x_source = torch.cat(x_source, dim=0)
    x_target = torch.cat(x_target, dim=0)
    
    print(f"Pairing complete. Created {len(x_source)} pairs.")
    return x_source, x_target

# def build_paired_dataloader(config):
#     task, x, y = get_design_bench_data(config.TASK_NAME)
    
#     # 1. 划分 Low / High Pool
#     # 根据中位数或指定比例划分
#     threshold = torch.quantile(y, 1 - config.DATA_SPLIT_RATIO)
    
#     high_mask = y >= threshold
#     low_mask = y < threshold
    
#     x_high, y_high = x[high_mask], y[high_mask]
#     x_low, y_low = x[low_mask], y[low_mask]
    
#     # 2. 执行配对策略
#     x_src, x_tgt = semantic_pairing(x_low, x_high, y_high, k=config.TOP_K_NEIGHBORS)
    
#     # 3. 封装 DataLoader
#     dataset = TensorDataset(x_src, x_tgt)
#     loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
#     return loader, task, x_low, y_low # 返回 task 和 原始数据用于评估
# def build_paired_dataloader(config, proxy=None):
#     task, x, y = get_design_bench_data(config.TASK_NAME)
    
#     # === SP-RFM v2: 全员向精英学习 (All-to-Elite) ===
#     # 保持 Target 是 Elite (Top 5%)
#     high_threshold = torch.quantile(y, 0.95)

#     # 但 Source 只选 Bottom 50%
#     low_threshold = torch.quantile(y, 0.50)

#     low_mask = y <= low_threshold
#     high_mask = y >= high_threshold


#     # 1. 定义精英阈值 (Top 5%)
#     # 只有这部分样本有资格成为 "Target"
#     elite_ratio = 0.05
#     # high_threshold = torch.quantile(y, 1.0 - elite_ratio)
    
#     # # 2. 划分 Target Pool (精英池)
#     # high_mask = y >= high_threshold
#     x_high = x[high_mask]
#     y_high = y[high_mask]
    
#     # 3. 划分 Source Pool (普通池)
#     # 你的建议：为什么不把剩下的 95% 全都做 Source？
#     # 采纳：只要不是精英，都是待优化对象！
#     # 注意：为了严谨，我们可以把 Source 定义为 "y < high_threshold"
#     low_mask = y < high_threshold
#     x_low = x[low_mask]
    
#     print(f"Strategy: Optimize Bottom {100*(1-elite_ratio):.0f}% -> Top {100*elite_ratio:.0f}%")
#     print(f"Pools: Source (Low+Mid)={len(x_low)}, Target (Elite)={len(x_high)}")
    
#     # 4. 执行配对
#     # 这步会稍慢一点，因为 x_low 变多了，但数据利用率更高
#     x_src, x_tgt = semantic_pairing(x_low, x_high, y_high, k=config.TOP_K_NEIGHBORS)
    
#     dataset = TensorDataset(x_src, x_tgt)
#     loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
#     # 返回 x_low 作为测试集，我们从这里面挑最差的来评估提升幅度
#     return loader, task, x_low, y[low_mask]
def build_paired_dataloader(config, proxy=None):
    task, x, y = get_design_bench_data(config.TASK_NAME)
    
    # === 策略: 95% -> 5% ===
    
    # 1. Target: Top 5%
    high_threshold = torch.quantile(y, 0.5)
    high_mask = y >= high_threshold
    x_high = x[high_mask]
    y_high = y[high_mask]
    
    # 2. Source: All Others (Bottom 95%)
    # 既然 Flow 能力强，我们就让它处理所有非精英样本
    low_mask = y < high_threshold
    x_low = x[low_mask]
    
    print(f"Strategy: Optimize Bottom 50% -> Top 50%")
    print(f"Pools: Source={len(x_low)}, Target={len(x_high)}")
    
    # 配对
    x_src, x_tgt = semantic_pairing(x_low, x_high, y_high, k=config.TOP_K_NEIGHBORS)
    
    dataset = TensorDataset(x_src, x_tgt)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 返回 x_low (Bottom 50%) 作为测试集，我们从中选最差的
    return loader, task, x_low, y[low_mask]