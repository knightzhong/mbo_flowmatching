import torch
import torch.nn as nn

class VectorFieldNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # === 修改：使用 y_high 和 y_low（与 ROOT 一致）===
        self.y_high_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.y_low_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 状态嵌入
        self.x_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 联合处理
        self.joint_mlp = nn.Sequential(
            # 输入维度变为 4 * hidden_dim (x + t + y_high + y_low)
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, y_high, y_low):  # 接收 y_high 和 y_low
        if x.dim() > 2: x = x.view(x.size(0), -1)
        if t.dim() == 1: t = t.view(-1, 1)
        if y_high.dim() == 1: y_high = y_high.view(-1, 1)  # 确保 y_high 是 (B, 1)
        if y_low.dim() == 1: y_low = y_low.view(-1, 1)  # 确保 y_low 是 (B, 1)
            
        t_emb = self.time_mlp(t)
        x_emb = self.x_mlp(x)
        y_high_emb = self.y_high_mlp(y_high)  # 处理 y_high
        y_low_emb = self.y_low_mlp(y_low)  # 处理 y_low
        
        # 拼接 (Concat) 比相加更能保留条件信息（与 ROOT 一致）
        h = torch.cat([x_emb, t_emb, y_high_emb, y_low_emb], dim=-1)
        v = self.joint_mlp(h)
        return v

# models.py (追加在后面)

class ScoreProxy(nn.Module):
    """
    一个简单的预测器：输入 DNA 序列 (B, 32)，输出分数 (B, 1)
    用于在推理时提供梯度引导。
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # 简单的 MLP 或 CNN 结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, 32) -> logits
        return self.net(x)

