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
        
        # === 新增：分数嵌入 ===
        self.y_mlp = nn.Sequential(
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
            # 输入维度变为 3 * hidden_dim (x + t + y)
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, y): # 接收 y
        if x.dim() > 2: x = x.view(x.size(0), -1)
        if t.dim() == 1: t = t.view(-1, 1)
        if y.dim() == 1: y = y.view(-1, 1) # 确保 y 是 (B, 1)
            
        t_emb = self.time_mlp(t)
        x_emb = self.x_mlp(x)
        y_emb = self.y_mlp(y) # 处理 y
        
        # 拼接 (Concat) 比相加更能保留条件信息
        h = torch.cat([x_emb, t_emb, y_emb], dim=-1)
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

