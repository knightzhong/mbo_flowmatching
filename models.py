import torch
import torch.nn as nn

class VectorFieldNet(nn.Module):
    """
    预测流场速度 v(x_t, t) 的网络。
    输入: x (batch, dim), t (batch, 1)
    输出: v (batch, dim)
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim) # 输出维度与输入相同
        )

    def forward(self, x, t):
        # 确保输入是 Flatten 的 (B, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 确保 t 是 (B, 1)
        if t.dim() == 1:
            t = t.view(-1, 1)
            
        t_emb = self.time_mlp(t)
        x_emb = self.x_mlp(x)
        
        # 简单的相加融合 (Conditioning)
        h = x_emb + t_emb 
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

