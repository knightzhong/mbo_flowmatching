import torch

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, x0, x1, y1): # 接收 target score y1
        """
        x1: Target 样本
        y1: Target 样本对应的分数 (真实分数)
        """
        x0, x1, y1 = x0.to(self.device), x1.to(self.device), y1.to(self.device)
        B = x0.shape[0]
        
        t = torch.rand(B, 1, device=self.device)
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0
        
        # 传入 y1 作为条件：告诉模型，沿着这个方向走，能得到 y1 的分数
        v_pred = self.model(x_t, t, y1)
        
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x_start, y_target, steps=100, guidance_fn=None, guidance_scale=1.0):
        """
        y_target: 我们期望达到的目标分数 (B, 1)
        """
        self.model.eval()
        x_current = x_start.clone().to(self.device)
        B = x_current.shape[0]
        
        # 确保 y_target 维度正确
        if isinstance(y_target, float):
            y_in = torch.full((B, 1), y_target, device=self.device)
        else:
            y_in = y_target.to(self.device)

        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # 传入目标分数条件
            v = self.model(x_current, t, y_in)
            
            # ... (guidance 逻辑保持不变，如果 scale=0 则不影响) ...
            if guidance_fn is not None and guidance_scale > 0:
                 # ... (原有 guidance 代码) ...
                 pass

            x_current = x_current + v * dt
            
        return x_current