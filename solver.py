import torch

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, x0, x1, y_high, y_low):  # 接收 y_high 和 y_low（与 ROOT 一致）
        """
        x0: Source 样本（低价值）
        x1: Target 样本（高价值）
        y_high: Target 样本对应的分数（高价值分数）
        y_low: Source 样本对应的分数（低价值分数）
        """
        x0, x1 = x0.to(self.device), x1.to(self.device)
        y_high = y_high.to(self.device)
        y_low = y_low.to(self.device)
        B = x0.shape[0]
        
        t = torch.rand(B, 1, device=self.device)
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0
        
        # 传入 y_high 和 y_low 作为条件（与 ROOT 一致）
        v_pred = self.model(x_t, t, y_high, y_low)
        
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x_start, y_high, y_low, steps=100, guidance_fn=None, guidance_scale=1.0):
        """
        x_start: 起始样本（低价值）
        y_high: 目标分数（高价值分数）
        y_low: 起始分数（低价值分数）
        """
        self.model.eval()
        x_current = x_start.clone().to(self.device)
        B = x_current.shape[0]
        
        # 确保 y_high 和 y_low 维度正确
        if isinstance(y_high, float):
            y_high_in = torch.full((B, 1), y_high, device=self.device)
        else:
            y_high_in = y_high.to(self.device)
            
        if isinstance(y_low, float):
            y_low_in = torch.full((B, 1), y_low, device=self.device)
        else:
            y_low_in = y_low.to(self.device)

        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # 传入 y_high 和 y_low 作为条件（与 ROOT 一致）
            v = self.model(x_current, t, y_high_in, y_low_in)
            
            # ... (guidance 逻辑保持不变，如果 scale=0 则不影响) ...
            if guidance_fn is not None and guidance_scale > 0:
                 # ... (原有 guidance 代码) ...
                 pass

            x_current = x_current + v * dt
            
        return x_current