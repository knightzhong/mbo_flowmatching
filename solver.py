import torch

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, x0, x1, y_high, y_low):
        """
        CFG 训练：随机丢弃 y_high
        """
        x0, x1 = x0.to(self.device), x1.to(self.device)
        y_high = y_high.to(self.device)
        y_low = y_low.to(self.device)
        B = x0.shape[0]
        
        t = torch.rand(B, 1, device=self.device)
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0
        
        # === CFG 训练: 15% 概率生成 Mask ===
        # p_uncond = 0.15 是经验值
        drop_mask = (torch.rand(B, 1, device=self.device) < 0.15)
        
        # 传入 Mask
        v_pred = self.model(x_t, t, y_high, y_low, drop_mask=drop_mask)
        
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x_start, y_high, y_low, steps=100, guidance_scale=3.0):
        """
        CFG 采样: v = v_uncond + scale * (v_cond - v_uncond)
        注意：现在的 guidance_scale 含义变了，不再是梯度步长，而是 CFG 强度。
        通常范围在 1.0 (无引导) 到 5.0 (强引导) 之间。
        """
        self.model.eval()
        x_current = x_start.clone().to(self.device)
        B = x_current.shape[0]
        
        # 准备条件
        if isinstance(y_high, float): y_high = torch.full((B, 1), y_high, device=self.device)
        else: y_high = y_high.to(self.device)
            
        if isinstance(y_low, float): y_low = torch.full((B, 1), y_low, device=self.device)
        else: y_low = y_low.to(self.device)

        # 预先创建 unconditional mask (全 True) 和 conditional mask (全 False)
        # 注意：在 eval 模式下，直接控制 mask 即可
        mask_uncond = torch.ones((B, 1), device=self.device, dtype=torch.bool)
        mask_cond = torch.zeros((B, 1), device=self.device, dtype=torch.bool)

        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # === CFG 采样步骤 ===
            # 为了效率，通常可以将 cond 和 uncond 拼接成 2*B 的 batch 一次跑完
            # 但为了代码清晰，这里写成两次 forward
            
            # 1. 有条件预测 (Conditional)
            v_cond = self.model(x_current, t, y_high, y_low, drop_mask=mask_cond)
            
            # 2. 无条件预测 (Unconditional)
            # 注意：即便这里传入了 y_high，因为 mask 是 True，模型内部会忽略它
            v_uncond = self.model(x_current, t, y_high, y_low, drop_mask=mask_uncond)
            
            # 3. 组合向量场
            # 公式: v_final = v_uncond + scale * (v_cond - v_uncond)
            v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # 更新状态
            x_current = x_current + v_final * dt
            
        return x_current