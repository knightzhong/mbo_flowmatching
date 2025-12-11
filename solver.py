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
            
            # # ... (guidance 逻辑保持不变，如果 scale=0 则不影响) ...
            # if guidance_fn is not None and guidance_scale > 0:
            #     # 必须开启梯度计算，因为我们需要对 x 求导
            #     with torch.enable_grad():
            #         # x_in 需要 detach 出来，避免计算图无限增长，并设置 requires_grad
            #         x_in = x_current.detach().requires_grad_(True)
                    
            #         # 预测分数
            #         score = guidance_fn(x_in)
                    
            #         # 计算梯度：我们要最大化 sum(score)，求 d(score)/dx
            #         grad = torch.autograd.grad(score.sum(), x_in)[0]
                
            #     # === 关键技巧：梯度归一化 ===
            #     # 原始梯度可能非常大或非常小，直接加会破坏 ODE 的稳定性。
            #     # 我们通常把梯度投影到与 v 相同的模长，或者限制其模长。
                
            #     # 计算模长 (Norm)
            #     v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
            #     grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-8
                
            #     # 策略：保持梯度的方向，但将其模长缩放到与当前速度 v 一致
            #     # 这样 guidance_scale 就变成了“混合比例”
            #     scaled_grad = (grad / grad_norm) * v_norm
                
            #     # 更新速度：原速度 + 引导分量
            #     v = v + scaled_grad * guidance_scale
            # 2. 计算引导梯度 (Clean Guidance 策略)
            if guidance_fn is not None and guidance_scale > 0:
                if i == 0:
                    print("iteration, guidance_scale", i, guidance_scale)
                with torch.enable_grad():
                    x_in = x_current.detach().requires_grad_(True)
                    
                    # === 关键修改：预测终点 x1 (Clean Data) ===
                    # 我们需要重新计算一遍 v (带梯度)，以便梯度能回传到 x_in
                    # 注意：为了节省计算，也可以直接对 x_in 计算 v，但这需要模型支持梯度回传
                    v_grad = self.model(x_in, t, y_high_in, y_low_in)
                    
                    # 根据公式 x1 = xt + (1-t)*v 推算终点
                    # 这里的 x_clean 就是去噪后的干净样本（近似 One-Hot）
                    x_clean = x_in + (1.0 - t_val) * v_grad
                    
                    # 让 Proxy 对“预测的终点”打分，而不是对“当前的混合态”打分
                    score = guidance_fn(x_clean)
                    
                    # 求导：我们希望 x_clean 分数更高，从而通过链式法则更新 x_in
                    grad = torch.autograd.grad(score.sum(), x_in)[0]
                
                # === 梯度归一化 (保持不变) ===
                # v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
                # grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-8
                # scaled_grad = (grad / grad_norm) * v_norm
                
                # 更新速度
                v = v + grad * guidance_scale

            x_current = x_current + v * dt
            
        return x_current