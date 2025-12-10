import torch

class ConditionalFlowMatching:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        
    def compute_loss(self, x0, x1):
        """
        训练阶段: 计算 CFM Loss
        x0: 低分样本 (Source)
        x1: 高分样本 (Target) - 也就是 x0 的'导师'
        """
        x0, x1 = x0.to(self.device), x1.to(self.device)
        B = x0.shape[0]
        
        # 1. 采样时间 t ~ U[0, 1]
        t = torch.rand(B, 1, device=self.device)
        
        # 2. 构建插值路径 (Optimal Transport Path / Linear Path)
        # psi_t(x) = (1-t)x0 + t*x1
        # 在 TFBind8 这种 One-hot 空间，这叫 "Continuous Relaxation"
        x_t = (1 - t) * x0 + t * x1
        
        # 3. 理想速度 (Target Velocity)
        # u_t = x1 - x0
        u_t = x1 - x0
        
        # 4. 模型预测
        v_pred = self.model(x_t, t)
        
        # 5. 回归 Loss (MSE)
        loss = torch.mean((v_pred - u_t) ** 2)
        return loss

    # @torch.no_grad()
    # def sample(self, x_start, steps=100):
    #     """
    #     推理阶段: 求解 ODE
    #     x_start: 初始低分样本
    #     """
    #     self.model.eval()
    #     x_current = x_start.clone().to(self.device)
    #     B = x_current.shape[0]
    #     dt = 1.0 / steps
        
    #     # 简单的欧拉积分器 (Euler Solver)
    #     for i in range(steps):
    #         t_val = i / steps
    #         t = torch.full((B, 1), t_val, device=self.device)
            
    #         # v = model(x_t, t)
    #         v = self.model(x_current, t)
            
    #         # x_{t+1} = x_t + v * dt
    #         x_current = x_current + v * dt
            
    #     return x_current
    # solver.py 中 ConditionalFlowMatching 类的新 sample 方法

    @torch.no_grad()
    def sample(self, x_start, steps=100, guidance_fn=None, guidance_scale=1.0):
        """
        推理阶段: 求解 ODE (支持梯度引导)
        """
        self.model.eval()
        x_current = x_start.clone().to(self.device)
        B = x_current.shape[0]
        dt = 1.0 / steps
        
        # 启用梯度计算 (为了计算 guidance)
        for i in range(steps):
            t_val = i / steps
            t = torch.full((B, 1), t_val, device=self.device)
            
            # 1. 计算流模型的速度 (Base Velocity)
            v = self.model(x_current, t)
            
            # 2. 计算引导梯度 (Guidance Velocity)
            # 我们希望 x 往分数高的地方走，所以加上 alpha * grad(score)
            if guidance_fn is not None:
                # 临时开启梯度
                with torch.enable_grad():
                    x_in = x_current.detach().requires_grad_(True)
                    score = guidance_fn(x_in)
                    # 我们想要 maximize score -> gradient ascent
                    grad = torch.autograd.grad(score.sum(), x_in)[0]
                
                # 叠加梯度： v_total = v_flow + scale * v_grad
                # 注意：grad 需要稍微归一化一下，防止步子太大
                grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-8
                v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
                
                # 这种归一化技巧能保证引导不会破坏流形结构
                scaled_grad = grad / grad_norm * v_norm 
                
                v = v + guidance_scale * scaled_grad
            
            # 3. 更新位置
            x_current = x_current + v * dt
        # for i in range(steps):
        #     t_val = i / steps
        #     t = torch.full((B, 1), t_val, device=self.device)
            
        #     # 1. Flow Velocity
        #     v = self.model(x_current, t)
            
        #     # 2. Guidance Velocity (带动态衰减!)
        #     if guidance_fn is not None:
        #         with torch.enable_grad():
        #             x_in = x_current.detach().requires_grad_(True)
        #             score = guidance_fn(x_in)
        #             grad = torch.autograd.grad(score.sum(), x_in)[0]
                
        #         # === 核心修改: 线性衰减 ===
        #         # 让强大的 Flow 在后期接管比赛
        #         current_scale = guidance_scale * (1.0 - t_val)
                
        #         grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-8
        #         v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
        #         scaled_grad = grad / grad_norm * v_norm 
                
        #         v = v + current_scale * scaled_grad
            
        #     # 3. Step
        #     x_current = x_current + v * dt
            
        # return x_current
            
        return x_current