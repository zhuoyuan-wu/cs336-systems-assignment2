import torch

def rmsnorm_backward_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    计算 RMSNorm 对权重 g 的梯度。
    grad_output: (*, H)
    x: (*, H)
    g: (H,)
    """
    # 1. 计算 RMS(x) = sqrt(mean(x^2) + eps)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    # 2. x_norm = x / RMS(x)
    x_norm = x / rms
    # 3. 对除最后一个维度以外的所有维度求和
    grad_g = grad_output * x_norm
    dims_to_sum = tuple(range(grad_g.ndim - 1))
    if dims_to_sum:
        grad_g = grad_g.sum(dim=dims_to_sum)
    return grad_g

def rmsnorm_backward_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    计算 RMSNorm 对输入 x 的梯度。
    grad_output: (*, H)
    x: (*, H)
    g: (H,)
    """
    H = x.shape[-1]
    # 1. 计算 RMS(x)
    var = torch.mean(x ** 2, dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    
    # 2. grad_y * g
    dy_g = grad_output * g
    
    # 3. sum(dy * g * x)
    sum_dy_g_x = torch.sum(dy_g * x, dim=-1, keepdim=True)
    
    # 4. 根据推导公式合成 grad_x
    # grad_x = (dy * g) / rms - (x * sum(dy * g * x)) / (H * rms^3)
    grad_x = (dy_g / rms) - (x * sum_dy_g_x) / (H * (rms ** 3))
    return grad_x

class RMSNormAutogradPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        x_norm = x / rms
        out = x_norm * g
        ctx.save_for_backward(x, g)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, g = ctx.saved_tensors
        eps = ctx.eps
        grad_x = rmsnorm_backward_x(grad_output, x, g, eps)
        grad_g = rmsnorm_backward_g(grad_output, x, g, eps)
        # 对应 forward 的三个参数 (x, g, eps)，eps 不需要梯度返回 None
        return grad_x, grad_g, None