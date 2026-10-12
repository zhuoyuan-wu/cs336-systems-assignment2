import torch
import triton
import triton.language as tl


# 1. 前向传播 Triton Kernel (一次性加载到 SRAM，一锅算完)

@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,          # 输入 x 的指针
    Y_ptr,          # 输出 y 的指针
    W_ptr,          # 权重 g 的指针
    stride_x_row,   # x 每行跨步
    stride_y_row,   # y 每行跨步
    N_COLS,         # 隐藏维度 H
    eps,            # 防除零极小值
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    # 指针移动到当前行的起始位置
    row_x_ptr = X_ptr + row_idx * stride_x_row
    row_y_ptr = Y_ptr + row_idx * stride_y_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # 1. 一次性从显存读入片上缓存 SRAM
    x = tl.load(row_x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 2. 在片上缓存直接计算 RMS
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / N_COLS
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # 3. 归一化并乘权重
    y = x * rrms * w

    # 4. 只写回显存一次
    tl.store(row_y_ptr + cols, y, mask=mask)



# 2. 反向传播 Triton Kernel (在片上直接融合求 dx 和 partial_dw)

@triton.jit
def _rmsnorm_bwd_kernel(
    dY_ptr,         # 上游传回来的 grad_output
    X_ptr,          # 输入 x
    W_ptr,          # 权重 g
    dX_ptr,         # 输出: grad_x
    dW_part_ptr,    # 输出: 权重的部分梯度
    stride_row,     # 行跨步
    N_COLS,         # 隐藏维度 H
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offset = row_idx * stride_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # 一次性读入片上
    dy = tl.load(dY_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 重新计算 rrms
    mean_sq = tl.sum(x * x, axis=0) / N_COLS
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # 算子融合梯度计算
    x_norm = x * rrms
    # 保存 dw 的单行分量
    tl.store(dW_part_ptr + offset + cols, dy * x_norm, mask=mask)

    dy_w = dy * w
    sum_dy_w_x = tl.sum(dy_w * x, axis=0)

    dx = rrms * (dy_w - (x / N_COLS) * (rrms * rrms) * sum_dy_w_x)
    tl.store(dX_ptr + offset + cols, dx, mask=mask)



# 3. PyTorch Autograd 封装类

class RMSNormAutogradTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5):
        # 兼容任意维度 (*, H)
        orig_shape = x.shape
        H = orig_shape[-1]
        x_2d = x.contiguous().view(-1, H)
        g = g.contiguous()

        n_rows, n_cols = x_2d.shape
        y_2d = torch.empty_like(x_2d)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # 启动 Kernel: 每个 row 一个实例
        _rmsnorm_fwd_kernel[(n_rows,)](
            x_2d, y_2d, g,
            x_2d.stride(0), y_2d.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_2d, g)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, g = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        BLOCK_SIZE = ctx.BLOCK_SIZE
        eps = ctx.eps

        grad_output_2d = grad_output.contiguous().view(-1, x_2d.shape[-1])
        n_rows, n_cols = x_2d.shape

        dx_2d = torch.empty_like(x_2d)
        dw_part = torch.empty_like(x_2d)

        _rmsnorm_bwd_kernel[(n_rows,)](
            grad_output_2d, x_2d, g, dx_2d, dw_part,
            x_2d.stride(0), n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # 把所有行的权重梯度累加起来
        dw = dw_part.sum(dim=0)
        return dx_2d.view(orig_shape), dw, None