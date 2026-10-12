import timeit
import torch
import torch.nn as nn
from cs336_basics.model import RMSNorm
from cs336_systems.rmsnorm_triton import RMSNormAutogradTriton

device = "cuda" if torch.cuda.is_available() else "cpu"

# 采用适度的数据量，既能压榨性能，又不会撑爆 8GB 显存
num_rows = 50000
hidden_dims = [1024, 2048, 4096]
num_trials = 50

print(f"{'Hidden Dim':<10} | {'LayerNorm (ms)':<16} | {'PyTorch RMS (ms)':<18} | {'Triton Fused (ms)':<18} | {'Triton 加速比':<12}")
print("-" * 85)

for h in hidden_dims:
    x = torch.randn(num_rows, h, device=device)
    w = torch.randn(h, device=device)

    # 1. PyTorch 官方 LayerNorm
    ln = nn.LayerNorm(h).to(device)
    # 2. Assignment 1 原生 RMSNorm
    rms_pt = RMSNorm(hidden_size=h).to(device)
    # 3. 你手写的 Triton 算子
    f_triton = RMSNormAutogradTriton.apply

    # Warmup
    for _ in range(5):
        _ = ln(x)
        _ = rms_pt(x)
        _ = f_triton(x, w)
    torch.cuda.synchronize()

    # 测 LayerNorm
    t0 = timeit.default_timer()
    for _ in range(num_trials):
        _ = ln(x)
    torch.cuda.synchronize()
    t_ln = (timeit.default_timer() - t0) * 1000.0 / num_trials

    # 测 PyTorch 原版 RMSNorm
    t0 = timeit.default_timer()
    for _ in range(num_trials):
        _ = rms_pt(x)
    torch.cuda.synchronize()
    t_pt = (timeit.default_timer() - t0) * 1000.0 / num_trials

    # 测 你的 Triton RMSNorm
    t0 = timeit.default_timer()
    for _ in range(num_trials):
        _ = f_triton(x, w)
    torch.cuda.synchronize()
    t_triton = (timeit.default_timer() - t0) * 1000.0 / num_trials

    speedup = t_pt / t_triton
    print(f"{h:<10} | {t_ln:<16.4f} | {t_pt:<18.4f} | {t_triton:<18.4f} | {speedup:<12.2f}x")