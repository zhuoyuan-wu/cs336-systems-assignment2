import torch
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_basics.model import BasicsTransformerLM

# 使用 small 模型剖析
model = BasicsTransformerLM(
    vocab_size=10000, context_length=128, d_model=768, num_layers=12, num_heads=12, d_ff=3072
).cuda()
inputs = torch.randint(0, 10000, (16, 128), device="cuda")

# 1. 预热
out = model(inputs)
loss = out.sum()
loss.backward()
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# 2. 测量 Forward
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_fwd:
    with record_function("forward_pass"):
        out = model(inputs)
    torch.cuda.synchronize()

print("\n" + "=" * 70)
print("【Forward Pass 耗时前 10 的 GPU 算子 (单位: 毫秒)】")
print("=" * 70)
# 只筛选 CUDA 内核并按 CUDA 耗时排序
fwd_cuda_events = [e for e in prof_fwd.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA or e.cuda_time_total > 0]
fwd_cuda_sorted = sorted(fwd_cuda_events, key=lambda x: x.cuda_time_total, reverse=True)

for e in fwd_cuda_sorted[:10]:
    print(f"- {e.key[:35]:<35} | GPU耗时: {e.cuda_time_total/1000.0:8.2f} ms | 调用次数: {e.count}")

# 3. 测量 Forward + Backward
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_both:
    with record_function("forward_pass"):
        out = model(inputs)
    with record_function("backward_pass"):
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

print("\n" + "=" * 70)
print("【Forward + Backward 耗时前 10 的 GPU 算子 (单位: 毫秒)】")
print("=" * 70)
both_cuda_events = [e for e in prof_both.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA or e.cuda_time_total > 0]
both_cuda_sorted = sorted(both_cuda_events, key=lambda x: x.cuda_time_total, reverse=True)

for e in both_cuda_sorted[:10]:
    print(f"- {e.key[:35]:<35} | GPU耗时: {e.cuda_time_total/1000.0:8.2f} ms | 调用次数: {e.count}")