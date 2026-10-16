import argparse
import timeit
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def run_step(model, inputs, mode="both"):
    with record_function("forward_pass"):
        out = model(inputs)
    
    if mode == "both":
        with record_function("backward_pass"):
            loss = out.sum()
            loss.backward()
        with record_function("optimizer_zero_grad"):
            model.zero_grad(set_to_none=True)

def profile_model(model_size="small", mode="forward", num_steps=3):
    cfg = MODEL_CONFIGS[model_size]
    device = "cuda"
    print(f"\n[PROFILER] 正在对 {model_size.upper()} 模型进行性能剖析 (模式: {mode})...")

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=128,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    ).to(device)

    inputs = torch.randint(0, 10000, (16, 128), device=device)

    # 1. 预热 1 步
    run_step(model, inputs, mode=mode)
    torch.cuda.synchronize()

    # 2. 开启 PyTorch Profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for _ in range(num_steps):
            run_step(model, inputs, mode=mode)
            torch.cuda.synchronize()
            prof.step()

    # 3. 导出用于生成火焰图的调用栈文件
    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")

    # 4. 打印耗时前 30 的算子表格（按 GPU 自身耗时排序）
    print("\n" + "=" * 80)
    print("Top CUDA Kernels (按 Self CUDA Time 排序):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="是否启动 PyTorch Profiler 详细剖析")
    parser.add_argument("--model_size", type=str, default="small", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "both"])
    args = parser.parse_args()

    if args.profile:
        profile_model(model_size=args.model_size, mode=args.mode)
    else:
        print(f"正在以普通测速模式运行 {args.model_size}...")
      
        from benchmark import run_step
        
        profile_model(model_size=args.model_size, mode=args.mode, num_steps=5)