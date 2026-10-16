import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional, List

class DDPIndividualParameters(nn.Module):
    """
    逐参数异步通信的 DDP 容器 
    在反向传播时，每当一个参数的梯度算完，立即触发异步 All-Reduce。
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        # 1. 广播初始权重，确保所有卡起点一致
        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # 2. 注册反向传播 Hook
        self._register_hooks()

    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                # 当该参数梯度累积完成时自动调用
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(p):
            if not dist.is_initialized() or p.grad is None:
                return
            # 异步执行 All-Reduce
            handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, p.grad))
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有异步通信完成并取平均值"""
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for handle, grad in self.handles:
            handle.wait()
            grad.div_(world_size)
        self.handles.clear()


class DDPBucketed(nn.Module):
    """
    梯度分桶 (Bucketing) 的 DDP 容器、
    将小张量融合成大块 (Bucket) 进行通信，最大化利用带宽并降低通信开销。
    """
    def __init__(self, module: nn.Module, bucket_size_mb: Optional[float] = 25.0):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = (bucket_size_mb * 1024 * 1024) if bucket_size_mb is not None else float("inf")
        self.handles = []
        
        # 1. 广播初始权重
        if dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # 2. 按照参数倒序进行分桶 反向传播 从后往前
        self.buckets = []
        self._create_buckets()
        self._register_hooks()

    def _create_buckets(self):
        current_bucket = []
        current_bytes = 0

        # 反向遍历参数
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            p_bytes = param.numel() * param.element_size()
            if current_bytes + p_bytes > self.bucket_size_bytes and len(current_bucket) > 0:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bytes = 0

            current_bucket.append(param)
            current_bytes += p_bytes

        if current_bucket:
            self.buckets.append(current_bucket)

        # 为每个 bucket 建立反查状态
        self.param_to_bucket_idx = {}
        for b_idx, bucket in enumerate(self.buckets):
            for p in bucket:
                self.param_to_bucket_idx[p] = b_idx

        self.bucket_ready_count = [0] * len(self.buckets)

    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(p):
            if not dist.is_initialized():
                return
            b_idx = self.param_to_bucket_idx[p]
            self.bucket_ready_count[b_idx] += 1

            # 当前 bucket 内所有参数就绪，立即发射通信！
            if self.bucket_ready_count[b_idx] == len(self.buckets[b_idx]):
                bucket_params = self.buckets[b_idx]
                # 拼接扁平化张量 (Flatten)
                grads = [bp.grad for bp in bucket_params]
                flat_grad = torch._utils._flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, flat_grad, bucket_params))
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for handle, flat_grad, bucket_params in self.handles:
            handle.wait()
            flat_grad.div_(world_size)
            # 还原张量 (Unflatten)
            unflat_grads = torch._utils._unflatten_dense_tensors(flat_grad, [p.grad for p in bucket_params])
            for p, g in zip(bucket_params, unflat_grads):
                p.grad.copy_(g)

        self.handles.clear()
        self.bucket_ready_count = [0] * len(self.buckets)