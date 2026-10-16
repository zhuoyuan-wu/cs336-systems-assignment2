import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, Dict, List

class ShardedOptimizer(Optimizer):
    """
    ZeRO-1 风格的优化器状态切分实现。
    
    核心机制:
    1. 每个 rank 仅维护 1/world_size 的参数优化器状态 (如 Adam 的 m, v 动量矩阵)，
       以此节省 rank 本地的优化器显存占用。
    2. step() 执行后，各 rank 仅更新自己负责的参数分片，并通过广播保持全局模型权重同步。
    """
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 初始化基类，在顶层维护完整的 param_groups 视图
        super().__init__(params, kwargs)

        # 按照 轮询切分 将参数均分到各个 rank
        # 仅将分配到当前 rank 的分片传递给底层的物理优化器实例
        local_param_groups = []
        for group in self.param_groups:
            sharded_params = [
                p for i, p in enumerate(group["params"]) 
                if i % self.world_size == self.rank
            ]
            local_group = {k: v for k, v in group.items() if k != "params"}
            local_group["params"] = sharded_params
            local_param_groups.append(local_group)

        # 底层本地优化器仅对当前 rank 负责的参数分配优化器状态
        self.local_optimizer = self.optimizer_cls(local_param_groups, **self.optimizer_kwargs)

    def step(self, closure=None, **kwargs):
        loss = None
        if closure is not None:
            loss = closure()

        # 1. 本地优化器仅更新当前 rank 分片内的参数
        self.local_optimizer.step(**kwargs)

        # 2. 通信阶段: 每个 rank 作为 src 将更新后的参数权重广播给其余 rank
        if dist.is_initialized() and self.world_size > 1:
            for group in self.param_groups:
                for i, p in enumerate(group["params"]):
                    owner_rank = i % self.world_size
                    dist.broadcast(p.data, src=owner_rank)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        # 同时清理顶层视图与本地优化器内部参数的梯度缓存
        super().zero_grad(set_to_none=set_to_none)
        self.local_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: Dict[str, Any]):
        # 支持动态添加参数组，并保持相同的切分映射规则
        super().add_param_group(param_group)
        if hasattr(self, "local_optimizer"):
            sharded_params = [
                p for i, p in enumerate(param_group["params"]) 
                if i % self.world_size == self.rank
            ]
            local_group = {k: v for k, v in param_group.items() if k != "params"}
            local_group["params"] = sharded_params
            self.local_optimizer.add_param_group(local_group)