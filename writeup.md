# CS336 作业 2（系统与并行计算）实验报告 Write-up

## 1. 基准测试与性能剖析 (§2.1)
### 1.1 问题 (benchmarking_script)：端到端测速与预热机制
#### (b) 前向与反向传播耗时分析
使用自定义的测速脚本，在 NVIDIA RTX 4060 Laptop GPU（8GB 显存）上配置 1 步预热、5 步连续测量的实际运行数据：

Small 模型（12 层, $d_{model}=768$）：
- 前向传播（Forward）：约 $82.13\ \text{ms}$（标准差：$10.60\ \text{ms}$）
- 前向 + 反向传播（Both）：约 $222.27\ \text{ms}$（标准差：$4.51\ \text{ms}$）
- 推导反向传播（Backward）单独耗时：约 $140.14\ \text{ms}$（约为前向耗时的 $1.71\times$）

Medium 模型（24 层, $d_{model}=1024$）：
- 前向传播（Forward）：约 $205.08\ \text{ms}$（标准差：$2.94\ \text{ms}$）
- 前向 + 反向传播（Both）：约 $719.71\ \text{ms}$（标准差：$30.74\ \text{ms}$）
- 推导反向传播（Backward）单独耗时：约 $514.63\ \text{ms}$（约为前向耗时的 $2.51\times$）

稳定性分析： 经过 Warm-up 预热后，测量的标准差极小（变异系数均低于 $5\%$），表明硬件执行进入稳态，测量高度可信。

#### (c) 预热（Warm-up）的必要性
实验现象： 省略预热步时，第一次迭代的耗时显著飙升。以 Small 模型前向传播为例：第 1 步耗时高达 $370.15\ \text{ms}$，而后续平稳运行仅需约 $74.3\ \text{ms}$（慢了整整 $5\times$ 倍），导致整体统计的标准差从 $10.60\ \text{ms}$ 骤增至 $115.85\ \text{ms}$。

物理机制分析：
1. CUDA 驱动上下文初始化： 首次调用 GPU 需要建立驱动 Context 与通信句柄。
2. PyTorch 显存池分配（Caching Allocator）： 第一步中 PyTorch 需要向操作系统申请物理显存块并构建内存池，后续迭代则直接复用缓存。
3. 算子加载与延迟优化： 底层 cuBLAS 库与算子内核在第一次执行矩阵乘法（GEMM）时触发了 JIT 动态加载或启发式算法选择。

### 1.2 问题 (benchmarking_mixed_precision)：混合精度机制
#### (b) `torch.autocast("cuda", dtype=torch.float16)` 上下文下的数据类型
针对原权重为 FP32 的 ToyModel：
- autocast 上下文中的模型参数本身： `torch.float32` (权重参数不会在原位被修改，依然以高精度主权重形式保存)。
- 第一个前馈层 fc1 的输出： `torch.float16` (线性层的 GEMM 计算会自动降精度到 FP16 以调用 Tensor Cores 提速)。
- LayerNorm 层的输出： `torch.float32` (关键机制：归一化算子涉及方差计算，为防止下溢和数值不稳定，Autocast 会强制将其保留为 FP32)。
- 模型最终输出的 Logits (fc2)： `torch.float16`。
- 损失值 Loss： `torch.float32` (Loss 计算会自动提升回 FP32，确保反向传播梯度的初始精度)。
- 模型参数的梯度： `torch.float32` (积累梯度的精度严格与主权重参数的数据类型保持一致)。

#### (c) 为什么 LayerNorm 对 FP16 极度敏感？BF16 有何区别？
FP16 的局限： LayerNorm 计算中包含 $\sum x_i^2$ 操作。FP16 的最大表示上限仅为 $65504$。当隐藏维度 $H\ge1024$ 时，平方求和极其容易溢出到无穷大（$\text{Inf}$），直接导致后续梯度或激活值崩溃为 $\text{NaN}$。

BF16 的优势： 若采用 BF16，则无需刻意升回 FP32。因为 BF16 拥有与 FP32 完全相同的 8 位指数位（动态范围达到 $\approx3.4\times10^{38}$），天然避免了 FP16 常见的数值溢出风险。

### 1.3 问题 (pytorch_layernorm)：原生 RMSNorm 的访存瓶颈
固定输入行数为 $50,000$，在各特征维度（$H$）下的前向耗时对比：

| 隐藏维度 ($H$) | PyTorch 官方 LayerNorm | 自定义未融合 RMSNorm | 性能落后倍数 (Slowdown) |
| :---: | :---: | :---: | :---: |
| 1024 | $1.78\ \text{ms}$ | $5.97\ \text{ms}$ | $3.36\times$ |
| 2048 | $3.50\ \text{ms}$ | $12.18\ \text{ms}$ | $3.48\times$ |
| 4096 | $7.07\ \text{ms}$ | $24.82\ \text{ms}$ | $3.51\times$ |

原因分析： 虽然 RMSNorm 算法去掉了均值偏移，理论计算量少于 LayerNorm，但纯 PyTorch 实现是未融合的（Unfused），拆分成了 pow、mean、rsqrt、mul 4 个独立算子。每次运算都需要在 GPU 显存（HBM）和芯片缓存之间反复读写，属于极其严重的显存带宽瓶颈（Memory-bound）。相反，PyTorch 原生 LayerNorm 是底层 C++/CUDA 手写的融合算子，全局只需读写一次显存。

## 2. Triton 算子融合优化 (§2.2)
### 2.1 RMSNorm 反向传播公式推导
已知前向公式：
$$
y=\frac{x}{\text{RMS}(x)}\odot g
$$
其中 $\text{RMS}(x)=\sqrt{\frac{1}{H}\sum x_k^2+\epsilon}$，上游梯度为 $\nabla_y L$。

对权重 $g$ 的梯度：
$$
\nabla_g L=\sum_{\text{batch},\ \text{seq}}\left(\nabla_y L \odot \frac{x}{\text{RMS}(x)}\right)
$$

对输入 $x$ 的梯度：
应用全微分链式法则化简，得到向量化计算式：
$$
\nabla_x L=\frac{1}{\text{RMS}(x)}\left[\left(\nabla_y L \odot g\right)-\frac{x}{H\cdot\text{RMS}(x)^2}\sum_{k=1}^H\left((\nabla_y L)_k\cdot g_k\cdot x_k\right)\right]
$$

### 2.2 Triton 融合算子加速效果实测
使用手写的 Triton Fused RMSNorm 算子重新进行三方性能对决：

| 隐藏维度 ($H$) | PyTorch 官方 LayerNorm | 原版 PyTorch RMSNorm | Triton 融合 RMSNorm | 相对 PyTorch 原版的提速比 |
| :---: | :---: | :---: | :---: | :---: |
| 1024 | $2.07\ \text{ms}$ | $6.76\ \text{ms}$ | $1.99\ \text{ms}$ | $3.39\times$ |
| 2048 | $3.88\ \text{ms}$ | $13.32\ \text{ms}$ | $3.92\ \text{ms}$ | $3.40\times$ |
| 4096 | $7.68\ \text{ms}$ | $26.61\ \text{ms}$ | $7.79\ \text{ms}$ | $3.41\times$ |

实验结论： Triton 融合算子彻底消除了显存往返读写的开销，在所有维度上稳定取得 $3.4\times$ 倍提速，抹平了性能差距，并在 1024 维度下超越了 PyTorch 官方 C++ 实现。

## 3. 分布式并行与显存切分 (§3 & §4)
### 3.1 DDP 通信重叠与分桶机制
逐参数异步通信重叠（Overlap）： 通过为参数注册反向传播累积勾子（post_accumulate_grad_hook），每当网络后层某一参数梯度计算完成，立即在后台异步发起 `dist.all_reduce`，使通信与前序层的反向计算完全重叠（Hide Latency）。

梯度分桶（Bucketing）： 对于层数较深的模型，数以百计的小张量独立发起通信会引发巨大的算子发射与网络握手开销（Latency Overhead）。分桶将连续小参数打包为例如 $25\ \text{MB}$ 的大块整体传输，最大化跑满网络总线带宽。

### 3.2 问题 (ddp_bucketed_benchmarking) (b)：DDP 通信开销建模
设：
- $s$ = 模型总参数量大小（字节 Bytes）
- $w$ = 实际 All-Reduce 通信带宽（Bytes/s）
- $o$ = 单次通信调用的固定握手开销（秒）
- $n_b$ = 分桶数量；单桶大小 $b=\frac{s}{n_b}$

假设单桶的计算时间与通信时间能够大致重叠，则反向传播结束后未被隐藏的剩余通信开销可建模为：
$$
\text{Overhead}(n_b)=o\cdot n_b+\frac{s/n_b}{w}
$$

对分桶数 $n_b$ 求导并令导数为 0（$\frac{d}{dn_b}\text{Overhead}=0$）：
$$
o-\frac{s}{w\cdot n_b^2}=0 \implies n_b^*=\sqrt{\frac{s}{o\cdot w}}
$$

代入 $b^*=\frac{s}{n_b^*}$，推导出的**理论最优分桶大小（Optimal Bucket Size）**为：
$$
b^*=\sqrt{s\cdot o\cdot w}
$$

### 3.3 优化器状态切分 (ZeRO-1) 显存账本分析
#### (c) 我们的实现与官方 ZeRO-1 ($P_{os}$) 的差异
通信逻辑差异：
本作业实现：各 rank 在本地完成各自参数的局部更新后，使用 `dist.broadcast` 将参数更新广播至其余所有节点。单轮训练产生 $O(s)$ 规模的参数广播通信量。

官方 ZeRO-1：在反向传播阶段使用 Reduce-Scatter 汇聚梯度，各 rank 仅接收属于自己分片的梯度；参数更新后，使用 All-Gather 重新聚合参数。

显存占用差异：
本实现主要切分了优化器状态（Adam 的 $m$ 和 $v$ 动量），有效节省了 $8\times$ 参数量的显存占用；但各 rank 本地依然完整保留了所有参数的全尺寸梯度缓存（Gradients）。官方 ZeRO-1 则连同梯度的存储也实现了切分。
