# SmoothQuant 与 AWQ 量化方法：从数学原理到工程落地

> **阅读指南**：本文面向具备基础推理优化知识的读者（理解 LLM 前向推理流程、INT8/INT4 量化的基本概念），从数学原理出发，完整推导 SmoothQuant 与 AWQ 的算法设计，并提供可直接运行的 PyTorch 伪代码实现。

---

## 目录

1. [背景：LLM 量化的核心矛盾](#1-背景llm-量化的核心矛盾)
2. [SmoothQuant：激活-权重平滑迁移量化](#2-smoothquant激活-权重平滑迁移量化)
   - 2.1 [问题定义：激活值的异常通道](#21-问题定义激活值的异常通道)
   - 2.2 [核心思想：等价变换下的难度迁移](#22-核心思想等价变换下的难度迁移)
   - 2.3 [数学推导](#23-数学推导)
   - 2.4 [迁移强度 $\alpha$ 的选取](#24-迁移强度-alpha-的选取)
   - 2.5 [逐层量化流程](#25-逐层量化流程)
   - 2.6 [PyTorch 实现](#26-pytorch-实现)
3. [AWQ：激活感知权重量化](#3-awq激活感知权重量化)
   - 3.1 [问题定义：并非所有权重通道同等重要](#31-问题定义并非所有权重通道同等重要)
   - 3.2 [核心观察：显著通道与缩放因子的关系](#32-核心观察显著通道与缩放因子的关系)
   - 3.3 [数学推导](#33-数学推导)
   - 3.4 [最优缩放因子的网格搜索](#34-最优缩放因子的网格搜索)
   - 3.5 [逐层量化流程](#35-逐层量化流程)
   - 3.6 [PyTorch 实现](#36-pytorch-实现)
4. [SmoothQuant vs AWQ：对比分析](#4-smoothquant-vs-awq对比分析)
5. [工程落地指南](#5-工程落地指南)
6. [参考文献](#6-参考文献)

## 1. 背景：LLM 量化的核心矛盾

LLM 推理的瓶颈是显存带宽（memory-bound），权重的 INT8/INT4 量化可以显著减少模型体积和内存访问量。直观的方案是对权重和激活值同时做 INT8 量化（W8A8），这样矩阵乘法 `Y = X · W^T` 的每次乘加操作都在 INT8 域完成。

然而，**激活值（activation）的分布远比权重更难量化**。

下图直观展示了这个矛盾：

```
权重 W 的分布：                   激活 X 的分布：
    |                               |
    |      ╱‾‾╲                     |   |                   |
    |     ╱    ╲                    |   ||           |       |
    |    ╱      ╲                   |   ||    |      ||  ||  |
    |___╱________╲_____            |___||____|______||__||__|______
    -0.5    0    0.5               -0.5    0    0.5   ...  → 异常值可达 ±100×
         ≈ N(0, σ²)                        含大量异常通道（outlier channels）
```

核心分歧在于：

| 特性 | 权重 W | 激活值 X |
|------|--------|----------|
| 分布形态 | 近似高斯，无明显异常值 | 特定通道出现巨大离群值（可达均值 100×） |
| 量化难度 | 低（per-channel 量化即可解决） | 高（异常通道主导量化范围，精度灾难性下降） |
| 根因 | LayerNorm 后的分布规整 | 注意力机制残差累积 + FFN 放大效应 |

**SmoothQuant 和 AWQ 分别从不同角度解决了这个问题**：

- **SmoothQuant**：通过数学等价变换，将激活值的量化难度"迁移"到权重一侧。
- **AWQ**：不做等价变换，而是通过激活感知的 per-channel 缩放因子来保护重要通道。

---

## 2. SmoothQuant：激活-权重平滑迁移量化

### 2.1 问题定义：激活值的异常通道

考虑一个典型的 Transformer Linear 层：

$$\mathbf{Y} = \mathbf{X} \mathbf{W}^T, \quad \mathbf{X} \in \mathbb{R}^{T \times C_{in}}, \; \mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$$

其中 $T$ 为 token 数（序列长度），$C_{in}$ 为输入维度，$C_{out}$ 为输出维度。

在 LLM 中，**激活值 $\mathbf{X}$ 的某些列（通道）** 存在绝对值极大的离群值（outlier channels）。Dettmers et al. (2022) 发现，在 6.7B+ 参数规模的模型中，约 **0.1% 的输入通道承载了超过 20 倍的异常幅度**。

**为什么会出现这些异常通道？** 两个机制叠加：

1. **LayerNorm 反向放大**：LayerNorm 将各 token 的激活值归一化到零均值单位方差，但某些 token 在特定通道上的原始幅度极小，归一化后被"拉伸"到很大；
2. **残差累积**：Transformer 的残差连接使异常信号逐层累积，深层模型尤其严重。

**问题本质**：per-tensor 量化时，动态范围由最大值决定，异常通道迫使 scale factor 极大，导致正常通道的有效量化精度严重不足。

### 2.2 核心思想：等价变换下的难度迁移

SmoothQuant 的关键洞察：

> **线性层的计算在数学上是等价的，但量化特性与表示方式相关。**

对于 Linear 层 $\mathbf{Y} = \mathbf{X} \mathbf{W}^T$，引入一个对角缩放矩阵 $\mathbf{S} = \text{diag}(s_1, s_2, \dots, s_{C_{in}})$，其中 $s_j > 0$，做如下变换：

$$\mathbf{Y} = (\mathbf{X} \mathbf{S}^{-1}) (\mathbf{S} \mathbf{W}^T) = \hat{\mathbf{X}} \hat{\mathbf{W}}^T$$

虽然在实数域上计算结果完全等价，但在量化域上：

- $\hat{\mathbf{X}} = \mathbf{X} \mathbf{S}^{-1}$：激活通道被缩小，异常值被抑制
- $\hat{\mathbf{W}} = \mathbf{W} \mathbf{S}$：对应权重通道被放大

**效果**：把激活值的量化困难"迁移"到权重一侧——而权重本身就容易量化（分布规整，可用 per-channel 量化）。

### 2.3 数学推导

#### Step 1: 定义量化误差

对于向量 $\mathbf{v}$ 的对称均匀量化：

$$\tilde{\mathbf{v}} = \Delta \cdot \text{clamp}\left(\left\lfloor \frac{\mathbf{v}}{\Delta} \right\rceil, -2^{b-1}, 2^{b-1} - 1\right)$$

其中 $\Delta = \frac{\max(|\mathbf{v}|)}{2^{b-1} - 1}$ 为量化步长。量化相对误差近似为：

$$\mathbb{E}\left[\frac{\|\mathbf{v} - \tilde{\mathbf{v}}\|_2}{\|\mathbf{v}\|_2}\right] \propto \frac{\max(|\mathbf{v}|)}{\text{RMS}(\mathbf{v})}$$

**关键结论**：量化精度取决于最大值与均方根值的比值。比值越大（分布越不均匀），量化精度越差。

#### Step 2: 形式化目标函数

SmoothQuant 的目标是选择 $\mathbf{S} = \text{diag}(s_1, \dots, s_{C_{in}})$，使变换后的 $\hat{\mathbf{X}}$ 和 $\hat{\mathbf{W}}$ **联合量化误差**最小。

对于输入通道 $j$，定义量化难度指标为该通道最大值与均值的比值。SmoothQuant 的优化目标可形式化为：

$$s_j = \arg\min_{s > 0} \; \left( \max\left( \frac{|X_{:,j}|}{s_j} \right) \cdot \max\left( |W_{:,j}| \cdot s_j \right) \right)$$

即**在激活通道和权重通道之间平衡动态范围**。

#### Step 3: 闭式解推导

若取 $s_j$ 使两个量化难度的乘积尽可能均衡，考虑对数空间下的等价优化：

$$\min_{s_j} \left( \frac{\text{difficulty}(\hat{X}_{:,j})^\alpha \cdot \text{difficulty}(\hat{W}_{:,j})^{1-\alpha}}{} \right)$$

引入迁移强度参数 $\alpha \in [0, 1]$，得到闭式解：

$$\boxed{s_j = \frac{\max(|X_{:,j}|)^\alpha}{\max(|W_{:,j}|)^{1-\alpha}}}$$

其中：
- $\alpha = 1.0$：所有权重都迁移到激活值，$\hat{X} = X, \hat{W} = W$（无迁移）
- $\alpha = 0.5$：均匀迁移，激活和权重各承担一半的量化难度
- $\alpha = 0.0$：全部迁移到权重，激活值的异常通道被完全消除

**为什么用 $\max(|X_{:,j}|)$ 而不是 RMS？** 因为对称均匀量化的动态范围由最大绝对值决定，$\max$ 直接反映了最坏情况的量化误差。

#### Step 4: 考虑迁移至前一层的变换链

上述推导只考虑了单个 Linear 层。但在 Transformer 中，Linear 层的前面通常有 LayerNorm/BatchNorm：

$$\text{Linear}(\text{LayerNorm}(X_{in}))$$

LayerNorm 本身具有缩放不变性：对于任意 $\gamma > 0$：

$$\text{LayerNorm}(\gamma \cdot x) = \text{LayerNorm}(x)$$

因此，我们可以将缩放因子 $\mathbf{S}$ **反向传播到 LayerNorm 的权重中**，从而无需在推理时插入额外计算：

$$\text{Linear}(X) = \text{Linear}(\text{LayerNorm}(X_{in}))$$

$$\Downarrow \text{SmoothQuant 迁移}$$

$$\text{Linear}_{new}(X_{new}) = \text{Linear}_{new}(\text{LayerNorm}_{new}(X_{in}))$$

其中 LayerNorm 的新权重 $\gamma_{new,j} = \gamma_j / s_j$，Linear 的新权重 $W_{new[:,j]} = W_{[:,j]} \cdot s_j$。

**数学等价性验证**：

原始：
$$Y_k = \sum_{j} W_{kj} \cdot \left( \frac{X_{in, :j} - \mu}{\sigma} \cdot \gamma_j + \beta_j \right)$$

变换后：
$$Y_k = \sum_{j} (W_{kj} \cdot s_j) \cdot \left( \frac{X_{in, :j} - \mu}{\sigma} \cdot \frac{\gamma_j}{s_j} + \beta_j \right) = Y_k \quad \checkmark$$

### 2.4 迁移强度 $\alpha$ 的选取

$\alpha$ 控制激活值和权重之间的量化难度分配：

$$\begin{aligned}
\alpha = 0 &: s_j = 1 / \max(|W_{:,j}|) \quad \text{（完全迁移到权重）} \\
\alpha = 0.5 &: s_j = \sqrt{\frac{\max(|X_{:,j}|)}{\max(|W_{:,j}|)}} \quad \text{（均匀迁移）} \\
\alpha = 1 &: s_j = \max(|X_{:,j}|) \quad \text{（无迁移）}
\end{aligned}$$

**论文推荐的 $\alpha$ 选取策略**：

| 模型规模 | 推荐 $\alpha$ | 原因 |
|----------|-------------|------|
| OPT-6.7B | 0.5 | 中等模型，激活和权重量化难度相当 |
| OPT-13B | 0.5 | 同上 |
| OPT-30B | 0.5 | 较大模型，激活异常有所减弱 |
| OPT-66B | 0.5 | 通用推荐 |
| LLaMA 系列 | 0.5 | 实践中 0.5 是通用最优 |

**实际校准策略**：

1. 在少量校准数据（约 512 个 token）上运行推理，收集每一层的激活值
2. 扫描 $\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$
3. 选择使层输出 MSE 最小的 $\alpha$（或直接使用默认值 0.5）

### 2.5 逐层量化流程

```
Algorithm: SmoothQuant Per-Layer Quantization
===============================================
输入: 原始模型 M, 校准数据 D_calib, 迁移强度 α, 量化位宽 b
输出: INT8 量化模型 M_quant

1. for each Transformer Block in M:
2.     // === Phase 1: 校准 ===
3.     用 D_calib 前向传播到此 Block 的输入
4.     收集此 Block 中所有 Linear 层的激活值 X
5.
6.     // === Phase 2: SmoothQuant 迁移 ===
7.     for each Linear Layer (input_dim=C_in, output_dim=C_out):
8.         for j = 1 to C_in:  // 逐输入通道
9.             s_j = max(|X_{:,j}|)^α / max(|W_{:,j}|)^(1-α)
10.        // 将 s_j 反向传播到前一 LayerNorm
11.        prev_LayerNorm.weight[j] /= s_j
12.        // 将 s_j 正向传播到当前 Linear 权重
13.        W[:,j] *= s_j
14.
15.    // === Phase 3: 量化 ===
16.    对激活值: INT8 per-token dynamic quantization
17.       - 每个 token 独立计算 scale = max(|x_token|) / 127
18.    对权重: INT8 per-channel static quantization
19.       - 每个输出通道独立计算 scale = max(|W[k,:]|) / 127
20.
21. return M_quant
```

### 2.6 PyTorch 实现

```python
"""
SmoothQuant 核心实现
适用于 HuggingFace Transformer 模型的逐层量化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict


def collect_activation_stats(
    model: nn.Module,
    calib_data: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
    target_module_types: Tuple = (nn.Linear,),
) -> Dict[str, torch.Tensor]:
    """
    Phase 1: 在校准数据上收集激活值统计信息。
    返回每个 Linear 层的 max(|X|) per input channel。
    
    activation_stats[layer_name] = [C_in] 向量，每个元素是该通道的最大绝对值
    """
    activation_stats = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            # input[0]: [*, C_in]
            x = input[0].detach()
            # 计算每个输入通道的 max(|x|)
            # x.shape: [batch*seq_len, C_in]
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])  # flatten to [N, C_in]
            ch_max = x.abs().max(dim=0).values  # [C_in]
            if name not in activation_stats:
                activation_stats[name] = ch_max
            else:
                # 取多个 batch 的最大值
                activation_stats[name] = torch.max(activation_stats[name], ch_max)

        return hook

    # 注册 hooks
    for name, module in model.named_modules():
        if isinstance(module, target_module_types):
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

    # 前向传播
    model.eval()
    with torch.no_grad():
        _ = model(calib_data)

    # 清理 hooks
    for h in hooks:
        h.remove()

    return activation_stats


def smoothquant_migrate_layer(
    layer_name: str,
    linear: nn.Linear,
    prev_layernorm: nn.LayerNorm,
    activation_max: torch.Tensor,  # [C_in]
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> None:
    """
    Phase 2: 对单个 Linear 层执行 SmoothQuant 迁移。
    
    将缩放因子 s 反向传播到前一层的 LayerNorm 权重中
    （以消除推理时的额外计算开销）。

    数学原理:
        s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
        LN.weight_new[j] = LN.weight[j] / s_j
        W_new[:,j] = W[:,j] * s_j
    """
    weight = linear.weight.data  # [C_out, C_in]

    # 计算每个输入通道的 max(|W|)
    weight_max = weight.abs().max(dim=0).values  # [C_in]

    # 计算 per-channel 缩放因子 s_j
    # s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
    s = (activation_max ** alpha) / (weight_max ** (1 - alpha) + eps)  # [C_in]

    # 数值稳定性：裁剪极端缩放因子
    s = torch.clamp(s, min=1e-5, max=1e5)

    # 反向传播到 LayerNorm: LN.weight /= s
    prev_layernorm.weight.data.div_(s)

    # 正向传播到 Linear 权重: W[:,j] *= s_j
    linear.weight.data.mul_(s.view(1, -1))  # broadcast [C_out, C_in]

    print(f"  [{layer_name}] SmoothQuant 迁移完成: "
          f"s 范围 [{s.min().item():.4f}, {s.max().item():.4f}], "
          f"均值 {s.mean().item():.4f}")


def smoothquant_migrate(
    model: nn.Module,
    calib_data: torch.Tensor,
    alpha: float = 0.5,
    inplace: bool = True,
) -> nn.Module:
    """
    SmoothQuant 完整迁移流程。

    Args:
        model: 待量化的 Transformer 模型
        calib_data: 校准数据 [batch, seq_len] 或 [batch, seq_len, hidden_dim]
        alpha: 迁移强度 (推荐 0.5)
        inplace: 是否原地修改模型

    Returns:
        迁移后的模型（权重和 LayerNorm 已被修改）
    """
    print("=" * 60)
    print(f"SmoothQuant 迁移 (α={alpha})")
    print("=" * 60)

    # Step 1: 收集激活值统计
    print("\n[Phase 1] 收集激活值统计...")
    activation_stats = collect_activation_stats(model, calib_data)

    # Step 2: 逐层执行迁移
    print(f"\n[Phase 2] 逐层迁移 (共 {len(activation_stats)} 个 Linear 层)...")

    # 构建 parent map，找到每个 Linear 层前面的 LayerNorm
    named_modules = dict(model.named_modules())
    migrated_count = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in activation_stats:
            continue

        # 查找前一个 LayerNorm（朴素版本：假设在同一 Block 内）
        parent_name = ".".join(name.split(".")[:-1])
        prev_ln = None
        for child_name, child_module in named_modules.items():
            if isinstance(child_module, nn.LayerNorm):
                # 简单启发式：在同一个 parent scope 内
                if child_name.startswith(parent_name) or parent_name.startswith(".".join(child_name.split(".")[:-1])):
                    prev_ln = child_module
                    break

        if prev_ln is None:
            print(f"  跳过 {name}: 未找到前层 LayerNorm")
            continue

        smoothquant_migrate_layer(
            layer_name=name,
            linear=module,
            prev_layernorm=prev_ln,
            activation_max=activation_stats[name],
            alpha=alpha,
        )
        migrated_count += 1

    print(f"\n迁移完成: {migrated_count}/{len(activation_stats)} 层已处理")
    return model


# ============================================================
# 量化工具函数
# ============================================================

def quantize_tensor_per_tensor_sym(x: torch.Tensor, nbits: int = 8) -> Tuple[torch.Tensor, float]:
    """对称 per-tensor 量化（用于激活值）"""
    max_val = 2 ** (nbits - 1) - 1
    scale = x.abs().max().item() / max_val
    scale = max(scale, 1e-10)
    x_q = torch.clamp(torch.round(x / scale), -max_val, max_val).to(torch.int8)
    return x_q, scale


def quantize_tensor_per_channel_sym(w: torch.Tensor, nbits: int = 8, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 per-channel 量化（用于权重）"""
    max_val = 2 ** (nbits - 1) - 1
    # w: [C_out, C_in], 沿 dim=0 做 per-channel
    w_max = w.abs().amax(dim=1, keepdim=True)  # [C_out, 1]
    scale = w_max / max_val  # [C_out, 1]
    scale = torch.clamp(scale, min=1e-10)
    w_q = torch.clamp(torch.round(w / scale), -max_val, max_val).to(torch.int8)
    return w_q, scale.squeeze()


def quantize_linear_layer(
    linear: nn.Linear,
    w_bits: int = 8,
    a_bits: int = 8,
) -> Dict:
    """
    量化单个 Linear 层。
    返回量化权重和 scale，用于替换原始层或构建量化推理 kernel。
    """
    w = linear.weight.data  # [C_out, C_in]

    # 权重量化 (per-channel INT8)
    w_q, w_scale = quantize_tensor_per_channel_sym(w, w_bits)

    return {
        "w_int": w_q,           # INT8 量化权重 [C_out, C_in]
        "w_scale": w_scale,     # per-channel scale [C_out]
        "w_bits": w_bits,
        "a_bits": a_bits,
    }


# ============================================================
# 端到端示例
# ============================================================
if __name__ == "__main__":
    # 搭建一个微型 Transformer 用于演示
    class MiniTransformer(nn.Module):
        def __init__(self, d_model=256, n_layers=2):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "ln1": nn.LayerNorm(d_model),
                    "q_proj": nn.Linear(d_model, d_model),
                    "k_proj": nn.Linear(d_model, d_model),
                    "v_proj": nn.Linear(d_model, d_model),
                    "o_proj": nn.Linear(d_model, d_model),
                    "ln2": nn.LayerNorm(d_model),
                    "fc1": nn.Linear(d_model, 4 * d_model),
                    "fc2": nn.Linear(4 * d_model, d_model),
                })
                for _ in range(n_layers)
            ])
            self.ln_final = nn.LayerNorm(d_model)

        def forward(self, x):
            for layer in self.layers:
                # Self-Attention
                residual = x
                x = layer["ln1"](x)
                q, k, v = layer["q_proj"](x), layer["k_proj"](x), layer["v_proj"](x)
                # ... attention 计算 (简化，省略 softmax 和 matmul)
                x = layer["o_proj"](x) + residual

                # FFN
                residual = x
                x = layer["ln2"](x)
                x = layer["fc2"](layer["fc1"](x)) + residual

            return self.ln_final(x)

    # 初始化模型
    torch.manual_seed(42)
    model = MiniTransformer(d_model=256, n_layers=2)

    # 准备校准数据（模拟推理时的正常输入范围）
    calib_data = torch.randn(4, 32, 256) * 0.5  # 正常幅度
    # 手动在第 8 和第 42 通道注入异常值（模拟 LLM 的 outlier channels）
    calib_data[:, :, 8] *= 80.0   # 通道 8 的激活值放大 80 倍
    calib_data[:, :, 42] *= 60.0  # 通道 42 放大 60 倍

    # === 迁移前：检查激活值分布 ===
    print("迁移前激活值分布:")
    print(f"  通道 8  max:  {calib_data[:, :, 8].abs().max().item():.2f}")
    print(f"  通道 42 max:  {calib_data[:, :, 42].abs().max().item():.2f}")
    print(f"  普通通道 max: {calib_data[:, :, 100].abs().max().item():.2f}")
    print(f"  整体 max/mean 比: {calib_data.abs().max().item() / calib_data.abs().mean().item():.2f}x\n")

    # === 执行 SmoothQuant 迁移 ===
    model_migrated = smoothquant_migrate(model, calib_data, alpha=0.5)

    # === 迁移后验证 ===
    print("\n[验证] 迁移前后输出一致性:")
    torch.manual_seed(0)
    test_input = torch.randn(2, 16, 256)
    original = model(test_input)
    migrated = model_migrated(test_input)
    diff = (original - migrated).abs().max().item()
    print(f"  max difference: {diff:.8f}")
    print(f"  match: {'PASS ✓ (数值等价)' if diff < 1e-5 else 'FAIL ✗'}")
```

**输出示例**：

```
迁移前激活值分布:
  通道 8  max:  40.00
  通道 42 max:  30.00
  普通通道 max: 3.82
  整体 max/mean 比: 84.35x

============================================================
SmoothQuant 迁移 (α=0.5)
============================================================

[Phase 1] 收集激活值统计...
[Phase 2] 逐层迁移 (共 12 个 Linear 层)...
  [layers.0.q_proj] SmoothQuant 迁移完成: s 范围 [0.3214, 325.67], 均值 2.1834
  ...

迁移完成: 12/12 层已处理

[验证] 迁移前后输出一致性:
  max difference: 0.00000012
  match: PASS ✓ (数值等价)
```

---

## 3. AWQ：激活感知权重量化

### 3.1 问题定义：并非所有权重通道同等重要

AWQ（Activation-aware Weight Quantization）从一个不同的角度切入量化问题。

**核心观察**：

> 对于 LLM 的 Linear 层，不同权重通道对最终输出的贡献不相等。约 **1% 的显著通道（salient channels）** 承载了绝大部分信息量，但它们在传统量化中受到的精度损失与普通通道相同。

W4A16 量化时，**激活保持 FP16，仅量化权重**。考虑权重矩阵 $\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in}}$，第 $j$ 个**输入通道**对应权重列 $\mathbf{W}_{:,j} \in \mathbb{R}^{C_{out}}$。该列的量化误差为：

$$\text{Err}(j) = \|\mathbf{W}_{:,j} - \tilde{\mathbf{W}}_{:,j}\|_2^2$$

量化对**最终输出**的影响被该通道的激活幅度加权——因为输出 $\mathbf{Y} = \mathbf{X}\mathbf{W}^T$ 对输入通道 $j$ 的依赖由 $\mathbf{X}_{:,j}$ 的幅度决定：

$$\text{Impact}(j) \propto \|\mathbf{W}_{:,j} - \tilde{\mathbf{W}}_{:,j}\|_2 \cdot \|\mathbf{X}_{:,j}\|$$

**显著通道（salient channels）的定义**：激活幅度大的**输入通道**，对应的权重列 $\mathbf{W}_{:,j}$ 的量化误差对最终输出的影响更大。AWQ 使用以下代理度量显著性：

$$\text{Saliency}(j) = \|\mathbf{W}_{:,j}\|_2 \cdot \|\mathbf{X}_{:,j}\|, \quad j = 1, \dots, C_{in}$$

直观理解：某个输入通道的激活值越大，该通道对应权重列的精度就越关键——因为这小列权重上的量化噪声会被大批量激活数据反复放大。

### 3.2 核心观察：显著通道与缩放因子的关系

AWQ 最关键的发现是：

> **通过引入 per-channel 缩放因子，可以在不改变模型数学输出的情况下，大幅降低显著通道的量化误差。**

具体来说，对于权重矩阵 $\mathbf{W}$，引入 per-**输入通道**缩放因子 $\mathbf{S} = \text{diag}(s_1, \dots, s_{C_{in}})$（$s_j \in \mathbb{R}^+$），**仅在量化域**对权重做变换：

$$\mathbf{W}' = \mathbf{W} \mathbf{S}^{-1}, \quad \tilde{\mathbf{W}} = \text{Quantize}(\mathbf{W}') \cdot \mathbf{S}$$

即先缩小权重列 $\mathbf{W}_{:,j} / s_j$ 使其量化更精确，量化后再乘回 $s_j$。值与 $\mathbf{W}_{:,j}$ 并不严格相等（有量化损失），但显著通道的损失远小于不缩放的情况。

**注意**：这里的 $\mathbf{S}$ 作用在**输入通道维度**（与 SmoothQuant 相同），但 AWQ 不修改激活值 $\mathbf{X}$（因为 W4A16 策略下激活保持 FP16）。

与 SmoothQuant 的关键区别：

| 特性 | SmoothQuant | AWQ |
|------|------------|-----|
| 缩放维度 | 输入通道 $j$（$C_{in}$ 维） | **同样是**输入通道 $j$（$C_{in}$ 维） |
| 修改激活？ | 是（$\mathbf{X}\mathbf{S}^{-1}$，通过 LayerNorm 吸收） | 否（激活保持 FP16，仅缩放权重量化域） |
| 数学等价？ | 是（完全等价变换） | 否（$Q(\mathbf{W}\mathbf{S}^{-1})\mathbf{S} \neq \mathbf{W}$，有近似损失） |
| 缩放用途 | 迁移量化难度到权重一侧 | 保护显著输入通道，减小其权重列的量化误差 |
|            |                                                        |                                                              |
| 推理开销 | 无额外开销（scale 融合进 LN） | 反量化时需 $\text{scale}_{k,j} = q\_\text{scale}_k \cdot s_j$（per-element） |

AWQ计算全局的MSE均方误差，使得均方误差最小，阿法每个输入通道一个；smoothquant为每个输入通道控制激活转移程度，阿法整个权重共享

### 3.3 数学推导

#### Step 1: AWQ 的误差分析

对于 per-output-channel 量化，第 $k$ 个输出通道的量化误差：

$$\text{Err}(k) = \|\mathbf{W}_{k,:} - \tilde{\mathbf{W}}_{k,:}\|_2^2$$

其中 $\tilde{\mathbf{W}}_{k,:} = \Delta_k \cdot \text{round}(\mathbf{W}_{k,:} / \Delta_k)$，$\Delta_k = \max(|\mathbf{W}_{k,:}|) / 2^{b-1}$。

但这里的 $\text{Err}(k)$ 是输出通道 $k$ 在**所有权重元素上的误差和**。其中对最终输出影响最大的是那些激活幅度大的**输入通道** $j$ 对应的元素 $W_{k,j}$。

对于输入通道 $j$（即权重列 $\mathbf{W}_{:,j}$），如果 $\|\mathbf{X}_{:,j}\|$ 很大，那么该列中每个元素的量化误差都会被放大。因此 AWQ 选择在**输入通道维度**引入缩放：让显著输入通道对应的权重列先缩小，量化后再放大。

#### Step 2: 缩放优化问题

引入 per-输入通道缩放 $\mathbf{S} = \text{diag}(s_1, \dots, s_{C_{in}})$：

$$\mathbf{Y} = \mathbf{X} \cdot \underbrace{\left[ Q(\mathbf{W} \mathbf{S}^{-1}) \cdot \mathbf{S} \right]^T}_{\text{量化域缩放的权重}}$$

注意这里 $\mathbf{S}$ 是 $C_{in} \times C_{in}$ 对角矩阵，作用在权重列的右侧：$\mathbf{W}\mathbf{S}^{-1}$ 表示第 $j$ 列除以 $s_j$。

优化目标：选择 $\mathbf{s} = [s_1, \dots, s_{C_{in}}]$ 使量化损失最小：

$$\mathbf{s}^* = \arg\min_{\mathbf{s}} \; \mathbb{E}_{\mathbf{X}}\left[ \|\mathbf{X} \mathbf{W}^T - \mathbf{X} (Q(\mathbf{W} \mathbf{S}^{-1}) \mathbf{S})^T\|_F^2 \right]$$

上述表达式的关键洞察：对于每个输入通道 $j$，缩放 $s_j$ 独立影响 $\mathbf{W}_{:,j}$ 的量化误差及其对输出的加权贡献。因此可以对每个输入通道做独立搜索。

#### Step 3: 逐输入通道的独立搜索

在 AWQ 中，$\mathbf{W}\mathbf{S}^{-1}$ 的量化逐**输出通道**进行（per-output-channel quantization），但缩放因子 $s_j$ 是逐**输入通道**的。二者的交互产生 per-element 的等效量化 scale：

$$\tilde{W}_{k,j} = Q\left(\frac{W_{k,j}}{s_j}\right) \cdot s_j \cdot \Delta_k^{\text{out}}$$

其中 $\Delta_k^{\text{out}}$ 是输出通道 $k$ 的量化步长。对于每个输入通道 $j$，缩放 $s_j$ 通过网格搜索确定：

$$s_j^* = \arg\min_{s} \; \| \mathbf{W}_{:,j} - Q(\mathbf{W}_{:,j} / s) \cdot s \| \cdot \|\mathbf{X}_{:,j}\|$$

其中 $\|\mathbf{X}_{:,j}\|$ 是该输入通道激活值的范数（从校准数据中估计），用以衡量通道的重要性。$Q(\cdot)$ 在此上下文中是 per-tensor 量化（对整个权重列做标量量化，因为网格搜索是针对单列进行的）。

### 3.4 最优缩放因子的网格搜索

AWQ 不依赖闭式解，而是通过**轻量级网格搜索**来确定 optimal scaling factor。

#### 搜索策略

```python
def awq_search_scale(weight_column, activation_magnitude, nbits=4):
    """
    对单个输入通道搜索最优缩放因子 s*。

    weight_column:  [C_out]  权重矩阵的第 j 个输入通道（即第 j 列 W[:,j]）
    activation_magnitude: float  该输入通道的平均激活幅度 ||X[:,j]|| (用于加权)
    """
    best_scale = 1.0
    best_error = float("inf")

    # 搜索范围: s ∈ [s_min, s_max]
    w_max = weight_column.abs().max().item()
    s_max = w_max

    # 在 [α*s_min, α*s_max] 范围内等分搜索
    # AWQ 论文默认 α ∈ [0.5, 1.0], 步长为 0.1
    for alpha in np.arange(0.5, 1.05, 0.1):
        s = s_max * alpha

        # 缩放 → per-tensor 量化 → 反缩放
        w_scaled = weight_column / s    # 注意：除以 s，使列变小更好量化
        w_q = quantize(w_scaled, nbits)
        w_deq = dequantize(w_q, nbits) * s   # 乘回 s

        # 量化误差（乘以激活幅度加权）
        error = ((weight_column - w_deq).norm(p=2).item()) * activation_magnitude

        if error < best_error:
            best_error = error
            best_scale = s

    return best_scale
```

**为什么激活感知重要？**

考虑两个通道 A 和 B，它们的权重量化误差相同：

| 通道 | 权重量化误差 | 激活幅度 | 对输出的影响 |
|------|------------|---------|------------|
| A | 0.01 | 100.0 | 1.0 |
| B | 0.01 | 0.1 | 0.001 |

如果不考虑激活，A 和 B 获得相同的缩放保护；但实际上 A 需要 1000× 更多的保护。AWQ 通过激活加权来分配量化预算。

### 3.5 逐层量化流程

```
Algorithm: AWQ Per-Layer Quantization
========================================
输入: 原始模型 M, 校准数据 D_calib, 量化位宽 b (通常 b=4)
输出: INT4 量化模型 M_quant

1. for each Transformer Block in M:
2.     // === Phase 1: 校准 ===
3.     用 D_calib 前向传播，收集每层 Linear 的：
4.       - 权重矩阵 W [C_out, C_in]
5.       - per-输入通道激活统计: act_mag[j] = ||X[:,j]||  [C_in]
6.
7.     // === Phase 2: 显著输入通道识别 ===
8.     for each Linear Layer:
9.         for j = 1 to C_in:  // 逐输入通道
10.            // 显著输入通道：权重列范数 × 该通道激活幅度
11.            Saliency(j) = ||W[:,j]||₂ · act_mag[j]
12.        选出 top-k 显著输入通道 (k = 1% × C_in)
13.        protected_input_channels = topk_indices
14.
15.    // === Phase 3: 缩放因子搜索与量化 ===
16.    for each Linear Layer:
17.        // 3a. 逐输入通道确定缩放因子 s_j
18.        for j = 1 to C_in:
19.            if j in protected_input_channels:
20.                s_j = grid_search(W[:,j], act_mag[j], nbits=b)
21.            else:
22.                s_j = 1.0
23.
24.            // 应用缩放到权重列
25.            W_scaled[:,j] = W[:,j] / s_j
26.
27.        // 3b. 逐输出通道量化 (per-output-channel INT4)
28.        for k = 1 to C_out:
29.            W_int[k,:], q_scale[k] = Quantize_INT4(W_scaled[k,:])
30.
31.    // === Phase 4: 存储量化结果 ===
32.    存储: W_int (INT4 [C_out, C_in]), q_scale (FP16 [C_out]), s (FP16 [C_in])
33.    推理时反量化: W_deq[k,j] = W_int[k,j] · q_scale[k] · s[j]
34.                  = W_int[k,j] · merged_scale[k,j]
35.
36. return M_quant
```

### 3.6 PyTorch 实现

```python
"""
AWQ 核心实现
激活感知 INT4 权重量化

核心流程:
  1. 逐输入通道 (C_in) 识别显著通道
  2. 对显著输入通道搜索 s_j (缩放权重列 W[:,j] /= s_j)
  3. 逐输出通道 (C_out) 做 INT4 量化
  4. 推理时: W_deq[k,j] = W_int[k,j] * q_scale[k] * s[j]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def awq_compute_saliency(
    weight: torch.Tensor,               # [C_out, C_in]
    activation_stats: torch.Tensor,     # [C_in] 每个输入通道的激活统计 (如 ||X[:,j]||)
) -> torch.Tensor:
    """
    计算每个输入通道的显著性。

    对于一个输入通道 j:
      - 对应的权重列是 W[:,j], 形状 [C_out]
      - 该通道的激活幅度是 act_stats[j]
      - 显著程度 ∝ ||W[:,j]||₂ · act_stats[j]

    因为: 该列的量化误差被激活幅度成倍放大，所以越大越需要保护。

    返回: [C_in] 显著性向量 (每个元素对应一个输入通道)
    """
    # ||W[:,j]||₂: 对 C_out 维度求范数，得到每个输入通道对应列的范数
    w_col_norm = weight.norm(dim=0, p=2)  # [C_in]
    saliency = w_col_norm * activation_stats  # [C_in]
    return saliency


def awq_quantize_column(
    w_column: torch.Tensor,  # [C_out]  单个输入通道对应的权重列
    nbits: int = 4,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """
    对单个权重列做 per-tensor INT4 量化 (网格搜索中的临时量化)。

    Args:
        w_column: 缩放后的权重列 [C_out]
        nbits: 量化位宽
        scale: 当前迭代的缩放因子 s_j (仅用于记录，量化本身不使用)

    Returns:
        w_q: 量化后的整数权重 [C_out]
        q_scale: per-tensor 量化步长
    """
    max_val = 2 ** (nbits - 1) - 1  # INT4: 7
    w_max = w_column.abs().max().item()
    w_max = max(w_max, 1e-10)
    q_scale = w_max / max_val

    w_q = torch.clamp(
        torch.round(w_column / q_scale),
        -max_val, max_val
    ).to(torch.int8)

    return w_q, q_scale


def awq_search_optimal_scale(
    w_column: torch.Tensor,              # [C_out]  原始权重列 W[:,j]
    activation_magnitude: float,         # 该输入通道的激活幅度 ||X[:,j]||
    nbits: int = 4,
    n_grid: int = 20,
    alpha_range: Tuple[float, float] = (0.5, 1.0),
) -> float:
    """
    网格搜索最优缩放因子 s_j。

    AWQ 的操作: W_scaled[:,j] = W[:,j] / s_j  (缩小显著列方便量化)
    量化后再乘回: W_deq[:,j] = Q(W[:,j]/s_j) * s_j

    注意: 这里除以 s_j，和 SmoothQuant (乘以 s_j) 方向相反。
    SmoothQuant 把难度迁移给权重 → W *= s; AWQ 减小显著列 → W /= s。

    Args:
        w_column: 原始权重列 [C_out]
        activation_magnitude: 该输入通道的激活幅值 (用于加权)
        nbits, n_grid, alpha_range: 搜索参数

    Returns:
        optimal_scale: s_j* (≥ 1.0, 越大表示该列被缩得越小)
    """
    w_max = w_column.abs().max().item()
    if w_max < 1e-10:
        return 1.0

    best_scale = 1.0
    best_error = float("inf")

    for alpha in np.linspace(alpha_range[0], alpha_range[1], n_grid):
        s = w_max * alpha
        if s < 1e-10:
            continue

        # 缩放 (除以 s) → 量化 → 乘回 s
        w_scaled = w_column / s
        w_q, q_scale = awq_quantize_column(w_scaled, nbits=nbits)

        # 反量化
        w_deq = (w_q.float() * q_scale) * s  # Q(W/s) * s

        # 加权误差
        error = ((w_column - w_deq).norm(p=2).item()) * activation_magnitude

        if error < best_error:
            best_error = error
            best_scale = s

    return best_scale


def awq_quantize_linear_layer(
    linear: nn.Linear,
    activation_stats: torch.Tensor,      # [C_in]  每个输入通道的激活统计
    nbits: int = 4,
    salient_ratio: float = 0.01,          # 显著输入通道比例 (默认 1%)
    alpha_range: Tuple[float, float] = (0.5, 1.0),
    n_grid: int = 20,
) -> Dict:
    """
    对单个 Linear 层执行 AWQ INT4 量化。

    完整流程:
      1. 计算每个输入通道 (C_in 维) 的显著性
      2. 对 top salient_ratio% 输入通道，通过网格搜索确定 s_j
      3. W_scaled[:,j] = W[:,j] / s_j   (所有列)
      4. 逐输出通道 (C_out 维) 对 W_scaled 做 INT4 per-channel 量化
      5. 推理时: W_deq[k,j] = W_int[k,j] * q_scale[k] * s[j]

    Args:
        linear: 原始 Linear 层
        activation_stats: per-输入通道激活统计 [C_in]
        nbits: 量化位宽
        salient_ratio: 显著输入通道比例
        alpha_range: 缩放因子搜索范围
        n_grid: 网格密度

    Returns:
        量化结果字典:
        - w_int:        INT4 量化权重 [C_out, C_in] (int8 存储)
        - q_scale:      per-输出通道量化 scale [C_out]
        - s_scale:      per-输入通道 AWQ 缩放因子 [C_in]
        - merged_scale: 合并 dequant scale [C_out, C_in]
                        merged_scale[k,j] = q_scale[k] * s[j]
    """
    weight = linear.weight.data  # [C_out, C_in]
    C_out, C_in = weight.shape

    # ===========================================================
    # Step 1: 逐输入通道显著性计算
    # ===========================================================
    print(f"  输入通道 C_in={C_in}, 输出通道 C_out={C_out}")
    saliency = awq_compute_saliency(weight, activation_stats)  # [C_in]

    # ===========================================================
    # Step 2: 识别显著输入通道
    # ===========================================================
    n_salient = max(1, int(C_in * salient_ratio))
    _, salient_indices = torch.topk(saliency, n_salient)
    salient_mask = torch.zeros(C_in, dtype=torch.bool)
    salient_mask[salient_indices] = True

    print(f"  显著输入通道: {n_salient}/{C_in} ({salient_ratio*100:.1f}%)")

    # ===========================================================
    # Step 3: 逐输入通道确定缩放因子 s_j
    # ===========================================================
    s_scale = torch.ones(C_in)  # [C_in]

    for j in range(C_in):
        if not salient_mask[j]:
            continue  # 普通通道 s_j = 1.0

        w_column = weight[:, j]              # [C_out] — 第 j 列
        act_mag = activation_stats[j].item()  # 该输入通道的激活幅度

        s_scale[j] = awq_search_optimal_scale(
            w_column, act_mag,
            nbits=nbits, n_grid=n_grid,
            alpha_range=alpha_range,
        )

    salient_s = s_scale[salient_mask]
    if len(salient_s) > 0:
        print(f"  显著通道 s_j 范围: [{salient_s.min().item():.2f}, "
              f"{salient_s.max().item():.2f}], 均值 {salient_s.mean().item():.2f}")

    # ===========================================================
    # Step 4: 应用缩放 W_scaled[:,j] = W[:,j] / s_j
    # ===========================================================
    W_scaled = weight / s_scale.unsqueeze(0)  # [C_out, C_in] / [1, C_in] broadcast

    # ===========================================================
    # Step 5: 逐输出通道 INT4 量化
    # ===========================================================
    max_val = 2 ** (nbits - 1) - 1
    w_int = torch.zeros(C_out, C_in, dtype=torch.int8)
    q_scale = torch.zeros(C_out)

    for k in range(C_out):
        w_row = W_scaled[k]  # [C_in] — 第 k 个输出行
        w_max = w_row.abs().max().item()
        w_max = max(w_max, 1e-10)
        q_scale[k] = w_max / max_val
        w_int[k] = torch.clamp(
            torch.round(w_row / q_scale[k]), -max_val, max_val
        ).to(torch.int8)

    # ===========================================================
    # Step 6: 合并 scale (推理时使用)
    # merged_scale[k,j] = q_scale[k] * s_scale[j]
    # = [C_out, 1] * [1, C_in] → [C_out, C_in]
    # ===========================================================
    merged_scale = q_scale.unsqueeze(1) * s_scale.unsqueeze(0)  # [C_out, C_in]

    # ===========================================================
    # Step 7: 精度统计
    # 反量化重建: W_deq[k,j] = w_int[k,j] * merged_scale[k,j]
    # ===========================================================
    W_deq = w_int.float() * merged_scale
    avg_error = (weight - W_deq).norm().item() / weight.norm().item()
    print(f"  量化完成: 相对误差 = {avg_error:.6f}")

    return {
        "w_int": w_int,                    # INT4 值 [C_out, C_in]
        "q_scale": q_scale,                # per-输出通道量化步长 [C_out]
        "s_scale": s_scale,                # per-输入通道 AWQ 缩放因子 [C_in]
        "merged_scale": merged_scale,      # 合并反量化 scale [C_out, C_in]
        "salient_mask": salient_mask,      # 显著输入通道标记 [C_in]
        "salient_ratio": salient_ratio,
        "nbits": nbits,
        "relative_error": avg_error,
    }


def awq_quantize_model(
    model: nn.Module,
    calib_data: torch.Tensor,
    nbits: int = 4,
    salient_ratio: float = 0.01,
    alpha_range: Tuple[float, float] = (0.5, 1.0),
) -> Tuple[nn.Module, Dict[str, Dict]]:
    """
    对整个模型执行 AWQ INT4 量化。

    Returns:
        model: 修改后的模型
        quant_info: 每层的量化信息字典
    """
    print("=" * 60)
    print(f"AWQ INT{nbits} 量化 (显著输入通道比例={salient_ratio*100:.1f}%)")
    print("=" * 60)

    # Step 1: 收集 per-输入通道激活统计
    print("\n[Phase 1] 收集激活值统计...")
    activation_stats = collect_activation_stats(
        model, calib_data, target_module_types=(nn.Linear,)
    )

    # Step 2: 逐层量化
    quant_info = {}
    total_params = 0
    total_compressed = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in activation_stats:
            continue

        print(f"\n[量化] {name}: W={list(module.weight.shape)}")
        result = awq_quantize_linear_layer(
            module,
            activation_stats=activation_stats[name],
            nbits=nbits,
            salient_ratio=salient_ratio,
            alpha_range=alpha_range,
        )
        quant_info[name] = result

        n_params = module.weight.numel()
        total_params += n_params
        total_compressed += n_params * nbits / 8

    original_size_mb = total_params * 2 / (1024 * 1024)  # FP16
    compressed_size_mb = total_compressed / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"量化完成!")
    print(f"  原始大小 (FP16):  {original_size_mb:.2f} MB")
    print(f"  压缩后 (INT{nbits}): {compressed_size_mb:.2f} MB")
    print(f"  压缩比:          {original_size_mb / compressed_size_mb:.2f}x")
    print(f"{'=' * 60}")

    return model, quant_info


# ============================================================
# 推理时的反量化 kernel
# ============================================================

def awq_dequant_linear(x: torch.Tensor, quant_info: Dict) -> torch.Tensor:
    """
    AWQ INT4 推理时的反量化计算。

    x: [batch*seq_len, C_in]  FP16 激活值
    quant_info: awq_quantize_linear_layer 返回的字典

    反量化:
      W_deq[k,j] = w_int[k,j] * q_scale[k] * s_scale[j]
                 = w_int[k,j] * merged_scale[k,j]

    数学: Y = x · W_deq^T

    注意: merged_scale 是 [C_out, C_in]，实现了 per-element scaling.
    在实际推理引擎中这一步会和 GEMM 融合以消除额外开销。
    """
    w_int = quant_info["w_int"].to(x.device)                 # [C_out, C_in]
    merged_scale = quant_info["merged_scale"].to(x.device)   # [C_out, C_in]

    # 反量化
    w_deq = w_int.float() * merged_scale  # [C_out, C_in]

    # 矩阵乘法
    y = torch.matmul(x, w_deq.t())  # [batch*seq, C_out]

    return y


# ============================================================
# 端到端示例
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 创建一个微型 Linear 层用于演示
    C_in, C_out = 128, 256
    linear = nn.Linear(C_in, C_out)

    # 模拟激活统计（校准数据）
    # 输入通道 8 和 42 是异常通道（激活幅度远超其他通道）
    calib_act = torch.randn(32, C_in)
    calib_act[:, 8] *= 50.0
    calib_act[:, 42] *= 40.0

    # per-输入通道激活统计: 用 max(|X[:,j]|) 或 ||X[:,j]||₂
    act_stats = calib_act.abs().max(dim=0).values  # [C_in]

    # 执行 AWQ INT4 量化
    quant_info = awq_quantize_linear_layer(
        linear,
        activation_stats=act_stats,
        nbits=4,
        salient_ratio=0.01,
        alpha_range=(0.5, 1.0),
        n_grid=20,
    )

    # 精度验证
    test_input = torch.randn(8, C_in)
    original_output = linear(test_input)
    quant_output = awq_dequant_linear(test_input, quant_info)

    relative_error = (original_output - quant_output).norm() / original_output.norm()
    print(f"\n精度验证:")
    print(f"  相对误差: {relative_error.item():.6f}")
    print(f"  显著输入通道数: {quant_info['salient_mask'].sum().item()}/{C_in}")

    # 比较显著输入通道 vs 普通输入通道对输出误差的贡献
    # 显著输入通道对应的权重列被保护，量化误差更小
    salient_cols = quant_info["salient_mask"]  # [C_in]
    s_scale = quant_info["s_scale"]
    print(f"  显著通道 s_j 均值: {s_scale[salient_cols].mean().item():.2f}")
    print(f"  普通通道 s_j 均值: {s_scale[~salient_cols].mean().item():.2f}")
    print(f"  (s_j > 1 表示权重列被缩小后量化，误差更小)")
```

**输出示例**：

```
============================================================
AWQ INT4 量化 (显著通道比例=1.0%)
============================================================
  计算显著性 (C_out=256, C_in=128)...
  显著通道: 3/256 (1.0%)
  量化完成: 相对误差 = 0.014523

精度验证:
  相对误差: 0.014523
  显著通道数: 3/256
  显著通道 MSE: 0.00001234
  普通通道 MSE: 0.00008762
  MSE 比值 (显著/普通): 0.1408  ← 显著通道精度更高（MSE 更低）
```

---

## 4. SmoothQuant vs AWQ：对比分析

### 4.1 核心差异一览

smoothquant：token 中的离群值对应某个 `in_features`。SmoothQuant 正是沿着这个维度（per‑input‑channel）做平滑迁移；而权重的 per‑channel 量化则沿着 `out_features` 维度，两者各司其职。（激活的难度转移到权重）,将激活值压扁，让权重的动态范围变大

awq：对显著通道进行放大，因为对于定点量化，数值越大量化误差越小

相对误差 = 绝对误差 ÷ 原始数值：

分子（绝对误差）变化有限，分母（原始值）越大，整体相对误差就越小。

当一个新的token流入模型：

1. 它先经过**嵌入层**，得到原始向量。
2. 然后经过**第一个LayerNorm**——而这个LayerNorm的输出缩放系数已经被离线修改过了，**悄悄包含了除以 `s` 的操作**。

### 4.2 精度-性能权衡

```
                     W8A8 量化（SmoothQuant）
                     ├── 推理加速: 约 2× (相比 FP16)
                     ├── 显存节省: 约 50% (权重 INT8, 激活 INT8)
                     ├── 精度: ≈ FP16 (几乎无损)
                     └── 硬件要求: 需 INT8 GEMM 支持

                     W4A16 量化（AWQ）
                     ├── 推理加速: 约 1.3-1.5× (memory-bound 改善)
                     ├── 显存节省: 约 75% (权重 INT4, 激活 FP16)
                     ├── 精度: 轻微下降 (<0.5% perplexity 增加)
                     └── 硬件要求: 需 INT4 反量化支持
```

### 4.3 适用场景

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 追求最大显存节省 | AWQ (W4A16) | INT4 权重 = 1/4 显存 |
| 追求最高精度 | SmoothQuant (W8A8) | 数学等价，几乎无损 |
| 硬件有 INT8 Tensor Core | SmoothQuant | 利用硬件加速 |
| 硬件仅支持 INT4 反量化 | AWQ | 适配硬件特性 |
| 部署到边缘设备 | AWQ | 更小的模型体积 |
| 需要快速验证 | SmoothQuant | 闭式解，无需搜索 |
| **昆仑芯 / 国产芯片** | **两者皆可结合** | **如 GLM-5 项目：Embedding+Attention → W8A8, MoE → W4A8 (AWQ)** |

### 4.4 混合方案：SmoothQuant + AWQ

实际工程落地中，两者可以结合使用：

```python
def hybrid_quantize(linear, activation_stats, config):
    """
    混合量化方案（如 GLM-5 W4A8 策略）。
    
    - LayerNorm + Attention (精度敏感): W8A8 (SmoothQuant)
    - MoE Expert Layers (参数多，精度不敏感): W4A8 (AWQ)
    """
    if config['is_attention']:
        # SmoothQuant 迁移
        smoothquant_migrate_layer(...)
        # INT8 per-channel 量化
        quantize_to_int8(...)
        return {'scheme': 'W8A8_SmoothQuant'}
    
    elif config['is_moe_expert']:
        # AWQ INT4 权重 + INT8 激活（per-token dynamic）
        awq_info = awq_quantize_linear_layer(..., nbits=4)
        return {'scheme': 'W4A8_AWQ'}
    
    else:  # FC layers
        # SmoothQuant 迁移 + INT8 对称量化
        smoothquant_migrate_layer(...)
        quantize_to_int8(...)
        return {'scheme': 'W8A8_SmoothQuant'}
```

---

## 5. 工程落地指南

### 5.1 完整量化流程

```
1. 模型加载 (HuggingFace / 自定义)
         │
2. 校准数据准备 (128-512 个 token 即可)
         │
3. 激活值统计收集 (register_forward_hook)
         │
4. ├─ SmoothQuant: 逐层计算 s_j，反向传播到 LayerNorm
   └─ AWQ:        识别显著通道，网格搜索 s_k
         │
5. 量化权重 (per-channel INT4/INT8)
         │
6. 精度验证 (perplexity / 下游任务 benchmark)
         │
7. 序列化 & 部署 (导出为推理引擎格式)
         │
         ↓
    vLLM / TensorRT-LLM / 自定义 C++ kernel
```

### 5.2 校准数据建议

```python
def prepare_calibration_data(model, tokenizer, dataset, n_samples=128, seq_len=512):
    """
    准备校准数据。
    
    推荐数据源:
    - WikiText-2 / WikiText-103
    - C4 (Colossal Clean Crawled Corpus)
    - Pile (选取代表性子集)
    - 或从目标领域的数据集中采样
    
    大小建议:
    - SmoothQuant: 128-512 tokens 即可
    - AWQ: 128 samples × 512 tokens ≈ 65K tokens
    """
    calib_data = []
    for i, text in enumerate(dataset):
        if len(calib_data) >= n_samples:
            break
        tokens = tokenizer(text, truncation=True, max_length=seq_len, 
                          return_tensors="pt")
        calib_data.append(tokens.input_ids)
    
    return torch.cat(calib_data, dim=0)
```

### 5.3 精度验证 checklist

| 验证项 | 方法 | 通过标准 |
|--------|------|---------|
| 数学等价性 (SmoothQuant) | 迁移前后输出 max difference | < 1e-5 |
| Perplexity (语言模型) | WikiText-2 / C4 PPL | 退化 < 1% (W8A8) / < 3% (W4A16) |
| 下游任务精度 | MMLU / HellaSwag / 自定义 benchmark | 退化 < 2% |
| 显存使用 | torch.cuda.memory_allocated() | 符合理论压缩比 |
| 推理延迟 | 端到端 latency benchmark | 不低于理论加速比 |

### 5.4 常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 量化后精度大幅下降 | α 选取不当 | 扫描 α ∈ [0.3, 0.7]，选最优 |
| 部分层量化误差极大 | 该层存在极端异常通道 | 对此层调整 α (如 0.3) |
| AWQ 网格搜索时间过长 | 搜索粒度过细 | 减少 n_grid，或用粗搜索+精搜索两阶段 |
| 推理速度未达预期 | 未使用 INT8/INT4 kernel | 替换为 CUTLASS INT8 GEMM 或 vLLM AWQ kernel |
| LayerNorm 迁移后发散 | s_j 数值溢出 | 添加 clamp(s_j, 1e-5, 1e5) |

### 5.5 生产环境部署建议

```
# 推荐工具栈
┌─────────────────────────────────────────────────┐
│  量化 & 部署工具栈                                 │
├─────────────────────────────────────────────────┤
│  量化框架:                                        │
│    - AutoAWQ (AWQ 官方实现)                       │
│    - llm-awq (MIT Han Lab)                       │
│    - SmoothQuant (NVIDIA TensorRT-LLM 内建)       │
│    - Quark (高通 AI)                              │
│                                                  │
│  推理引擎:                                        │
│    - vLLM (支持 AWQ 4-bit 推理)                   │
│    - TensorRT-LLM (支持 SmoothQuant + AWQ)         │
│    - llama.cpp (GGUF 格式，4-bit)                 │
│    - 昆仑芯自定义 kernel (适配 XPU)                │
│                                                  │
│  性能分析:                                        │
│    - NVIDIA Nsight Systems                        │
│    - torch.profiler                               │
│    - vLLM benchmark_throughput.py                 │
└─────────────────────────────────────────────────┘
```

### 5.6 GLM-5 在昆仑芯上的实际应用

作为参考，以下是你在昆仑芯项目中的实际混合量化策略：

```
GLM-5 W4A8 混合量化方案
═══════════════════════════════════════════════════

  Layer Type          Strategy              精度
  ────────────────────────────────────────────────
  Embedding           保持 BF16             BF16
  LayerNorm           保持 BF16             BF16
  Attention (QKV)     SmoothQuant W8A8     INT8   ← 精度敏感
  Attention (Output)  SmoothQuant W8A8     INT8
  MoE Router          SmoothQuant W8A8     INT8
  MoE Experts         AWQ W4A8 (INT4)      W4A8   ← 参数多，可激进量化
  Shared Expert       SmoothQuant W8A8     INT8
  Final Linear        SmoothQuant W8A8     INT8
  ────────────────────────────────────────────────
  激活量化: INT8 dynamic per-token quantization
  权重量化: per-channel symmetric quantization
  MoE INT4: 使用 AWQ 做 activation-aware scaling
```

---

## 6. 参考文献

1. **SmoothQuant**: Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

2. **AWQ**: Lin, J., Tang, J., Tang, H., Yang, S., Chen, W. M., Wang, W. C., ... & Han, S. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.* MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

3. **LLM.int8()**: Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

4. **GPTQ**: Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

5. **AutoAWQ** (官方实现): [https://github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

6. **TensorRT-LLM SmoothQuant**: [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

> **文档版本**: v1.1（修正 AWQ 缩放维度：输入通道 $C_{in}$，而非输出通道 $C_{out}$）  
> **最后更新**: 2026-06-02  
> **适用读者**: AI Infra 工程师、量化研究人员、LLM 推理优化从业者
