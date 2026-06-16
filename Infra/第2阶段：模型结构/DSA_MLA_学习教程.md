# DSA + MLA 深度技术教程

## ——以 DeepSeek V3.2 与 GLM-5 为对象的架构解析

---

## 目录

1. [引言：从标准注意力到稀疏注意力](#1-引言从标准注意力到稀疏注意力)
2. [注意力机制演进：MHA → GQA → MQA → MLA](#2-注意力机制演进mha--gqa--mqa--mla)
3. [MLA：多头潜在注意力](#3-mla多头潜在注意力)
   - 3.1 核心思想：低秩压缩
   - 3.2 Query 投影
   - 3.3 KV 投影与缓存压缩
   - 3.4 混合位置编码：NOPE + ROPE
   - 3.5 Naive 模式与 Absorb 模式
   - 3.6 完整前向传播流程
4. [DSA：DeepSeek 稀疏注意力](#4-dsadeepseek-稀疏注意力)
   - 4.1 设计动机
   - 4.2 闪电索引器（Lightning Indexer）
   - 4.3 细粒度 Token 选择
   - 4.4 两阶段继续预训练
   - 4.5 推理加速分析
5. [DeepSeek V3.2 架构全解析](#5-deepseek-v32-架构全解析)
6. [GLM-5 架构全解析](#6-glm-5-架构全解析)
7. [两大架构对比分析](#7-两大架构对比分析)
8. [总结与展望](#8-总结与展望)

---

## 1. 引言：从标准注意力到稀疏注意力

### 1.1 注意力机制的根本瓶颈

Transformer 模型的核心是自注意力机制。对于一个长度为 $L$ 的序列，标准缩放点积注意力（Scaled Dot-Product Attention）的复杂度为 $\mathcal{O}(L^2)$：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$，注意力分数矩阵 $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{L \times L}$。

当 $L$ 增长到 128K 甚至 200K 时，平方级的计算量成为不可逾越的瓶颈。这一问题的两个子方向分别是：

| 问题 | 描述 | 解决方向 |
|------|------|----------|
| **推理时延** | 每生成一个 token，需与全部历史 KV 计算注意力 | DSA（稀疏注意力） |
| **显存占用** | 需缓存全部历史 KV 对 | MLA（低秩压缩） |

本文所讨论的 **MLA（Multi-head Latent Attention）** 和 **DSA（DeepSeek Sparse Attention）** 正是分别从缓存压缩与计算稀疏化两个维度攻克上述问题，并在 **DeepSeek V3.2** 和 **GLM-5** 两大先进模型中得到了深度融合。

---

## 2. 注意力机制演进：MHA → GQA → MQA → MLA

在深入 MLA 之前，有必要回顾注意力机制的演进脉络。

### 2.1 MHA（Multi-Head Attention）

标准多头注意力将 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 分别投影到 $h$ 个头：

$$
\mathbf{Q}_i = \mathbf{X}\mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{X}\mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{X}\mathbf{W}_i^V
$$

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i)
$$

$$
\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O
$$

每个头有独立的 KV 投影，推理时需缓存 $2 \times h \times d_h$ 维 KV。

### 2.2 MQA（Multi-Query Attention）

所有头共享同一组 K 和 V：

每个查询头共享 $\mathbf{K}, \mathbf{V}$，仅 $\mathbf{Q}$ 有多头。KV 缓存缩减为 $h$ 分之一，但模型容量受损。

### 2.3 GQA（Grouped-Query Attention）

将 $h$ 个头分为 $g$ 组，组内共享 KV：

KV 头数从 $h$ 降为 $g$，通常取 $g=8$（即 GQA-8），在效率和容量之间取得平衡。

**GLM-5 技术报告中的关键发现**：在 576 维的潜在 KV 缓存下，MLA 无法匹配 GQA-8（2048 维 KV 缓存）的性能，这驱动了后续 MLA-256 变体与 Muon Split 的提出。

### 2.4 演进对比

| 机制 | KV 缓存维度（每层每 token） | 模型容量 | 推理效率 |
|------|---------------------------|----------|----------|
| MHA | $2 \times h \times d_h$ | ★★★★★ | ★☆☆☆☆ |
| GQA-8 | $2 \times g \times d_h, g=8$ | ★★★★☆ | ★★★☆☆ |
| MQA | $2 \times d_h$ | ★★★☆☆ | ★★★★★ |
| **MLA** | $d_c + d_R$（压缩后 $< 600$） | ★★★★★ | ★★★★★ |

---

## 3. MLA：多头潜在注意力

MLA（Multi-head Latent Attention）由 DeepSeek-V2 首次提出并沿用于 DeepSeek-V3 系列。其核心思想是：通过 **低秩分解** 将 KV 投影压缩到一个远小于原始空间的潜在空间，大幅减少推理时的 KV 缓存。

### 3.1 核心思想：低秩压缩

MLA 的数学本质是：注意力机制中的键（Key）和值（Value）矩阵存在显著的 **低秩结构**，可以通过矩阵分解将高维的 KV 投影压缩到低维潜在空间，在推理时再从这一低维表示恢复。

下投影压缩，上投影还原，最终x输出投影

设输入隐藏状态为 $\mathbf{h}_t \in \mathbb{R}^{d}$，其中 $d$ 为模型维度。

**标准 MHA 的 KV 投影**：

$$
\mathbf{k}_t^C = \mathbf{W}^{UK}\mathbf{h}_t \in \mathbb{R}^{h \times d_h}, \quad \mathbf{v}_t^C = \mathbf{W}^{UV}\mathbf{h}_t \in \mathbb{R}^{h \times d_h}
$$

其中 $\mathbf{W}^{UK} \in \mathbb{R}^{(h \cdot d_h) \times d}$，$\mathbf{W}^{UV} \in \mathbb{R}^{(h \cdot d_h) \times d}$。参数量为 $2 \times d \times h \times d_h$，推理时每条缓存占 $2 \times h \times d_h$ 个元素。

**MLA 的低秩分解**：
$$
\mathbf{c}_t^{KV} = \mathbf{W}^{DKV}\mathbf{h}_t \in \mathbb{R}^{d_c}
$$

其中 $d_c \ll h \times d_h$（在 DeepSeek V3 中 $d_c = 512$，而 $h \times d_h = 128 \times 128 = 16384$）。

然后从压缩向量 $\mathbf{c}_t^{KV}$ 恢复出完整的 K 和 V：

$$
\mathbf{k}_t^C = \mathbf{W}^{UK}\mathbf{c}_t^{KV}, \quad \mathbf{v}_t^C = \mathbf{W}^{UV}\mathbf{c}_t^{KV}
$$

其中 $\mathbf{W}^{UK} \in \mathbb{R}^{(h \cdot d_h) \times d_c}$，$\mathbf{W}^{UV} \in \mathbb{R}^{(h \cdot d_h) \times d_c}$。

### 3.2 Query 投影

MLA 对 Query 也采用了类似的低秩压缩，以进一步减少训练参数量。

**Query 的两阶段投影**：

**第一阶段（下投影）**：

$$
\mathbf{c}_t^Q = \mathbf{W}^{DQ}\mathbf{h}_t \in \mathbb{R}^{d_c'}
$$

其中 $\mathbf{W}^{DQ} \in \mathbb{R}^{d_c' \times d}$，在 DeepSeek V3 中 $d_c' = 1536$。

**中间归一化**：

$$
\tilde{\mathbf{c}}_t^Q = \text{RMSNorm}(\mathbf{c}_t^Q)
$$

**第二阶段（上投影）**：

$$
\mathbf{q}_t^C = \mathbf{W}^{UQ}\tilde{\mathbf{c}}_t^Q \in \mathbb{R}^{h \times d_h}
$$

其中 $\mathbf{W}^{UQ} \in \mathbb{R}^{(h \cdot d_h) \times d_c'}$。

**参数量对比**（以 DeepSeek V3 为例）：

$$
\begin{aligned}
\text{MHA 直接投影} &: d \times h \times d_h = 7168 \times 128 \times 192 = 176\text{M} \\
\text{MLA LoRA 投影} &: d \times d_c' + d_c' \times h \times d_h = 7168 \times 1536 + 1536 \times 128 \times 192 = 48.5\text{M} \\
\text{参数缩减} &: \mathbf{72\%}
\end{aligned}
$$

### 3.3 KV 投影与缓存压缩

KV 投影是 MLA 节省推理显存的核心所在。

**压缩投影**（编码器）：

$$
\mathbf{c}_t^{KV} = \mathbf{W}^{DKV}\mathbf{h}_t \in \mathbb{R}^{d_c + d_R}
$$

其中 $d_c$ 为非位置分量维度（在 DeepSeek V3 中 $d_c = 512$），$d_R$ 为位置编码分量维度（$d_R = 64$）。

将压缩向量拆分为内容部分和位置部分：

$$
\mathbf{c}_t^{KV} = [\mathbf{c}_t^{KV, \text{nope}} \;\|\; \mathbf{k}_t^R]
$$

其中 $\mathbf{c}_t^{KV, \text{nope}} \in \mathbb{R}^{d_c}$，$\mathbf{k}_t^R \in \mathbb{R}^{d_R}$。

**展开投影**（解码器）：

$$
[\mathbf{k}_{t,1}^C, \ldots, \mathbf{k}_{t,h}^C, \mathbf{v}_{t,1}^C, \ldots, \mathbf{v}_{t,h}^C] = \mathbf{W}^{UV}\mathbf{c}_t^{KV, \text{nope}}
$$

其中 $\mathbf{W}^{UV} \in \mathbb{R}^{h \times (d_h^{\text{nope}} + d_h^v) \times d_c}$。

**KV 缓存压缩比**（DeepSeek V3 Absorb 模式）：

$$
\begin{aligned}
\text{MHA 缓存} &: 2 \times h \times d_h = 2 \times 128 \times 192 = 49,152 \text{ dims/token} \\
\text{MLA 缓存} &: d_c + d_R = 512 + 64 = 576 \text{ dims/token} \\
\text{压缩比} &: \mathbf{85.3\times} \;\;(\mathbf{98.8\%} \text{ 缩减})
\end{aligned}
$$

### 3.4 混合位置编码：NOPE + ROPE

MLA 采用混合位置编码策略，将 Query 和 Key 拆分为内容感知部分（NOPE）和位置感知部分（ROPE）。

**Query 拆分**：

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^{\text{nope}} \;\|\; \mathbf{q}_{t,i}^{\text{rope}}]
$$

其中 $\mathbf{q}_{t,i}^{\text{nope}} \in \mathbb{R}^{d_h^{\text{nope}}}$（内容部分），$\mathbf{q}_{t,i}^{\text{rope}} \in \mathbb{R}^{d_h^{\text{rope}}}$（位置部分）。

对位置部分应用旋转位置编码（RoPE）：

$$
\mathbf{q}_{t,i}^{\text{rope}} \leftarrow \text{RoPE}(\mathbf{q}_{t,i}^{\text{rope}}, t)
$$

**Key 的对应处理**：

$$
\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^{\text{nope}} \;\|\; \mathbf{k}_{t,i}^{\text{rope}}]
$$

其中 $\mathbf{k}_{t,i}^{\text{nope}} \in \mathbb{R}^{d_h^{\text{nope}}}$ 来源于 $\mathbf{W}^{UV}$ 的展开，$\mathbf{k}_{t,i}^{\text{rope}} \in \mathbb{R}^{d_h^{\text{rope}}}$ 直接来自编码器的 $\mathbf{k}_t^R$ 经 RoPE 编码：

$$
\mathbf{k}_{t}^{\text{rope}} \leftarrow \text{RoPE}(\mathbf{k}_t^R, t)
$$

这里有一个重要的实现细节：**$\mathbf{k}^{\text{rope}}$ 在所有注意力头之间共享**，即 MQA（Multi-Query Attention）模式用于位置编码部分。

**RoPE 的数学形式**：

RoPE 通过旋转矩阵对向量进行相位编码。对于维度 $d$，构造分块对角旋转矩阵：

$$
\mathbf{R}_{\Theta, t} = \begin{bmatrix}
\cos t\theta_1 & -\sin t\theta_1 & 0 & 0 & \cdots \\
\sin t\theta_1 & \cos t\theta_1 & 0 & 0 & \cdots \\
0 & 0 & \cos t\theta_2 & -\sin t\theta_2 & \cdots \\
0 & 0 & \sin t\theta_2 & \cos t\theta_2 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix} \in \mathbb{R}^{d \times d}
$$

其中 $\theta_j = 10000^{-2j/d}$。则：

$$
\text{RoPE}(\mathbf{x}, t) = \mathbf{R}_{\Theta, t}\mathbf{x}
$$

<img src="assets/20260104003909889.png" alt="MLA理解" style="zoom:50%;" />

### 3.5 Naive 模式与 Absorb 模式

MLA 有两种推理实现模式，在效率与显存之间提供灵活的权衡。

#### Naive 模式

Naive 模式在推理时**预计算并缓存完整的 K 和 V**，与标准注意力类似：

**缓存结构**：

```
K_cache: (batch, seq_len, n_local_heads, qk_head_dim)
V_cache: (batch, seq_len, n_local_heads, v_head_dim)
```

**注意力计算**（一步完成）：

$$
\text{score}_{t} = \frac{\mathbf{q}_t \cdot \mathbf{K}_{\text{cache}}^\top}{\sqrt{d_h}}
$$

$$
\mathbf{u}_t = \text{Softmax}(\text{score}_t) \cdot \mathbf{V}_{\text{cache}}
$$

**适用场景**：显存充足、追求最大吞吐量。

#### Absorb 模式（默认）

Absorb 模式将 $\mathbf{W}^{UV}$ 的展开操作 **延迟到注意力计算时进行**，仅缓存压缩后的潜在表示：

**缓存结构**：

```
kv_cache: (batch, seq_len, kv_lora_rank)       ← 仅 512 维
pe_cache: (batch, seq_len, qk_rope_head_dim)   ← 仅 64 维
```

**注意力计算**（两阶段）：

**阶段一：内容注意力**

将 $\mathbf{W}^{UV}$ 的 Key 部分（记为 $\mathbf{W}^{UK}$）吸收到注意力计算中：

$$
\text{score}_{t}^{\text{content}} = \mathbf{q}_t^{\text{nope}} \cdot \left(\mathbf{W}^{UK} \cdot \mathbf{KV}_{\text{cache}}\right)^\top
$$

其中 $\mathbf{q}_t^{\text{nope}} \in \mathbb{R}^{h \times d_h^{\text{nope}}}$，$\mathbf{W}^{UK} \in \mathbb{R}^{(h \times d_h^{\text{nope}}) \times d_c}$，$\mathbf{KV}_{\text{cache}} \in \mathbb{R}^{t \times d_c}$。

**阶段二：位置注意力**

位置部分的 Key 在所有头之间共享（MQA 模式）：

$$
\text{score}_{t}^{\text{pos}} = \mathbf{q}_t^{\text{rope}} \cdot \left(\mathbf{K}_{\text{pe\_cache}}\right)^\top
$$

其中 $\mathbf{q}_t^{\text{rope}} \in \mathbb{R}^{h \times d_h^{\text{rope}}}$，$\mathbf{K}_{\text{pe\_cache}} \in \mathbb{R}^{t \times d_h^{\text{rope}}}$。

**合并注意力分数**：

$$
\text{score}_{t} = \frac{\text{score}_{t}^{\text{content}} + \text{score}_{t}^{\text{pos}}}{\sqrt{d_h^{\text{nope}} + d_h^{\text{rope}}}}
$$

**最终输出**：

$$
\mathbf{u}_t = \text{Softmax}(\text{score}_t) \cdot \left(\mathbf{W}^{UV} \cdot \mathbf{KV}_{\text{cache}}\right)
$$

其中 $\mathbf{W}^{UV} \in \mathbb{R}^{(h \times d_h^v) \times d_c}$ 提取 V 部分。

**单层 KV 缓存大小对比**（DeepSeek V3，batch=8, seq_len=16384）：

| 缓存类型 | Naive 模式 | Absorb 模式 | 缩减比 |
|----------|-----------|-------------|--------|
| K | 6.44 GB | — | — |
| V | 4.29 GB | — | — |
| KV（压缩） | — | 0.134 GB | — |
| PE | — | 0.017 GB | — |
| **总计** | **10.73 GB** | **0.151 GB** | **71×** |

### 3.6 完整前向传播流程

MLA 层的完整前向传播可以用如下计算图表示：

```
输入: h_t ∈ R^d
│
├─ Query 路径:
│   h_t → W^DQ → c_t^Q ∈ R^{d_c'} → RMSNorm → W^UQ → q_t ∈ R^{h × d_h}
│   q_t → [q_t^nope ∥ q_t^rope]
│   q_t^rope → RoPE(q_t^rope, t)
│
├─ KV 路径:
│   h_t → W^DKV → c_t^KV ∈ R^{d_c + d_R} → [c_t^{KV,nope} ∥ k_t^R]
│   k_t^R → RoPE(k_t^R, t)
│   c_t^{KV,nope} → 存入 KV_cache
│   k_t^R → 存入 PE_cache
│
├─ 注意力计算 (Absorb 模式):
│   score_content = q^nope · (W^UK · KV_cache)^⊤
│   score_pos     = q^rope · PE_cache^⊤
│   score         = (score_content + score_pos) / √d_h
│   attn_weights  = Softmax(score)
│   u_t           = attn_weights · (W^UV · KV_cache)
│
└─ 输出投影:
    o_t = u_t · W^O
    输出: h_t' = h_t + o_t (残差连接)
```

```
#MLA的TP方式
# 切 W_DKV
c_kv1 = X @ W_DKV1        # 卡1 [B,S,128]
c_kv2 = X @ W_DKV2        # 卡2 [B,S,128]

# 切 Q
Q1, Q2 = 切分Q

# 只和自己分片算
attn1 = SDPA(Q1, c_kv1)   # 卡1
attn2 = SDPA(Q2, c_kv2)   # 卡2

# 合并
output = attn1 + attn2
```

## 4. DSA：DeepSeek 稀疏注意力

DSA（DeepSeek Sparse Attention）由 DeepSeek-V3.2 首次引入，是 DeepSeek-V3.1-Terminus 到 V3.2 的**唯一架构修改**。其核心思想是将注意力计算从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(Lk)$（其中 $k \ll L$），通过一个轻量级的 **闪电索引器** 动态选择重要的历史 token。

### 4.1 设计动机：稀疏 ≠ 固定模式

传统的稀疏注意力（如滑动窗口、跨步注意力）使用**固定的稀疏模式**，与内容无关。这种方案的致命缺陷是：对于需要跨越窗口的长程依赖（如代码中跨文件的函数引用、长文档中跨越数万 token 的前后呼应），固定模式必然丢失关键信息。

DSA 的核心创新在于：**稀疏模式由内容动态决定**。

### 4.2 闪电索引器（Lightning Indexer）

闪电索引器是一个小型辅助网络，负责评估历史 token 对当前查询 token 的重要性。

**索引分数计算**：

对于查询 token $\mathbf{h}_t \in \mathbb{R}^d$ 和历史 token $\mathbf{h}_s \in \mathbb{R}^d$（$s < t$），索引分数 $I_{t,s}$ 定义为：

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)
$$

其中：

| 符号 | 含义 | 形状/维度 |
|------|------|-----------|
| $H^I$ | 索引器头数 | DeepSeek V3.2: 32, GLM-5: 32 |
| $\mathbf{q}_{t,j}^I$ | 第 $j$ 个索引头的查询向量 | $\mathbb{R}^{d^I}$ ($d^I=128$) |
| $\mathbf{k}_s^I$ | 键向量（跨头共享） | $\mathbb{R}^{d^I}$ |
| $w_{t,j}^I$ | 第 $j$ 个索引头的权重 | $\mathbb{R}$ |

$\mathbf{q}_{t,j}^I$、$\mathbf{k}_s^I$ 和 $w_{t,j}^I$ 均由 $\mathbf{h}_t$ 和 $\mathbf{h}_s$ 通过小型线性投影得到：

$$
\mathbf{q}_{t,j}^I = \mathbf{W}_j^{QI}\mathbf{h}_t, \quad \mathbf{k}_s^I = \mathbf{W}^{KI}\mathbf{h}_s, \quad w_{t,j}^I = \text{Sigmoid}(\mathbf{w}_j^{I} \cdot \mathbf{h}_t)
$$

**选择 ReLU 而非 Softmax 的原因**：ReLU 可直接在 FP8 精度下高效实现，减少索引器本身的吞吐量开销。

**索引器的计算复杂度**：虽然索引器也是 $\mathcal{O}(L^2)$，但由于头数少、精度低，其实际计算量远小于完整的 MLA 注意力。对于长上下文场景，索引器的额外开销被主注意力 $\mathcal{O}(Lk)$ 带来的节省完全覆盖。

### 4.3 细粒度 Token 选择

基于索引器输出的分数 $\{I_{t,s}\}_{s=1}^{t}$，DSA 选择 top-$k$ 个历史 token 参与注意力计算：

$$
\mathcal{S}_t = \left\{s \;\middle|\; I_{t,s} \in \text{Top-}k(\{I_{t,s'}\}_{s'=1}^{t})\right\}
$$

然后只在这些选中的 token 上计算主注意力：

$$
\mathbf{u}_t = \text{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \;\middle|\; s \in \mathcal{S}_t\right\}\right)
$$

其中 $\mathbf{c}_s$ 是 MLA 中 token $s$ 的 key-value 条目。

**关键设计细节**：

1. **Top-$k$ 选择在对数空间进行**，而非在 softmax 之后的概率空间 —— 保证选择的是信息量最大的 token 而非最"确定"的 token。

2. **DSA 基于 MLA 的 MQA 模式**：在解码阶段，MLA 在 MQA 模式下运行（即 key-value 在所有查询头之间共享），DSA 在此基础上应用稀疏选择。

3. **$k$ 的选择**：在 DeepSeek V3.2 和 GLM-5 中，$k = 2048$。对于 128K 的上下文，这意味着仅需计算 1.6% 的 token 对。

### 4.4 两阶段继续预训练

DSA 通过**从稠密模型继续预训练**的方式引入，分为两个阶段。

#### 第一阶段：密集热身（Dense Warm-up）

| 属性 | 配置 |
|------|------|
| 注意力模式 | 保持密集注意力（全量 $\mathcal{O}(L^2)$） |
| 参数冻结 | 冻结除闪电索引器外的所有参数 |
| 学习率 | $10^{-3}$ |
| 训练步数 | 1000 步 |
| 总训练量 | ~21 亿 tokens |

**训练目标**：使索引器输出与主注意力分布对齐。

定义目标分布 $p_{t,:}$ 为所有注意力头聚合、沿序列维度 $\ell_1$ 归一化后的主注意力分数：

$$
p_{t,s} = \frac{\sum_{j=1}^{H} A_{t,s}^{(j)}}{\sum_{s'=1}^{t} \sum_{j=1}^{H} A_{t,s'}^{(j)}}
$$

其中 $A_{t,s}^{(j)}$ 为第 $j$ 个注意力头在查询位置 $t$ 对键位置 $s$ 的注意力分数。

索引器的训练损失为 KL 散度：

$$
\mathcal{L}^I = \sum_{t} D_{\mathrm{KL}}\left(p_{t,:} \;\middle\|\; \text{Softmax}(I_{t,:})\right)
$$

#### 第二阶段：稀疏训练（Sparse Training）

| 属性 | 配置 (DeepSeek V3.2) | 配置 (GLM-5) |
|------|----------------------|-------------|
| 注意力模式 | DSA 稀疏（top-$k$） | DSA 稀疏（top-$k$） |
| 参数优化 | 优化全部模型参数 | 优化全部模型参数 |
| 选中 token 数 $k$ | 2048 | 2048 |
| 学习率 | $7.3 \times 10^{-6}$ | 沿用 mid-training 学习率 |
| 训练步数 | 15000 步 | — |
| 总训练量 | ~9437 亿 tokens | ~200 亿 tokens |

第二阶段仅考虑被选中的 token 集合 $\mathcal{S}_t$：

$$
\mathcal{L}^I = \sum_{t} D_{\mathrm{KL}}\left(p_{t,\mathcal{S}_t} \;\middle\|\; \text{Softmax}(I_{t,\mathcal{S}_t})\right)
$$

**关键的训练分离策略**：
- 索引器输入从计算图中分离（detach），避免梯度污染主模型
- 索引器仅由 $\mathcal{L}^I$ 训练
- 主模型仅由语言建模损失训练
- 这保证了索引器的选择策略不被语言建模的梯度"腐蚀"

### 4.5 推理加速分析

**复杂度对比**：

| 组件 | 稠密 MLA (V3.1) | DSA + MLA (V3.2) |
|------|----------------|------------------|
| 主模型注意力 | $\mathcal{O}(L^2 \cdot h \cdot d_h)$ | $\mathcal{O}(Lk \cdot h \cdot d_h)$ |
| 索引器 | 不存在 | $\mathcal{O}(L^2 \cdot H^I \cdot d^I)$（轻量） |

当 $L = 128\text{K}$、$k = 2048$、$h = 128$、$d_h = 192$、$H^I = 32$、$d^I = 128$ 时：

$$
\begin{aligned}
\text{稠密主注意力 FLOPs} &\propto L^2 \cdot h \cdot d_h = (131072)^2 \times 128 \times 192 \approx 4.2 \times 10^{14} \\
\text{DSA 主注意力 FLOPs} &\propto L \cdot k \cdot h \cdot d_h = 131072 \times 2048 \times 128 \times 192 \approx 6.6 \times 10^{12} \\
\text{DSA 索引器 FLOPs} &\propto L^2 \cdot H^I \cdot d^I = (131072)^2 \times 32 \times 128 \approx 7.0 \times 10^{13}
\end{aligned}
$$

**总加速比**：

$$
\frac{4.2 \times 10^{14}}{6.6 \times 10^{12} + 7.0 \times 10^{13}} \approx 5.5\times
$$

实际推理中，由于索引器的 FP8 实现和 Kernel 融合，DSA 在长上下文解码阶段可实现 $\mathbf{1.5\times \sim 2\times}$ 的端到端延迟降低。

**短序列优化**：对于短序列预填充，DeepSeek V3.2 实现了**掩码 MHA 模式**来模拟 DSA 的选择行为，在短上下文条件下实现更高效率。

---

## 5. DeepSeek V3.2 架构全解析

### 5.1 整体架构概览

DeepSeek V3.2 基于 V3.1-Terminus 继续训练而来，V3.2 相对于 V3.1 的**唯一架构修改**是引入了 DSA。其余架构参数与 V3 基本一致。

```
DeepSeek V3.2 架构
├── Embedding (129,280 × 7168)
├── Layer 0-2: Dense Transformer（前 3 层无 MoE）
│   └── MLA (DSA) + RMSNorm + FFN
├── Layer 3-60: MoE Transformer（58 层 MoE）
│   └── MLA (DSA) + RMSNorm + MoE FFN
├── RMSNorm
└── LM Head (7168 → 129,280)
    └── MTP Module (1 层，14B 参数)
```

### 5.2 详细参数配置

#### 基础参数

| 参数 | 值 |
|------|-----|
| 总参数量 | 671B（含 MTP: 685B） |
| 每 Token 激活参数 | 37B |
| 模型维度 $d$ | 7168 |
| 总层数 | 61（3 Dense + 58 MoE） |
| 词汇量 | 129,280 |
| 上下文长度 | 128K tokens |

#### MLA 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_heads` | 128 | 注意力头总数 |
| `q_lora_rank` ($d_c'$) | 1536 | Query 压缩秩 |
| `kv_lora_rank` ($d_c$) | 512 | KV 压缩秩 |
| `qk_nope_head_dim` ($d_h^{\text{nope}}$) | 128 | 每头非位置 Q/K 维度 |
| `qk_rope_head_dim` ($d_h^{\text{rope}}$) | 64 | 每头旋转位置维度 |
| `qk_head_dim` ($d_h$) | 192 (=128+64) | 每头总 Q/K 维度 |
| `v_head_dim` ($d_h^v$) | 128 | 每头 Value 维度 |

#### MoE 参数

| 参数 | 值 |
|------|-----|
| 路由专家数 `n_routed_experts` | 256 |
| 每 Token 激活专家数 | 8 |
| 共享专家数 | 2 |
| 专家分组数 | 8 |
| 专家隐藏维度 `moe_inter_dim` | 1408 |
| 路由评分函数 | Sigmoid |
| 路由缩放因子 | 2.5 |

#### DSA 参数

| 参数 | 值 |
|------|-----|
| 索引器头数 $H^I$ | 32 |
| 索引器头维度 $d^I$ | 128 |
| Top-$k$ | 2048 |
| 索引器激活函数 | ReLU |
| 索引器精度 | FP8 |

#### MTP（Multi-Token Prediction）

| 属性 | 配置 |
|------|------|
| MTP 层数 | 1 层 |
| MTP 参数量 | 14B |
| 预测 token 数 | 1（推测时 2） |

### 5.3 MLA 在 DeepSeek V3.2 中与 DSA 的融合

每个 Transformer 层的计算流程：

```
输入: h ∈ R^(bsz × seqlen × 7168)
│
1. Pre-Norm:  h_norm = RMSNorm(h)
│
2. MLA + DSA:
│   ├─ Query:  c_q = W^DQ · h_norm → RMSNorm → W^UQ → q ∈ R^(128 × 192)
│   ├─ KV:     c_kv = W^DKV · h_norm → [c_kv^nope ∥ k_pe]
│   ├─ 缓存:   如果 Absorb 模式: 存入 c_kv^nope (512 维) 和 k_pe (64 维)
│   ├─ DSA:    I = LightningIndexer(h_norm) → Top-2048 选择
│   └─ Attn:   仅对 Top-2048 token 计算注意力 → u ∈ R^(128 × 128) → concat → W^O
│
3. Residual:  h = h + attn_out
│
4. Post-Norm: h_norm = RMSNorm(h)
│
5. MoE FFN (Layer 3-60) 或 Dense FFN (Layer 0-2):
│   ├─ Router: g_i = Sigmoid(W_r · h_norm) → Top-8 专家选择
│   ├─ 专家计算: expert_i = W_2 · SiLU(W_1 · h_norm)
│   ├─ 共享专家: shared = W_2^s · SiLU(W_1^s · h_norm)
│   └─ 输出: ffn_out = Σ g_i · expert_i + shared
│
6. Residual:  h = h + ffn_out → 输出至下一层
```

### 5.4 DSA 在 DeepSeek V3.2 中的训练配置

| 阶段 | 配置 | 训练量 |
|------|------|--------|
| Dense Warm-up | 冻结主模型，仅训练索引器，LR=10⁻³ | 1000 步 × 16 seq × 128K = ~21 亿 tokens |
| Sparse Training | 训练全参数，LR=7.3×10⁻⁶ | 15000 步 × 480 seq × 128K = ~9437 亿 tokens |

### 5.5 性能验证

DSA 在 DeepSeek V3.2 中实现了**无损压缩**：

- 标准基准测试：V3.2-Exp 与 V3.1-Terminus 在短期和长上下文任务上表现相似，**无实质性退化**
- 长上下文评估（AA-LCR）：V3.2-Exp 比 V3.1 高 **4 分**
- Fiction.liveBench：V3.2-Exp 在多个指标上 **一致优于** V3.1
- ChatbotArena Elo：V3.2 与 V3.1 极为接近

---

## 6. GLM-5 架构全解析

### 6.1 整体架构概览

GLM-5 是智谱 AI 于 2025 年底发布的新一代基础模型，同样采用 MoE + MLA + DSA 架构，但在多个维度上有独特的创新。

```
GLM-5 架构
├── Embedding (154880 × 6144)
├── Layer 0-2: Dense Transformer（前 3 层无 MoE）
│   └── MLA (DSA) + RMSNorm + FFN
├── Layer 3-77: MoE Transformer（75 层 MoE）
│   └── MLA (DSA) + RMSNorm + MoE FFN
├── RMSNorm
└── LM Head (6144 → 154880)
    └── MTP Module (1 层共享参数，3 个预测深度)
```

### 6.2 详细参数配置

#### 基础参数

| 参数 | 值 |
|------|-----|
| 总参数量 | 744B |
| 每 Token 激活参数 | 40B |
| 模型维度 $d$ | 6144 |
| 总层数 | 78（3 Dense + 75 MoE） |
| 词汇量 | 154,880 |
| 上下文长度 | 202,752 tokens |
| 架构标识 | `GlmMoeDsaForCausalLM` |

#### MLA-256 参数

GLM-5 对 MLA 进行了关键改进，提出了 **MLA-256** 变体：

| 参数 | GLM-5 | DeepSeek V3 | 说明 |
|------|-------|-------------|------|
| `n_heads` | 64 | 128 | GLM-5 头数减半 |
| `q_lora_rank` | 2048 | 1536 | GLM-5 使用更大的 Q 压缩秩 |
| `kv_lora_rank` | 512 | 512 | 相同 |
| `qk_nope_head_dim` | **192** | 128 | GLM-5 扩大 NOPE 维度 |
| `qk_rope_head_dim` | 64 | 64 | 相同 |
| `qk_head_dim` | **256** (=192+64) | 192 (=128+64) | GLM-5 头维度增加 33% |
| `v_head_dim` | **256** | 128 | GLM-5 V 维度翻倍 |
| `head_dim` | 64 | — | 基础头维度（用于某些计算） |

**MLA-256 的动机**：

在解码阶段，MLA 需要执行 $d_h$ 维度的点积运算，而 GQA 只需 $d_h^{\text{GQA}}$ 维。如果 $d_h^{\text{MLA}}$ 过大，解码阶段的点积计算量会超过 GQA。

GLM-5 的策略：
1. 将头维度从 192 扩大到 256
2. 同时将注意力头数从 128 减少到 64（减少 1/3）
3. 净效果：总计算量（$n_{\text{heads}} \times d_h$）保持不变（$128 \times 192 = 64 \times 256 = 24576$），但解码阶段的单次点积更高效

#### Muon Split 优化

GLM-5 技术报告中还提出了 **Muon Split**，这是对 Muon 优化器在 MLA 上的适配改进。

**问题**：原始 Muon 优化器对整个上投影矩阵 $\mathbf{W}^{UQ}$、$\mathbf{W}^{UK}$、$\mathbf{W}^{UV}$ 整体应用矩阵正交化。但由于 MLA 的上投影矩阵由多个注意力头共享，不同头可能需要不同的更新尺度。

**Muon Split 方案**：将大矩阵按注意力头拆分为多个独立的小矩阵，分别进行正交化：

$$
\mathbf{W}^{UQ} \rightarrow [\mathbf{W}_1^{UQ}, \mathbf{W}_2^{UQ}, \ldots, \mathbf{W}_h^{UQ}], \quad \mathbf{W}_i^{UQ} \in \mathbb{R}^{d_h \times d_c'}
$$

每个子矩阵独立进行 Muon 正交化。这使得不同注意力头的投影权重可以以不同尺度更新，性能匹配 GQA-8。

**性能对比**（摘自 GLM-5 技术报告 Table 1）：

| 数据集 | GQA-8 | MLA | MLA + Muon Split | MLA-256 + Muon Split |
|--------|-------|-----|-------------------|-----------------------|
| Hellaswag | 77.3 | 77.3 | 77.8 | 77.4 |
| MMLU | 61.2 | 61.5 | 62.5 | 62.0 |
| C-Eval | 60.0 | 59.7 | 62.1 | 59.9 |
| GSM8K | 47.6 | 46.2 | 45.0 | 47.5 |
| HumanEval | 38.5 | 33.5 | 36.7 | 36.6 |

#### MoE 参数

| 参数 | 值 | 对比 DS V3 |
|------|-----|-----------|
| 路由专家数 | 256 | 256（相同） |
| 每 Token 激活专家数 | 8 | 8（相同） |
| 共享专家数 | 1 | 2 |
| 专家隐藏维度 | 2048 | 1408 |
| 路由评分函数 | Sigmoid | Sigmoid |
| 路由缩放因子 | 2.5 | 2.5 |
| Top-$k$ 方法 | `noaux_tc` | — |

#### DSA 参数

| 参数 | GLM-5 | DeepSeek V3.2 |
|------|-------|---------------|
| `index_n_heads` | 32 | 32 |
| `index_head_dim` | 128 | 128 |
| `index_topk` | 2048 | 2048 |
| 索引器输出激活 | ReLU | ReLU |
| 索引器精度 | FP8 | FP8 |

#### MTP 参数

| 属性 | GLM-5 | DeepSeek V3.2 |
|------|-------|---------------|
| MTP 层数 | 3（参数共享） | 1 |
| 推测步数 | 4 | 2 |
| 接受长度 | **2.76** | 2.55 |

### 6.3 GLM-5 中 MLA 与 DSA 的独特融合

GLM-5 在 DSA 的 RL 训练中有着重要的工程创新：

#### 确定性 Top-$k$ 算子

在 RL（强化学习）训练阶段，DSA 索引器的 Top-$k$ 选择必须使用确定性实现：

```python
# ✅ 正确：确定性实现
selected_indices = torch.topk(index_scores, k=2048, dim=-1).indices

# ❌ 错误：SGLang 的 CUDA 非确定性实现会导致 RL 几步后性能急剧退化
```

原因：非确定性选择的细微差异在 RL 的多步训练中被累积放大，导致策略崩溃（伴随熵值骤降）。

#### 索引器参数冻结

在 RL 阶段，GLM-5 **默认冻结索引器参数**，仅优化主模型：

```
RL 阶段:
  - 索引器参数: 冻结 ✓（加速训练 + 防止不稳定学习）
  - 主模型参数: 可训练
  - 类似 MoE 中的 routing replay，但 DSA 的 k=2048 远大于 MoE 的 k=8
```

### 6.4 GLM-5 的训练规模

| 阶段 | 训练量 | 上下文长度 |
|------|--------|-----------|
| 预训练 | 27T tokens | 4K → 32K |
| Mid-Training (阶段1) | 1T tokens | 32K |
| Mid-Training (阶段2) | 500B tokens | 128K |
| Mid-Training (阶段3) | 50B tokens | 200K |
| DSA 稀疏适配 | 20B tokens | 202,752 |
| **总计** | **~28.5T tokens** | |

### 6.5 SFT 阶段的 INT4 QAT

GLM-5 在 SFT 阶段应用了 INT4 量化感知训练（QAT），支持在国产芯片（华为昇腾、昆仑芯、摩尔线程等）上进行低精度推理。训练和离线量化内核保证逐比特一致。

---

## 7. 两大架构对比分析

### 7.1 架构参数总览

| 维度 | DeepSeek V3.2 | GLM-5 |
|------|---------------|-------|
| 总参数 | 671B | 744B |
| 激活参数 | 37B | 40B |
| 激活比 | 5.5% | 5.4% |
| 模型维度 $d$ | 7168 | 6144 |
| 总层数 | 61 | 78 |
| Dense 层 | 3 | 3 |
| MoE 层 | 58 | 75 |
| 注意力头数 | 128 | 64 |
| Q 压缩秩 | 1536 | 2048 |
| KV 压缩秩 | 512 | 512 |
| QK 头维度 (nope+rope) | 192 (128+64) | 256 (192+64) |
| V 头维度 | 128 | 256 |
| 词汇量 | 129,280 | 154,880 |
| 最大上下文 | 128K | 202,752 |
| MoE 激活专家数 | 8 | 8 |
| 共享专家数 | 2 | 1 |
| 专家隐藏维度 | 1408 | 2048 |

### 7.2 注意力机制对比

| 维度 | DeepSeek V3.2 | GLM-5 |
|------|---------------|-------|
| 注意力范式 | MLA (192 维头) | MLA-256 (256 维头) |
| 注意力总带宽 ($n_h \times d_h$) | 128×192 = 24,576 | 64×256 = 24,576 |
| 解码点积维度 | 192 | 256 |
| 优化器适配 | 标准 Muon | Muon Split |
| Q LoRA | d→1536→128×192 | d→2048→64×256 |
| KV LoRA | d→512+64→128×256 | d→512+64→64×512 |
| 每 token KV 缓存 (Absorb) | 576 dims | 576 dims |

**关键洞察**：GLM-5 通过减少头数、增大每头维度，在保持总注意力带宽不变的前提下，降低了解码阶段的点积计算量。此外，Muon Split 解决了原始 MLA 无法匹配 GQA-8 的性能问题。

### 7.3 DSA 实现对比

| 维度 | DeepSeek V3.2 | GLM-5 |
|------|---------------|-------|
| 索引器头数 | 32 | 32 |
| 索引器头维度 | 128 | 128 |
| Top-$k$ | 2048 | 2048 |
| DSA 训练量 | ~943.7B tokens | ~20B tokens |
| RL 阶段索引器 | — | 冻结 |
| 确定性 Top-$k$ | — | 显式要求 |

**关键洞察**：GLM-5 在远小于 DeepSeek V3.2 的 DSA 训练预算下（20B vs 943.7B tokens）实现了匹配稠密模型的性能，显示了更高效的 DSA 适配策略。

### 7.4 设计哲学差异

| 维度 | DeepSeek V3.2 | GLM-5 |
|------|---------------|-------|
| 设计重心 | 极致效率（更少的层、更小的激活比） | Agentic Engineering（更大容量、更长上下文） |
| MLA 策略 | 多小头（128×192） | 少大头（64×256） |
| MTP 策略 | 1 层独立参数 | 3 层参数共享 |
| 量化路线 | FP8 原生训练 | INT4 QAT + 国产芯片适配 |
| 上下文策略 | 128K | 202K（多阶段 mid-training） |

### 7.5 可视化：架构层次对比

```
DeepSeek V3.2                              GLM-5
═══════════════                            ═══════════
Embedding  129280→7168                     Embedding  154880→6144
  │                                          │
Layer 0-2:  Dense + MLA (DSA)              Layer 0-2:  Dense + MLA-256 (DSA)
Layer 3-60: MoE + MLA (DSA)    ×58         Layer 3-77: MoE + MLA-256 (DSA)    ×75
  │ 256 experts / 8 activated               │ 256 experts / 8 activated
  │ 2 shared experts                        │ 1 shared expert
  │ moe_inter_dim: 1408                     │ moe_inter_dim: 2048
  │ qk_head_dim: 192 (128+64)              │ qk_head_dim: 256 (192+64)
  │ n_heads: 128                            │ n_heads: 64
  │                                          │
RMSNorm                                    RMSNorm
  │                                          │
LM Head  7168→129280                       LM Head  6144→154880
  │                                          │
MTP: 1层 / 2 推测步 / 接受长度 2.55         MTP: 3层共享 / 4 推测步 / 接受长度 2.76
```

---

## 8. 总结与展望

### 8.1 核心要点回顾

1. **MLA 解决了 KV 缓存爆炸问题**：通过低秩分解，将 KV 缓存从 $\mathcal{O}(n_h \times d_h)$ 压缩到 $\mathcal{O}(d_c)$（缩减约 85 倍），使长上下文推理成为可能。

2. **DSA 解决了注意力的平方复杂度**：通过动态稀疏选择，将注意力计算从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(Lk)$，在 128K 上下文中实现约 5.5 倍的理论加速。

3. **MLA + DSA 的协同**：MLA 压缩了缓存大小，DSA 压缩了计算量。两者结合使长上下文推理的显存和延迟同时大幅降低。

4. **DeepSeek V3.2 的定位**：极致效率优先，671B 参数仅 37B 激活，128 个注意力头 × 192 维。

5. **GLM-5 的定位**：Agent 能力优先，744B 参数 / 40B 激活，MLA-256 + Muon Split，202K 上下文，3 层 MTP 参数共享。

### 8.2 关键公式汇总

| 名称 | 公式 |
|------|------|
| **MLA Query 压缩** | $\mathbf{q}_t^C = \mathbf{W}^{UQ}\cdot\text{RMSNorm}(\mathbf{W}^{DQ}\mathbf{h}_t)$ |
| **MLA KV 压缩** | $\mathbf{c}_t^{KV} = \mathbf{W}^{DKV}\mathbf{h}_t$ |
| **MLA 混合位置编码** | $\mathbf{q}_t = [\mathbf{q}_t^{\text{nope}} \;\|\; \text{RoPE}(\mathbf{q}_t^{\text{rope}}, t)]$ |
| **MLA Absorb 注意力** | $\text{score} = \frac{\mathbf{q}^{\text{nope}}(\mathbf{W}^{UK}\mathbf{KV})^\top + \mathbf{q}^{\text{rope}}(\mathbf{K}^{\text{pe}})^\top}{\sqrt{d_h}}$ |
| **DSA 索引分数** | $I_{t,s} = \sum_j w_{t,j}^I \cdot \text{ReLU}(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I)$ |
| **DSA 稀疏选择** | $\mathcal{S}_t = \{s \mid I_{t,s} \in \text{Top-}k(\{I_{t,s'}\}_{s'=1}^t)\}$ |
| **DSA 索引器 KL 损失** | $\mathcal{L}^I = \sum_t D_{\mathrm{KL}}(p_{t,\mathcal{S}_t} \|\ \text{Softmax}(I_{t,\mathcal{S}_t}))$ |
| **MLA 参数缩减比** | $\frac{d \times d_c + d_c \times h \times d_h}{d \times h \times d_h} \approx 28\%$ |
| **MLA KV 缓存压缩比** | $\frac{d_c + d_R}{2 \times h \times d_h} \approx 1.2\%$ |

### 8.3 未来方向

- **MLA 的进一步压缩**：是否可以探索更激进的低秩分解或结构化稀疏？
- **DSA 训练效率**：GLM-5 用 20B tokens 实现了匹配性能，是否可以进一步降低 DSA 适配成本？
- **端侧 MLA+DSA**：在移动端 / 端侧设备上，MLA 的缓存压缩和 DSA 的计算稀疏化是否可以实现高效的本地推理？
- **MLA+DSA 的量化协同**：低秩结构与量化（如 W4A8、INT4）如何协同优化？

---

> **参考文献**
>
> 1. DeepSeek-AI, "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models", arXiv:2512.02556, 2025.
> 2. DeepSeek-AI, "DeepSeek-V3 Technical Report", 2024.
> 3. Zhipu AI, "GLM-5: from Vibe Coding to Agentic Engineering", arXiv:2602.15763, 2025.
> 4. DeepSeek-AI, "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", Best Paper ACL 2025.
> 5. GLM-5 Model Configuration (config.json), HuggingFace: zai-org/GLM-5.
> 6. DeepSeek-V3.2 Model Configuration, HuggingFace: deepseek-ai/DeepSeek-V3.2.
