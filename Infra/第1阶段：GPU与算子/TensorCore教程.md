# CUDA Tensor Core 编程实战教程

> 以主流热点算子为线索，系统学习 Tensor Core 编程，并与 CUDA Core 实现进行对比。

---

## 目录

1. [前置知识：GPU 架构概览](#1-前置知识gpu-架构概览)
2. [CUDA Core 编程快速回顾](#2-cuda-core-编程快速回顾)
3. [Tensor Core 原理与编程模型](#3-tensor-core-原理与编程模型)
4. [算子实战：CUDA Core vs Tensor Core 双版本对比](#4-算子实战cuda-core-vs-tensor-core-双版本对比)
   - 4.1 [矩阵乘法 GEMM —— 算子之王](#41-矩阵乘法-gemm--算子之王)
   - 4.2 [Flash Attention —— 面试最高频](#42-flash-attention--面试最高频)
   - 4.3 [卷积 Conv2D —— 经典必考](#43-卷积-conv2d--经典必考)
   - 4.4 [Softmax —— CUDA 入门经典](#44-softmax--cuda-入门经典)
   - 4.5 [LayerNorm / RMSNorm —— LLM 必考](#45-layernorm--rmsnorm--llm-必考)
   - 4.6 [SwiGLU / FFN —— MoE 前馈网络](#46-swiglu--ffn--moe-前馈网络)
   - 4.7 [TopK / Gating —— MoE 路由算子](#47-topk--gating--moe-路由算子)
5. [性能调优实战 checklist](#5-性能调优实战-checklist)
6. [面试高频问题与回答思路](#6-面试高频问题与回答思路)
7. [推荐学习路线](#7-推荐学习路线)

---

## 1. 前置知识：GPU 架构概览

### 1.1 CUDA Core 与 Tensor Core 的物理关系

```
┌─────────────────────────────────────────────┐
│                    SM (Streaming Multiprocessor)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  CUDA    │  │  CUDA    │  │  CUDA    │  │  CUDA    │            │
│  │  Core ×32│  │  Core ×32│  │  Core ×32│  │  Core ×32│            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                       │                                               │
│              ┌────────┴────────┐                                     │
│              │  Tensor Core ×4 │  ← Volta+ (V100), Ampere (A100)     │
│              └────────┬────────┘     Hopper (H100) 含 TMA              │
│                       │                                               │
│  ┌─────────────────────────────────────────────┐                     │
│  │  Register File (64K × 32-bit per SM in A100)│                     │
│  └─────────────────────────────────────────────┘                     │
│  ┌─────────────────────────────────────────────┐                     │
│  │  Shared Memory / L1 Cache (192 KB per SM)    │                     │
│  └─────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────┘
```

**关键事实：**
- Tensor Core 不是独立单元，嵌入在 SM 中，一个 SM 有 4 个 Tensor Core（Volta/Turing/Ampere）
- Hopper (H100) 每个 SM 有 4 个 Tensor Core，但每个更强
- Tensor Core 执行 `D = A × B + C` 这个单一操作，但在 1 个时钟周期内完成

### 1.2 各代 Tensor Core 能力

| 架构 | GPU 型号 | mma 形状 | FP16 吞吐/TC | 精度支持 |
|------|---------|----------|-------------|---------|
| Volta | V100 | 16×16×16 (m8n8k4) | 64 FMA/cycle | FP16 |
| Turing | T4 | 16×16×16 | 64 FMA/cycle | FP16, INT8, INT4 |
| Ampere | A100 | 16×8×16 (sparse: 2x) | 256 FMA/cycle | BF16, TF32, FP16, INT8 |
| Hopper | H100 | 16×8×32 (wgmma) | 512 FMA/cycle | FP8, FP16, BF16, TF32 |

### 1.3 为什么需要 Tensor Core？

```
场景：计算 16×16 FP16 矩阵乘法 C = A × B

CUDA Core：16×16×16 = 4096 次 FMA，需 4096 个时钟周期（1 FMA/cycle/CUDA Core）

Tensor Core (A100)：
  - mma.sync.aligned.m16n8k16 指令
  - 1 个 Tensor Core 执行 16×8×16 = 2048 FMA / 1 cycle
  - 4 个 TC/SM → 8192 FMA/cycle/SM
  - 实际吞吐 ~312 TFLOPS FP16 (A100 标称)
  
理论加速比：~8-16x（实际受限于内存带宽时更低）
```

---

## 2. CUDA Core 编程快速回顾

### 2.1 线程层次与内存层次

```cuda
// 线程层次
gridDim    → 整个 grid 的维度         (kernel<<<gridDim, blockDim>>>)
blockIdx   → 当前 block 的索引
blockDim   → 每个 block 的线程数
threadIdx  → 当前线程在 block 中的索引

// 内存层次（从快到慢）
register     → 每个线程私有，最快，容量最小（~255 registers/thread）
shared mem   → block 内共享，~1.5 TB/s，192KB/SM (A100)
L1 cache     → 与 shared mem 共享硬件
L2 cache     → 所有 SM 共享，~40 MB (A100)
global mem   → HBM，~2 TB/s (A100 80GB)，最慢
```

### 2.2 典型 CUDA Core GEMM（朴素版）

这是最经典的基础面试题：手写 GEMM。

```cuda
// ============================================
// CUDA Core 朴素 GEMM: C[M×N] = A[M×K] × B[K×N]
// ============================================

__global__ void sgemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 每个线程负责 C 的一个元素
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**问题：** 全局内存访问次数 = `M × N × K × 2` 次读 + `M × N` 次写，访存比计算多得多。

### 2.3 CUDA Core 优化技巧清单

| 优化手段 | 原理 | 典型收益 |
|---------|------|---------|
| **共享内存分块 (Tiling)** | 将 A/B 矩阵分块加载到共享内存，减少全局内存访问 | 5-10x |
| **寄存器重用** | 每个线程计算多个输出元素，复用加载的数据 | 2-4x |
| **Bank conflict 消解** | shared memory 的 32 个 bank 避免同一 bank 多线程同时访问 | 1.2-1.5x |
| **向量化访问 (float4)** | 128-bit 对齐加载，减少指令数 | 1.5-2x |
| **Warp shuffle** | 线程束内直接交换寄存器数据，无需 shared memory | 减少同步开销 |
| **双缓冲** | 计算当前 tile 时异步预取下一个 tile | 隐藏延迟 |
| **warp-level 归约** | 使用 `__shfl_xor_sync` 在 warp 内做快速归约 | 2-3x |

### 2.4 共享内存分块 GEMM（优化版）

```cuda
// ============================================
// CUDA Core 共享内存分块 GEMM
// 核心思想：将 M×K、K×N 矩阵切分为 tile，每个 tile 加载到 shared memory
// ============================================

#define TILE_SIZE 32  // 每个 block 处理 32×32 的 C 子矩阵
#define BLOCK_ROWS 8  // 每个线程处理 8×8 元素块

__global__ void sgemm_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 共享内存：存储 A 和 B 的 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 当前 block 负责的 C 子矩阵起始位置
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    // 线程在 tile 内的局部位置
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;

    // 累积器（寄存器内）
    float acc[BLOCK_ROWS][BLOCK_ROWS] = {0.0f};

    // 遍历 K 维度的所有 tile
    for (int tile = 0; tile < K; tile += TILE_SIZE) {

        // 协作加载 A 的 tile 到共享内存
        int a_row = block_row + ty;
        int a_col = tile + tx;
        As[ty][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

        // 协作加载 B 的 tile 到共享内存
        int b_row = tile + ty;
        int b_col = block_col + tx;
        Bs[ty][tx] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();  // 确保整个 tile 加载完毕

        // 每个线程计算 BLOCK_ROWS×BLOCK_ROWS 个输出（寄存器重用）
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < BLOCK_ROWS; i++) {
                float a_val = As[ty * BLOCK_ROWS + i][k];
                for (int j = 0; j < BLOCK_ROWS; j++) {
                    acc[i][j] += a_val * Bs[k][tx * BLOCK_ROWS + j];
                }
            }
        }

        __syncthreads();  // 确保 tile 不会被下一个迭代覆盖
    }

    // 写回全局内存
    for (int i = 0; i < BLOCK_ROWS; i++) {
        for (int j = 0; j < BLOCK_ROWS; j++) {
            int row = block_row + ty * BLOCK_ROWS + i;
            int col = block_col + tx * BLOCK_ROWS + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}
```

---

## 3. Tensor Core 原理与编程模型

### 3.1 Tensor Core 的数学本质

Tensor Core 执行的是一个**矩阵乘加**运算：

$$D_{m \times n} = A_{m \times k} \times B_{k \times n} + C_{m \times n}$$

在 1 个时钟周期内完成 $m \times n \times k$ 次浮点运算。

```
示意图（16×8×16，Ampere）：
                                         
       A[16×16]              B[16×8]            C/D[16×8]
    ┌──────┬──────┐      ┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐
    │ a00  │ a01..│      │b0│b1│b2│b3│      │c0│c1│c2│c3│
    ├──────┼──────┤      ├──┼──┼──┼──┤      ├──┼──┼──┼──┤
    │ a10  │ ...  │  ×   │..│..│..│..│  +   │..│..│..│..│
    ├──────┼──────┤      ├──┼──┼──┼──┤      ├──┼──┼──┼──┤
    │ ...  │      │      │. │. │. │. │      │..│..│..│..│
    └──────┴──────┘      └──┴──┴──┴──┘      └──┴──┴──┴──┘
    
    一次 MMA 指令：D[16×8] += A[16×16] × B[16×8]
    这个操作在 1 cycle 完成！（Ampere 架构）
```

### 3.2 PTX MMA 指令详解

Tensor Core 编程的核心是 PTX 级别的 `mma` 指令（或从 CUDA C++ 使用 WMMA API）。

```cuda
// PTX 内联汇编形式（最底层控制）
// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//
// 解析：
//   m16n8k16    → M=16, N=8, K=16（矩阵形状）
//   row.col     → A 按行主序，B 按列主序
//   f16.f16.f16.f16 → A/B/C/D 均为 FP16
//
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0, %1, %2, %3}, "     // D 矩阵（4个f16x2寄存器 = 8个元素）
    "{%4, %5, %6, %7}, "     // A 矩阵（4个f16x2寄存器，每行1个）
    "{%8, %9}, "             // B 矩阵（2个f16x2寄存器，每列1个）
    "{%10, %11, %12, %13};"  // C 矩阵（累加器）
    : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
    :  "r"(a0),"r"(a1),"r"(a2),"r"(a3),
       "r"(b0),"r"(b1),
       "r"(c0),"r"(c1),"r"(c2),"r"(c3)
);
```

**Ampere 架构支持的 MMA 形状：**

| 指令 | M | N | K | 适用场景 |
|------|---|---|---|---------|
| `.m16n8k16` | 16 | 8 | 16 | 通用 GEMM |
| `.m16n8k8` | 16 | 8 | 8 | TF32 精度 |
| `.m8n8k4` | 8 | 8 | 4 | 小矩阵（双精度） |
| `.m16n8k32` | 16 | 8 | 32 | INT8 |
| `.m16n8k64` | 16 | 8 | 64 | INT4 |

### 3.3 WMMA API（高层 API，推荐入门）

WMMA (Warp Matrix Multiply-Accumulate) 是 CUDA 提供的高层 API，封装了 PTX mma 指令。

```cuda
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// WMMA 声明一个 16×16×16 的 FP16 矩阵乘法片段
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

// 从内存加载到 fragment（每个 warp 内的线程协作加载）
wmma::load_matrix_sync(a_frag, A_ptr, lda);
wmma::load_matrix_sync(b_frag, B_ptr, ldb);

// 执行矩阵乘法
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 存储结果
wmma::store_matrix_sync(C_ptr, c_frag, ldc, wmma::mem_row_major);
```

**WMMA 支持的矩阵形状（Ampere）：**

| 精度 | M×N×K | fragment 大小 |
|------|-------|--------------|
| FP16 | 16×16×16 | 8 个 f16×2 per fragment |
| BF16 | 16×16×16 | 同 FP16 |
| TF32 | 16×16×8 | 4 个 f32 per fragment |
| FP64 | 8×8×4 | 1 个 f64 per fragment |

**WMMA 的边界限制：**
- `matrix_a` 必须是 `row_major`，`matrix_b` 必须是 `col_major`
- `accumulator` 固定 `row_major`
- 每个 warp 内的所有线程必须**同步**调用（`__syncwarp()`）

### 3.4 WMMA 布局细节（面试常问）

这是面试中非常高频的细节问题：**WMMA fragment 的线程持有方式**。

```
FP16 16×16×16 WMMA fragment 布局：

thread 0 (warp lane 0) 持有：
  A fragment: A[0:8, 0:1] 的 8 个 half（每行 8 个，1 列 → 8 个 half）
  B fragment: B[0:1, 0:8] 的 8 个 half
  C fragment: C[0:8, 0:1] 的 8 个 half

实际上 32 个线程按特定映射持有矩阵的不同部分
具体映射由 NVIDIA 内部定义，但规律是：
  - 每线程持有 fragment 中的 8 个 half（= 4×32-bit 寄存器）
  - 线程 ID 决定了在矩阵中的哪个子区域
```

**关键理解：** 你不需要在 WMMA 层级别手动排列 fragment 内的元素，`load_matrix_sync` 和 `store_matrix_sync` 会自动处理排列。只有在使用 PTX `mma` 指令时才需要手动按约定排列寄存器。

---

## 4. 算子实战：CUDA Core vs Tensor Core 双版本对比

### 4.1 矩阵乘法 GEMM —— 算子之王

**面试场景：** "请手写一个高性能 GEMM，考虑 Tensor Core 加速"

#### 4.1.1 CUDA Core 版本（复习）

前面 2.4 节已给出，这里补充一个**更优版本**：每个线程计算 8×8 子块 + float4 向量化加载。

```cuda
// ============================================
// CUDA Core 高性能 GEMM（面试参考实现）
// 关键优化：
//   1. 共享内存双缓冲
//   2. float4 向量化加载（128-bit）
//   3. 每个线程计算 8×8 = 64 个元素
//   4. 寄存器预取
// ============================================

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 8

__global__ void sgemm_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 双缓冲共享内存
    __shared__ float4 As[TILE_K][TILE_M / 4];  // float4 存储
    __shared__ float4 Bs[TILE_K][TILE_N / 4];

    // 线程映射
    int tid_m = threadIdx.x / (TILE_N / THREAD_N);
    int tid_n = threadIdx.x % (TILE_N / THREAD_N);

    float accum[THREAD_M][THREAD_N] = {0.0f};
    float frag_a[THREAD_M];
    float frag_b[THREAD_N];

    int global_m = blockIdx.y * TILE_M + tid_m * THREAD_M;
    int global_n = blockIdx.x * TILE_N + tid_n * THREAD_N;

    for (int k_block = 0; k_block < K; k_block += TILE_K) {
        // 协作加载 A tile → float4
        for (int i = threadIdx.x; i < TILE_K * TILE_M / 4; i += blockDim.x) {
            int k = i / (TILE_M / 4);
            int m = i % (TILE_M / 4);
            int a_row = blockIdx.y * TILE_M + m * 4;
            int a_col = k_block + k;
            if (a_row < M && a_col < K) {
                As[k][m] = *reinterpret_cast<const float4*>(&A[a_row * K + a_col]);
            } else {
                As[k][m] = make_float4(0,0,0,0);
            }
        }

        // 协作加载 B tile
        for (int i = threadIdx.x; i < TILE_K * TILE_N / 4; i += blockDim.x) {
            int k = i / (TILE_N / 4);
            int n = i % (TILE_N / 4);
            int b_row = k_block + k;
            int b_col = blockIdx.x * TILE_N + n * 4;
            if (b_row < K && b_col < N) {
                Bs[k][n] = *reinterpret_cast<const float4*>(&B[b_row * N + b_col]);
            } else {
                Bs[k][n] = make_float4(0,0,0,0);
            }
        }

        __syncthreads();

        // 计算：每个线程 8×8 子块
        for (int k = 0; k < TILE_K; k++) {
            // 预取 A 的一行到寄存器
            for (int m = 0; m < THREAD_M; m++) {
                float4 a4 = As[k][tid_m * THREAD_M / 4 + m / 4];
                frag_a[m] = ((float*)&a4)[m % 4];
            }
            // 预取 B 的一行到寄存器
            for (int n = 0; n < THREAD_N; n++) {
                float4 b4 = Bs[k][tid_n * THREAD_N / 4 + n / 4];
                frag_b[n] = ((float*)&b4)[n % 4];
            }
            // FMA
            for (int m = 0; m < THREAD_M; m++) {
                for (int n = 0; n < THREAD_N; n++) {
                    accum[m][n] += frag_a[m] * frag_b[n];
                }
            }
        }

        __syncthreads();
    }

    // 写回
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            int c_row = global_m + m;
            int c_col = global_n + n;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = accum[m][n];
            }
        }
    }
}
```

#### 4.1.2 Tensor Core 版本 —— WMMA API

```cuda
// ============================================
// Tensor Core FP16 GEMM（WMMA API）
// 核心：每个 warp 独立计算一个 16×16×16 的矩阵乘加
// ============================================

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 每个 block 使用多个 warp，每个 warp 处理一个 16×16 子块
// block 维度：ceil(M/16)×ceil(N/16) 个 warp → TILE_M×TILE_N 的 block
#define TILE_M 64
#define TILE_N 64
#define WARPS_PER_BLOCK (TILE_M * TILE_N / (WMMA_M * WMMA_N))  // = 16 warps

__global__ void hgemm_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // 每个 warp 独立工作：计算 C[16×16] 子块
    // warp 在 block 内的映射
    int warp_id = threadIdx.x / 32;
    int warp_m = (warp_id / (TILE_N / WMMA_N)) * WMMA_M;  // warp 负责的 M 偏移
    int warp_n = (warp_id % (TILE_N / WMMA_N)) * WMMA_N;  // warp 负责的 N 偏移

    int global_m = blockIdx.y * TILE_M + warp_m;
    int global_n = blockIdx.x * TILE_N + warp_n;

    // 声明 WMMA fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 维度滑动
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载 A 的 16×16 tile（row-major）
        // 注意：A 的 leading dimension = K（因为 A 是 M×K）
        wmma::load_matrix_sync(a_frag, A + global_m * K + k, K);

        // 加载 B 的 16×16 tile（column-major 意味原存储是 row-major N×K 的转置）
        // B 存储为 K×N（row-major），但 WMMA 需要 B 是列主序
        // 实际加载：取 B[k:N][global_n:global_n+16]，leading dim = N
        wmma::load_matrix_sync(b_frag, B + k * N + global_n, N);

        // 执行 MMA：C += A × B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储结果
    wmma::store_matrix_sync(C + global_m * N + global_n, c_frag, N, wmma::mem_row_major);
}
```

**CUDA Core vs Tensor Core GEMM 对比：**

| 维度 | CUDA Core (float4 tiled) | Tensor Core (WMMA FP16) |
|------|-------------------------|------------------------|
| 代码复杂度 | 中等（需手动分块、向量化） | 较低（API 封装分块逻辑） |
| 单 SM 理论吞吐 (A100) | 19.5 TFLOPS FP32 | 312 TFLOPS FP16 (~16×) |
| 实际加速比 | baseline | 5-12×（取决于矩阵大小） |
| 内存带宽敏感度 | 高 | 更高（计算太快，更容易带宽瓶颈） |
| 精度 | FP32 | FP16（需处理数值范围） |

#### 4.1.3 Tensor Core 进阶 —— PTX MMA 指令（面试加分项）

面试中最能体现深度的是直接使用 PTX mma 指令：

```cuda
#include <cuda_fp16.h>

// ============================================
// PTX MMA GEMM：每个 warp 独立执行 mma 指令
// A100 Ampere 架构 m16n8k16 FP16
// ============================================

__global__ void hgemm_ptx_mma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // 每个 warp 计算 C[16×8] 子块
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) * 16;
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    // 寄存器分配（PTX 层面操作 32-bit 寄存器对）
    // A fragment: 4 个寄存器, 每个寄存器存 2 个 FP16
    // B fragment: 2 个寄存器
    // C/D fragment: 4 个寄存器（累加器）
    uint32_t a[4], b[2], c[4];

    // 初始化累加器
    c[0] = c[1] = c[2] = c[3] = 0;

    for (int k_idx = 0; k_idx < K; k_idx += 16) {
        // === 手动加载 A[16×16] 到寄存器 ===
        // A 在全局内存中是 M×K row-major
        // 每个线程加载不同的元素（warp 级协作）
        int lane_id = threadIdx.x % 32;

        // A 的 16×16 矩阵按特定映射分配到 32 个线程
        // Volta/Ampere layout: 每线程 4 组 f16×2 = 8 个 half
        // 映射规则（简化）：
        //   group_id = lane_id / 4  (0-7)
        //   col_in_group = lane_id % 4
        //   加载 A[group_id*2 + 0, col_in_group*2 + k_idx] 等

        // 实际简化版：当前 warp 内所有线程协作加载
        for (int i = 0; i < 4; i++) {
            int mm = lane_id / 4 + i * 8;     // 行
            int kk = lane_id % 4;              // 列组
            half2 val = __half2half2(__float2half(0.0f));
            if (warp_m + mm < M && k_idx + kk * 2 < K) {
                half v0 = A[(warp_m + mm) * K + k_idx + kk * 2];
                half v1 = A[(warp_m + mm) * K + k_idx + kk * 2 + 1];
                val = __halves2half2(v0, v1);
            }
            a[i] = *reinterpret_cast<uint32_t*>(&val);
        }

        // === 手动加载 B[16×8] 到寄存器 ===
        // B 存储为 K×N row-major
        for (int i = 0; i < 2; i++) {
            int kk = lane_id / 4 + i * 8;
            int nn = lane_id % 4;
            half2 val = __half2half2(__float2half(0.0f));
            if (k_idx + kk < K && warp_n + nn * 2 < N) {
                half v0 = B[(k_idx + kk) * N + warp_n + nn * 2];
                half v1 = B[(k_idx + kk) * N + warp_n + nn * 2 + 1];
                val = __halves2half2(v0, v1);
            }
            b[i] = *reinterpret_cast<uint32_t*>(&val);
        }

        // === PTX MMA 指令 ===
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
        );
    }

    // === 存回全局内存 ===
    for (int i = 0; i < 4; i++) {
        int mm = lane_id / 4 + i * 8;
        int nn = lane_id % 4;
        half2 val = *reinterpret_cast<half2*>(&c[i]);
        if (warp_m + mm < M && warp_n + nn * 2 < N) {
            C[(warp_m + mm) * N + warp_n + nn * 2] = __low2half(val);
            C[(warp_m + mm) * N + warp_n + nn * 2 + 1] = __high2half(val);
        }
    }
}
```

> **注意：** 实际生产中的 PTX MMA GEMM（如 CUTLASS）会使用 shared memory 做 double buffering，这里为展示 PTX 指令本身而简化。PTX MMA 的寄存器映射细节很复杂，面试中说出思路比完全手写更重要。

---

### 4.2 Flash Attention —— 面试最高频

**面试场景：** "讲讲 Flash Attention 的原理，以及怎么用 CUDA/Tensor Core 实现"

#### 4.2.1 原理简述

标准 Attention：
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

朴素实现需要存储完整的 $S = QK^T \in \mathbb{R}^{N \times N}$ 矩阵（$O(N^2)$ 显存）。

**Flash Attention 核心思想（tiling + recomputation）：**
1. 将 Q 按行分块（分块大小 $B_r$），K/V 按列分块（分块大小 $B_c$）
2. 对每个 Q 块，流式处理 K/V 块，不存储完整 S
3. 使用 online softmax 算法增量更新
4. 反向传播时重新计算 S，而非从显存读取

#### 4.2.2 Online Softmax（核心子问题）

这是 Flash Attention 中使用 Tensor Core 最关键的子部分。

```cuda
// ============================================
// Online Softmax：在流式处理中维护 running max 和 sum
// ============================================

__global__ void online_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ running_max,
    float* __restrict__ running_sum,
    int N, int D)
{
    // 每个 block 处理一个 query 向量（Q 的一行）
    int q_idx = blockIdx.x;

    // 分块处理 K 的列
    for (int j = 0; j < N; j += BLOCK_SIZE) {
        // 1. 计算 S_ij = Q_i @ K_j^T（使用 Tensor Core GEMM）
        // 2. 找到局部最大值 m_ij = max(S_ij)
        // 3. 更新全局最大值：
        //    m_new = max(m_old, m_ij)
        //    scale = exp(m_old - m_new)
        //    P_new = P_old * scale + exp(S_ij - m_new)
        // 4. 累加 V：O_new = O_old * scale + P_new * V_j
    }
    // 5. 最终归一化：O /= sum_exp
}
```

#### 4.2.3 Flash Attention Forward（简化版）

```cuda
// ============================================
// Flash Attention Forward（简化实现）
// 核心：Q×K 用 Tensor Core，online softmax 用 warp shuffle
// ============================================

#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define HEAD_DIM 64      // d_head（必须 >= 16 才能用 WMMA）
#define Br 32            // Q 行分块大小
#define Bc 32            // K/V 列分块大小

__global__ void flash_attention_fwd_kernel(
    const half* __restrict__ Q,   // [batch, heads, seq_q, head_dim]
    const half* __restrict__ K,   // [batch, heads, seq_k, head_dim]
    const half* __restrict__ V,   // [batch, heads, seq_k, head_dim]
    half* __restrict__ O,         // [batch, heads, seq_q, head_dim]
    int seq_q, int seq_k)
{
    // 共享内存：Q tile, K tile, V tile
    __shared__ half Qs[Br][HEAD_DIM];
    __shared__ half Ks[Bc][HEAD_DIM];
    __shared__ half Vs[Bc][HEAD_DIM];

    // 该 block 处理的 Q 行
    int q_start = blockIdx.x * Br;
    int q_end = min(q_start + Br, seq_q);

    // 加载 Q tile 到共享内存
    for (int i = threadIdx.x; i < Br * HEAD_DIM; i += blockDim.x) {
        int r = i / HEAD_DIM, c = i % HEAD_DIM;
        Qs[r][c] = (q_start + r < seq_q) ? Q[(q_start + r) * HEAD_DIM + c] : __float2half(0.0f);
    }
    __syncthreads();

    // 每个线程处理一个 Q 行（简化：warp 级并行更好但复杂）
    int local_q = threadIdx.x / (HEAD_DIM / WMMA_M);  // 每个线程一个 Q 行
    if (local_q >= Br) return;

    // O 累加器（寄存器）：HEAD_DIM 个 half
    half O_acc[HEAD_DIM] = {__float2half(0.0f)};
    float m_old = -INFINITY;  // running max（FP32 精度）
    float l_old = 0.0f;       // running sum

    // WMMA fragment 用于 Q×K
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> s_frag;

    // 沿 K 维度分块扫描
    for (int kv_start = 0; kv_start < seq_k; kv_start += Bc) {
        int kv_end = min(kv_start + Bc, seq_k);
        int kv_len = kv_end - kv_start;

        // 加载 K 和 V tile 到共享内存
        for (int i = threadIdx.x; i < Bc * HEAD_DIM; i += blockDim.x) {
            int r = i / HEAD_DIM, c = i % HEAD_DIM;
            Ks[r][c] = (r < kv_len) ? K[(kv_start + r) * HEAD_DIM + c] : __float2half(0.0f);
            Vs[r][c] = (r < kv_len) ? V[(kv_start + r) * HEAD_DIM + c] : __float2half(0.0f);
        }
        __syncthreads();

        // === Step 1: S = Q[local_q] × K^T ===
        wmma::fill_fragment(s_frag, 0.0f);
        for (int d = 0; d < HEAD_DIM; d += WMMA_K) {
            wmma::load_matrix_sync(q_frag, &Qs[local_q][d], HEAD_DIM);
            // K 需要列主序加载（转置）
            // 简化：逐行复制到临时 buffer
            // 实际实现会用 shared memory swizzle
            extern __shared__ half K_transposed[];
            for (int k = 0; k < kv_len; k++) {
                for (int dd = 0; dd < WMMA_K && d + dd < HEAD_DIM; dd++) {
                    K_transposed[(d + dd) * Bc + k] = Ks[k][d + dd];
                }
            }
            wmma::load_matrix_sync(k_frag, &K_transposed[d * Bc], Bc);
            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }

        // === Step 2: Online Softmax 更新 ===
        // 将 WMMA fragment 中的 S 值读取到寄存器
        half s_vals[Bc];
        wmma::store_matrix_sync(s_vals, s_frag, Bc, wmma::mem_row_major);

        // 找到局部最大值
        float m_new = m_old;
        for (int k = 0; k < kv_len; k++) {
            float val = __half2float(s_vals[k]) * __fdividef(1.0f, sqrtf((float)HEAD_DIM));
            m_new = fmaxf(m_new, val);
        }

        // 计算 softmax 分母修正
        float scale = expf(m_old - m_new);
        float l_new = l_old * scale;

        // 累加 P × V
        for (int k = 0; k < kv_len; k++) {
            float s_val = __half2float(s_vals[k]) * __fdividef(1.0f, sqrtf((float)HEAD_DIM));
            float p_val = expf(s_val - m_new);
            l_new += p_val;

            for (int d = 0; d < HEAD_DIM; d++) {
                O_acc[d] = __float2half(
                    __half2float(O_acc[d]) * scale + p_val * __half2float(Vs[k][d])
                );
            }
        }

        m_old = m_new;
        l_old = l_new;

        __syncthreads();
    }

    // === Step 3: 归一化 O /= l ===
    for (int d = 0; d < HEAD_DIM; d++) {
        O[(q_start + local_q) * HEAD_DIM + d] =
            __float2half(__half2float(O_acc[d]) / l_old);
    }
}
```

#### 4.2.4 CUDA Core 版本对比

对于 Flash Attention，CUDA Core 版本的主要区别在于 **Q×K 的计算方式**：

```cuda
// CUDA Core 版本的核心差异：Q×K 用 FMA 而非 WMMA
// 将上面的 WMMA 部分替换为：

for (int k = 0; k < kv_len; k++) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
        dot += __half2float(Qs[local_q][d]) * __half2float(Ks[k][d]);
    }
    s_vals[k] = __float2half(dot * __fdividef(1.0f, sqrtf((float)HEAD_DIM)));
}
```

**对比总结：**

| 维度 | CUDA Core Flash Attn | Tensor Core Flash Attn |
|------|---------------------|----------------------|
| Q×K 计算 | FMA 循环 | WMMA mma_sync |
| 适用 head_dim | 任意（16/32/64/128) | 需 ≥ 16 且对齐 |
| 短序列性能 | 接近 | Tensor Core 略好 |
| 长序列性能 | 瓶颈在 Q×K | Q×K 不再是瓶颈 |
| 实现复杂度 | 简单 | 中等（需处理 WMMA 布局） |

---

### 4.3 卷积 Conv2D —— 经典必考

**面试场景：** "用 Tensor Core 实现 2D 卷积，说说跟 CUDA Core 的区别"

#### 4.3.1 核心思路：Img2Col 转 GEMM

$$
\text{Conv2D}(I, W) \xrightarrow{\text{img2col}} \text{GEMM}(I_{\text{col}}, W)
$$

```
输入 I: [C, H, W]
卷积核 W: [K, C, R, S]
输出 O: [K, H', W']

Img2Col：将每个 R×S×C 的滑动窗口展开为一行
  I_col: [H'×W', R×S×C]  → 每个位置是一个窗口的所有像素
  W     : [K, R×S×C]     → 每个输出通道是一个权重向量

GEMM: O = I_col × W^T  → [H'×W', K]
```

一旦转成 GEMM，就可以直接复用 4.1 节的 Tensor Core GEMM。

#### 4.3.2 直接卷积的 CUDA Core 实现

```cuda
// ============================================
// CUDA Core 3×3 Conv2D（直接卷积）
// 输入：NCHW, 输出：NKPQ
// ============================================

__global__ void conv2d_3x3_cuda_core(
    const float* __restrict__ input,   // [N, C, H, W]
    const float* __restrict__ weight,  // [K, C, 3, 3]
    float* __restrict__ output,        // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q,
    int stride, int padding)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    int p = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z % K;
    int n = blockIdx.z / K;

    if (n < N && p < P && q < Q) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                    int h_in = p * stride + r - padding;
                    int w_in = q * stride + s - padding;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        float i_val = input[((n * C + c) * H + h_in) * W + w_in];
                        float w_val = weight[((k * C + c) * R + r) * S + s];
                        sum += i_val * w_val;
                    }
                }
            }
        }
        output[((n * K + k) * P + p) * Q + q] = sum;
    }
}
```

#### 4.3.3 Tensor Core 版本（Img2Col + WMMA GEMM）

```cuda
// ============================================
// Tensor Core Conv2D = Img2Col + WMMA GEMM
// Step 1: Img2Col 将卷积转化为 GEMM（CUDA kernel）
// Step 2: 调用 WMMA GEMM kernel（复用 4.1.2）
// ============================================

// Step 1: Img2Col kernel
__global__ void img2col_kernel(
    const half* __restrict__ input,   // [N, C, H, W]
    half* __restrict__ col_buffer,    // [N, P*Q, C*R*S]
    int N, int C, int H, int W,
    int R, int S, int P, int Q,
    int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * P * Q;

    if (idx < total) {
        int n = idx / (P * Q);
        int resid = idx % (P * Q);
        int p = resid / Q;
        int q = resid % Q;

        int c_rs = C * R * S;
        for (int c = 0; c < C; c++) {
            for (int r = 0; r < R; r++) {
                for (int s = 0; s < S; s++) {
                    int h_in = p * stride + r - padding;
                    int w_in = q * stride + s - padding;
                    int col_idx = c * R * S + r * S + s;
                    half val = __float2half(0.0f);
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        val = input[((n * C + c) * H + h_in) * W + w_in];
                    }
                    col_buffer[idx * c_rs + col_idx] = val;
                }
            }
        }
    }
}

// Step 2: 复用之前的 WMMA GEMM
// O[P*Q × K] = I_col[P*Q × C*R*S] × W^T[C*R*S × K]
// 直接调用 4.1.2 的 hgemm_wmma kernel
```

**对比总结：**

| 维度 | CUDA Core 直接卷积 | Tensor Core (Img2Col+GEMM) |
|------|-------------------|---------------------------|
| 额外显存 | 无 | col_buffer: O(P×Q×C×R×S) |
| 内存访问模式 | 不规则（跨步读取） | 规则（连续 GEMM） |
| Tensor Core 利用率 | 不可用 | 高 |
| 适用场景 | 小卷积核 (1×1, 3×3) | 任意卷积核 |
| 工程实践 | cuDNN implicit gemm | cuDNN/CUTLASS 默认 |

---

### 4.4 Softmax —— CUDA 入门经典

**面试场景：** "写一个高效的 CUDA Softmax，分析数值稳定性"

#### 4.4.1 CUDA Core 版本

标准的三步走：max → exp → normalize

```cuda
// ============================================
// CUDA Core Softmax（safe version）
// 每个 block 处理一个 token（一行）
// ============================================

__global__ void softmax_cuda_core(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x;

    // Step 1: 找最大值（warp reduce）
    __shared__ float s_max;
    float local_max = -INFINITY;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, input[row * cols + i]);
    }

    // Warp 级归约求最大值
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    float max_val = s_max;

    // Step 2: 计算 exp(x - max) 和 sum
    __shared__ float s_sum;
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float exp_val = expf(input[row * cols + i] - max_val);
        output[row * cols + i] = exp_val;
        local_sum += exp_val;
    }

    // Warp 归约求和
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();
    float sum_val = s_sum;

    // Step 3: 归一化
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        output[row * cols + i] = __fdividef(output[row * cols + i], sum_val);
    }
}
```

#### 4.4.2 Tensor Core 能加速 Softmax 吗？

**答案：不能直接加速。** Softmax 是非线性运算，Tensor Core 只能做矩阵乘加。

但可以优化前置的 QK^T：
- **QK^T 计算** → Tensor Core GEMM
- **Softmax 本身** → CUDA Core（或 warp shuffle + math intrinsics）

这就是 Flash Attention 的设计思路：Tensor Core 做 GEMM，CUDA Core 做 softmax。

---

### 4.5 LayerNorm / RMSNorm —— LLM 必考

**面试场景：** "写一个高效的 CUDA LayerNorm 和 RMSNorm"

#### 4.5.1 RMSNorm 原理

RMSNorm 是 LLaMA 等模型使用的归一化方式，比 LayerNorm 更简单：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

只有两个操作：求平方和 + 逐元素除法。

#### 4.5.2 CUDA Core RMSNorm

```cuda
// ============================================
// CUDA Core RMSNorm（warp shuffle optimize）
// 每个 block 处理一个 token
// ============================================

__global__ void rmsnorm_cuda_core(
    const half* __restrict__ input,    // [tokens, dim]
    const half* __restrict__ gamma,    // [dim]
    half* __restrict__ output,         // [tokens, dim]
    int tokens, int dim, float eps)
{
    int token_idx = blockIdx.x;

    // Step 1: 并行求平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[token_idx * dim + i]);
        sum_sq += val * val;
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    // Broadcast sum_sq from lane 0
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    // Step 2: 计算 rms 并归一化
    float rms = rsqrtf(sum_sq / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[token_idx * dim + i]);
        float g = __half2float(gamma[i]);
        output[token_idx * dim + i] = __float2half(val * rms * g);
    }
}
```

#### 4.5.3 向量化内存访问优化

```cuda
// ============================================
// RMSNorm float4 向量化版本
// 用 float4 (128-bit) 加载，减少指令数
// ============================================

__global__ void rmsnorm_float4(
    const half* __restrict__ input,    // [tokens, dim]
    const half* __restrict__ gamma,    // [dim]
    half* __restrict__ output,         // [tokens, dim]
    int tokens, int dim, float eps)
{
    int token_idx = blockIdx.x;

    // 用 float2 加载（2 个 half = 1 个 float2 = 32-bit）
    // 每个线程处理多个 half
    const int vec_size = 8;  // 每个线程一次处理 8 个 half = 4 个 float2

    float sum_sq = 0.0f;

    // 向量化求和
    for (int i = threadIdx.x * vec_size; i < dim; i += blockDim.x * vec_size) {
        const half* in_ptr = input + token_idx * dim + i;
        float2 v0 = *reinterpret_cast<const float2*>(in_ptr);
        float2 v1 = *reinterpret_cast<const float2*>(in_ptr + 2);
        float2 v2 = *reinterpret_cast<const float2*>(in_ptr + 4);
        float2 v3 = *reinterpret_cast<const float2*>(in_ptr + 6);

        half2 h0 = *reinterpret_cast<half2*>(&v0);
        half2 h1 = *reinterpret_cast<half2*>(&v1);
        half2 h2 = *reinterpret_cast<half2*>(&v2);
        half2 h3 = *reinterpret_cast<half2*>(&v3);

        float f[8] = {
            __half2float(h0.x), __half2float(h0.y),
            __half2float(h1.x), __half2float(h1.y),
            __half2float(h2.x), __half2float(h2.y),
            __half2float(h3.x), __half2float(h3.y)
        };
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            sum_sq += f[j] * f[j];
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }
    sum_sq = __shfl_sync(0xffffffff, sum_sq, 0);

    float rms = rsqrtf(sum_sq / dim + eps);

    // 向量化写回
    for (int i = threadIdx.x * vec_size; i < dim; i += blockDim.x * vec_size) {
        const half* in_ptr = input + token_idx * dim + i;
        half* out_ptr = output + token_idx * dim + i;
        const half* g_ptr = gamma + i;

        float2 gv0 = *reinterpret_cast<const float2*>(g_ptr);
        float2 gv1 = *reinterpret_cast<const float2*>(g_ptr + 2);
        float2 gv2 = *reinterpret_cast<const float2*>(g_ptr + 4);
        float2 gv3 = *reinterpret_cast<const float2*>(g_ptr + 6);

        half2 gh0 = *reinterpret_cast<half2*>(&gv0);
        half2 gh1 = *reinterpret_cast<half2*>(&gv1);
        half2 gh2 = *reinterpret_cast<half2*>(&gv2);
        half2 gh3 = *reinterpret_cast<half2*>(&gv3);

        float2 iv0 = *reinterpret_cast<const float2*>(in_ptr);
        float2 iv1 = *reinterpret_cast<const float2*>(in_ptr + 2);
        float2 iv2 = *reinterpret_cast<const float2*>(in_ptr + 4);
        float2 iv3 = *reinterpret_cast<const float2*>(in_ptr + 6);

        half2 ih0 = *reinterpret_cast<half2*>(&iv0);
        half2 ih1 = *reinterpret_cast<half2*>(&iv1);
        half2 ih2 = *reinterpret_cast<half2*>(&iv2);
        half2 ih3 = *reinterpret_cast<half2*>(&iv3);

        half2 res[4];
        res[0] = __halves2half2(
            __float2half(__half2float(ih0.x) * rms * __half2float(gh0.x)),
            __float2half(__half2float(ih0.y) * rms * __half2float(gh0.y))
        );
        res[1] = __halves2half2(
            __float2half(__half2float(ih1.x) * rms * __half2float(gh1.x)),
            __float2half(__half2float(ih1.y) * rms * __half2float(gh1.y))
        );
        res[2] = __halves2half2(
            __float2half(__half2float(ih2.x) * rms * __half2float(gh2.x)),
            __float2half(__half2float(ih2.y) * rms * __half2float(gh2.y))
        );
        res[3] = __halves2half2(
            __float2half(__half2float(ih3.x) * rms * __half2float(gh3.x)),
            __float2half(__half2float(ih3.y) * rms * __half2float(gh3.y))
        );

        *reinterpret_cast<float2*>(out_ptr)     = *reinterpret_cast<float2*>(&res[0]);
        *reinterpret_cast<float2*>(out_ptr + 2) = *reinterpret_cast<float2*>(&res[1]);
        *reinterpret_cast<float2*>(out_ptr + 4) = *reinterpret_cast<float2*>(&res[2]);
        *reinterpret_cast<float2*>(out_ptr + 6) = *reinterpret_cast<float2*>(&res[3]);
    }
}
```

**RMSNorm 不能用 Tensor Core** 的原因：只有逐元素操作 + reduce，没有矩阵乘法。

---

### 4.6 SwiGLU / FFN —— MoE 前馈网络

**面试场景：** "手写 SwiGLU 激活函数的 CUDA 实现"

#### 4.6.1 SwiGLU 定义

$$\text{SwiGLU}(x, W, V, W_2) = (\text{SiLU}(xW) \odot xV) \cdot W_2$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$，$\sigma$ 是 sigmoid 函数。

等价于：$(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)) \cdot \text{down\_proj}$

#### 4.6.2 CUDA Core 版本（gate_up 融合 + SiLU）

```cuda
// ============================================
// SwiGLU fused gate_up + SiLU（CUDA Core）
// 融合 gate_proj, up_proj 和 SiLU 激活
// ============================================

__global__ void swiglu_gate_up_fused(
    const half* __restrict__ input,        // [tokens, in_dim]
    const half* __restrict__ gate_weight,  // [in_dim, hidden_dim]
    const half* __restrict__ up_weight,    // [in_dim, hidden_dim]
    half* __restrict__ gate_output,        // [tokens, hidden_dim] 临时
    half* __restrict__ up_output,          // [tokens, hidden_dim] 临时
    int tokens, int in_dim, int hidden_dim)
{
    int token_idx = blockIdx.x;
    int h_idx = threadIdx.x;

    if (h_idx < hidden_dim) {
        // 计算 gate = SiLU(x @ gate_weight)
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            float x_val = __half2float(input[token_idx * in_dim + i]);
            gate_sum += x_val * __half2float(gate_weight[i * hidden_dim + h_idx]);
            up_sum += x_val * __half2float(up_weight[i * hidden_dim + h_idx]);
        }

        // SiLU(x) = x * sigmoid(x)
        float gate_silu = gate_sum * (1.0f / (1.0f + expf(-gate_sum)));

        gate_output[token_idx * hidden_dim + h_idx] = __float2half(gate_silu);
        up_output[token_idx * hidden_dim + h_idx] = __float2half(up_sum);
    }
}

// Step 2: element-wise multiply + down_proj（可复用 GEMM）
__global__ void swiglu_elementwise_mul(
    half* __restrict__ gate_output,
    const half* __restrict__ up_output,
    int tokens, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tokens * hidden_dim) {
        float g = __half2float(gate_output[idx]);
        float u = __half2float(up_output[idx]);
        gate_output[idx] = __float2half(g * u);
    }
}
```

#### 4.6.3 Tensor Core 加速方案

SwiGLU 的三个矩阵乘法都可以用 Tensor Core：

```
1. gate_proj: X @ W_gate^T   → WMMA GEMM   ← Tensor Core
2. up_proj:   X @ W_up^T     → WMMA GEMM   ← Tensor Core
3. SiLU(gate) ⊙ up           → CUDA Core   (非矩阵操作)
4. down_proj: fused @ W_down^T → WMMA GEMM   ← Tensor Core
```

**实际工程中 (vLLM/SGLang) 的做法：**
- gate_proj 和 up_proj 合并为一个 GEMM（权重拼接），一次 WMMA 调用
- down_proj 另一次 WMMA 调用
- 中间的元素级乘法用独立的 CUDA kernel

```cuda
// ============================================
// 合并 gate+up projection（Tensor Core 优化）
// 将 W_gate 和 W_up 按列拼接：[W_gate | W_up]
// 一次 GEMM 同时得到 gate 和 up 结果
// ============================================

__global__ void swiglu_merged_gateup_wmma(
    const half* __restrict__ input,            // [tokens, in_dim]
    const half* __restrict__ merged_weight,    // [in_dim, 2*hidden_dim]
    half* __restrict__ merged_output,          // [tokens, 2*hidden_dim]
    int tokens, int in_dim, int hidden_dim)
{
    // 与 4.1.2 的 WMMA GEMM 完全相同
    // 区别仅在于矩阵维度：N = 2 * hidden_dim
    // 输出 shape：[tokens, 2*hidden_dim]
    //   前 hidden_dim 列 = gate 结果
    //   后 hidden_dim 列 = up 结果
}

// 然后再用一个 kernel 做 SiLU + element-wise multiply
__global__ void swiglu_activate(
    half* __restrict__ merged_output,
    int tokens, int hidden_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = idx / hidden_dim;
    int h_idx = idx % hidden_dim;

    if (token_idx < tokens) {
        half gate = merged_output[token_idx * 2 * hidden_dim + h_idx];
        half up = merged_output[token_idx * 2 * hidden_dim + hidden_dim + h_idx];

        float gate_f = __half2float(gate);
        float up_f = __half2float(up);

        // SiLU
        float silu = gate_f * (1.0f / (1.0f + expf(-gate_f)));

        // 结果写回 gate 部分（覆盖，供后续 down_proj 使用）
        merged_output[token_idx * 2 * hidden_dim + h_idx] = __float2half(silu * up_f);
    }
}
```

---

### 4.7 TopK / Gating —— MoE 路由算子

**面试场景：** "写 CUDA 实现 MoE 的 TopK 路由机制"

#### 4.7.1 MoE TopK Gating 原理

```
输入: router_logits [tokens, num_experts]
输出: topk_weights [tokens, top_k]  (归一化后的权重)
      topk_indices [tokens, top_k]  (选择的专家索引)
```

#### 4.7.2 CUDA Core 版本

```cuda
// ============================================
// MoE TopK Gating（CUDA Core）
// 每个 block 处理一个 token
// ============================================

__global__ void moe_topk_gating(
    const float* __restrict__ router_logits,   // [tokens, num_experts]
    float* __restrict__ topk_weights,           // [tokens, top_k]
    int* __restrict__ topk_indices,             // [tokens, top_k]
    int tokens, int num_experts, int top_k)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    // 共享内存：存储 local top-k（每个 warp 维护自己的）
    __shared__ float s_vals[32][8];    // 假设最多 8 个
    __shared__ int s_idxs[32][8];

    // Step 1: 初始化
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    // 每个 warp 的 top-k 缓存
    float top_vals[8] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY,
                         -INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int top_idxs[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    // Step 2: 每个 warp 并行扫描 num_experts
    int experts_per_warp = (num_experts + num_warps - 1) / num_warps;
    int start_exp = warp_id * experts_per_warp;
    int end_exp = min(start_exp + experts_per_warp, num_experts);

    for (int e = start_exp; e < end_exp; e++) {
        float val = router_logits[token_idx * num_experts + e];

        // 插入排序（top-k 维护）
        for (int k = 0; k < top_k; k++) {
            if (val > top_vals[k]) {
                // 后移
                for (int j = top_k - 1; j > k; j--) {
                    top_vals[j] = top_vals[j - 1];
                    top_idxs[j] = top_idxs[j - 1];
                }
                top_vals[k] = val;
                top_idxs[k] = e;
                break;
            }
        }
    }

    // Step 3: Warp 间归约（取全局 top-k）
    // 写入共享内存
    for (int k = 0; k < top_k; k++) {
        s_vals[warp_id][k] = top_vals[k];
        s_idxs[warp_id][k] = top_idxs[k];
    }
    __syncthreads();

    // 只有 warp 0 做最终归约
    if (warp_id == 0) {
        float final_vals[8];
        int final_idxs[8];
        int count = 0;

        // 合并所有 warp 的 top-k
        for (int w = 0; w < num_warps; w++) {
            for (int k = 0; k < top_k; k++) {
                if (count < top_k * num_warps) {
                    final_vals[count] = s_vals[w][k];
                    final_idxs[count] = s_idxs[w][k];
                    count++;
                }
            }
        }

        // 排序取全局 top-k
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                if (final_vals[j] > final_vals[i]) {
                    float tmp_v = final_vals[i]; final_vals[i] = final_vals[j]; final_vals[j] = tmp_v;
                    int tmp_i = final_idxs[i]; final_idxs[i] = final_idxs[j]; final_idxs[j] = tmp_i;
                }
            }
        }

        // Softmax 归一化
        float max_val = final_vals[0];
        float sum_exp = 0.0f;
        for (int k = 0; k < top_k; k++) {
            sum_exp += expf(final_vals[k] - max_val);
        }

        // 写入输出
        for (int k = 0; k < top_k; k++) {
            topk_weights[token_idx * top_k + k] = expf(final_vals[k] - max_val) / sum_exp;
            topk_indices[token_idx * top_k + k] = final_idxs[k];
        }
    }
}
```

#### 4.7.3 优化版：使用 warp shuffle 的 bitonic sort

```cuda
// ============================================
// Warp-level TopK 使用 Bitonic Sort（更适合 GPU）
// 比上面逐元素插入排序更高效
// ============================================

__device__ void warp_topk(
    float* vals, int* idxs, int k, int num_items)
{
    // 每个线程处理 num_items / 32 个元素
    int lane_id = threadIdx.x % 32;

    // 局部扫描找 top-k
    float local_vals[8];
    int local_idxs[8];
    int local_count = 0;

    for (int i = lane_id; i < num_items; i += 32) {
        // 维护 top-k（略，同上）
    }

    // Bitonic sort across warp（所有 lane 排序）
    // 每轮比较交换，确保 warp 内顺序
    for (int stage = 1; stage <= 5; stage++) {
        for (int step = stage; step > 0; step--) {
            int partner = lane_id ^ (1 << (step - 1));
            // 交换比较
        }
    }

    // lane 0-3 持有 top-k
    if (lane_id < k) {
        vals[lane_id] = local_vals[lane_id];
        idxs[lane_id] = local_idxs[lane_id];
    }
}
```

**TopK 不能用 Tensor Core** 的原因：这是排序/选择操作，不是矩阵乘法。

---

## 5. 性能调优实战 Checklist

### 5.1 使用 Nsight Compute 分析

```bash
# 基础 profiling
ncu --set full -o profile_report ./your_binary

# 重点关注指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
     --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
     --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
     ./your_binary
```

### 5.2 逐项检查清单

| 检查项 | 工具/方法 | 阈值 |
|--------|----------|------|
| **Occupancy** | ncu: `achieved_occupancy` | > 50% |
| **SM 计算利用率** | ncu: `sm__throughput` | > 60% |
| **全局内存带宽利用率** | ncu: `dram__throughput` | 计算瓶颈时 < 30% |
| **共享内存 bank conflict** | ncu: `shared_load_transactions` vs `shared_loads` | 比值 > 1 说明有问题 |
| **Warp divergence** | ncu: `branch_efficiency` | > 90% |
| **寄存器溢出 (spilling)** | ncu: `l1tex__t_sectors_pipe_lsu_mem_global_op_st` | 任何非零都可能有问题 |
| **Tensor Core 利用率** | ncu: `sm__pipe_tensor_cycles_active` | 越高越好 |

### 5.3 常见性能瓶颈与解决

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 低 occupancy | 寄存器/共享内存使用过多 | 减少 block 中的线程数，或拆分 kernel |
| 低 SM 利用率 | 内存带宽瓶颈 | 增加算术强度（分块、数据复用） |
| 低内存带宽 | 不对齐/非连续访问 | float4 向量化，padding 消除 bank conflict |
| Tensor Core 利用率低 | K 维度不对齐 | pad K 到 16 的倍数 |
| warp stall (Long Scoreboard) | 全局内存延迟 | 增加 warp 数隐藏延迟，或使用 async copy |

### 5.4 性能模型（Roofline Model）

```
性能 = min(计算峰值, 内存带宽 × 算术强度)

                      ┌──────────────────┐
                      │   计算瓶颈区域     │
                      │    (Arith > AI_point)
   每秒操作数          │   ╱               │
   (FLOPS)           │  ╱                │
                     │ ╱   内存瓶颈区域    │
                     │╱                   │
                     └───────────────────
                          算术强度 (FLOP/Byte)

AMAT = L1_latency + L1_miss_rate × (L2_latency + L2_miss_rate × HBM_latency)

对于 FP16 GEMM on A100:
  - 计算峰值: 312 TFLOPS
  - HBM 带宽: 2 TB/s  
  - 临界算术强度: 312,000 / 2,000 = 156 FLOP/Byte
```

---

## 6. 面试高频问题与回答思路

### Q1: Tensor Core 和 CUDA Core 的区别？什么时候用哪个？

**回答框架：**
1. **物理层面：** Tensor Core 是 SM 内的专用硬件，执行单一操作（矩阵乘加），CUDA Core 是通用计算单元（FMA/IADD/...）
2. **吞吐对比：** Tensor Core FP16 吞吐是 CUDA Core FP32 的 16×（A100），但只适用于矩阵乘
3. **适用场景：** GEMM/Conv → Tensor Core；Element-wise/Reduction/Sort → CUDA Core
4. **混合使用：** Flash Attention = Tensor Core (Q×K) + CUDA Core (Softmax)

### Q2: WMMA vs PTX MMA 怎么选？

**回答框架：**
1. WMMA：高层 API，易用，跨架构兼容，损失 5-10% 性能
2. PTX MMA：底层控制，可精确控制寄存器分配和 shared memory 布局，CUTLASS 使用的方案
3. 选择：原型用 WMMA，生产用 PTX MMA（或直接用 CUTLASS）

### Q3: Bank Conflict 是什么？怎么解决？

**回答框架：**
1. 定义：Shared memory 有 32 个 bank，同一 bank 被多个线程访问时串行化
2. 检测：ncu 中 shared_load_transactions > shared_loads
3. 解决：padding（每行加 1 个元素，错开 bank）、swizzle（异或重排地址）
4. 代码：`__shared__ float smem[32][32 + 1];` // +1 = padding

### Q4: 为什么要用 FP16/BF16？数值稳定性如何处理？

**回答框架：**
1. FP16 范围有限（max ≈ 65504），BF16 范围与 FP32 一致但精度更低
2. 处理：loss scaling（前向用 FP32，反向梯度 scale up/down）
3. Tensor Core 专用：A100 支持 BF16 的 mma 指令
4. 实践：LLM 推理用 FP16/BF16 没问题（权重范围可控），训练需要 loss scaling

### Q5: Flash Attention 为什么不用存完整的 Attention Matrix？

**回答框架：**
1. 核心：online softmax 允许增量更新，不需要看到全部 S 值
2. 分块：Q 分块 B_r，K/V 分块 B_c，逐块计算 softmax
3. 维护：running max m 和 running sum l，每块更新时修正历史
4. 反向：重计算 S（tiling 后每块很小），不存储完整 S

### Q6: CUTLASS 的分层抽象是什么？

**回答框架：**
1. 五层：Tile → Warp Tile → MMA/WMMA → Epilogue → Kernel
2. Tile：全局内存 → shared memory（TiledCopy）
3. Warp Tile：shared memory → 寄存器
4. MMA/WMMA：寄存器内矩阵乘加（调 PTX mma）
5. Epilogue：激活函数、量化等后处理
6. 为什么分层：解耦数据搬运和计算，方便组合不同策略

---

## 7. 推荐学习路线

### 7.1 循序渐进

```
第 1 周：CUDA Core 基础
├── vector_add, matrix_transpose
├── shared memory tiling（GEMM 朴素版）
├── warp shuffle reduction
└── 练习：CUDA Core Softmax, LayerNorm

第 2 周：Tensor Core 入门
├── WMMA API 基础（4.1.2 的 GEMM）
├── FP16 数据类型（__half, half2）
├── Img2Col + WMMA Conv2D
└── 练习：WMMA GEMM + ReLU epilogue

第 3 周：Flash Attention 实战
├── Online Softmax 推导
├── Flash Attention forward（4.2.3）
├── Nsight Compute profiling
└── 练习：对比 CUDA Core vs Tensor Core 版本

第 4 周：进阶优化
├── PTX MMA 指令（4.1.3）
├── CUTLASS 源码阅读
├── SwiGLU/MoE 融合 kernel
└── 练习：手写一个 CUTLASS 风格的 GEMM
```

### 7.2 推荐资料

| 资源 | 说明 | 优先级 |
|------|------|--------|
| **CUDA C++ Programming Guide** | 官方文档，必读 | ⭐⭐⭐⭐⭐ |
| **CUDA C++ Best Practices Guide** | 优化指南 | ⭐⭐⭐⭐⭐ |
| **Parallel Thread Execution ISA** | PTX 指令参考 | ⭐⭐⭐⭐ |
| **CUTLASS 源码** | 模板库，最佳实践 | ⭐⭐⭐⭐⭐ |
| **Flash Attention 论文** | 算法原理 | ⭐⭐⭐⭐⭐ |
| **NVIDIA 官方博客 (Parallel Forall)** | 大量实战文章 | ⭐⭐⭐⭐ |
| **Programming Tensor Cores (NVIDIA GTC)** | 视频 + PPT | ⭐⭐⭐⭐ |
| **vLLM/SGLang 源码** | 工业级 LLM 推理 | ⭐⭐⭐⭐ |

### 7.3 实战项目建议

1. **基础：** 从零实现 FP16 GEMM，用 ncu 对比 WMMA vs PTX MMA vs cuBLAS
2. **进阶：** 实现简化版 Flash Attention forward，支持 causal mask
3. **高阶：** 为 LLM 推理实现 fused RMSNorm + SwiGLU kernel，benchmark vs vLLM
4. **专家：** 阅读并注释 CUTLASS GEMM 的 500+ 行核心代码

---

> **最后的话：** Tensor Core 编程的精髓不是记住 API，而是理解**数据流**——从全局内存到共享内存，从共享内存到寄存器，从寄存器到 MMA 指令，每一步的布局转换和同步开销才是性能的关键。建议先写 WMMA，跑通后看 ncu profile，再尝试用 PTX mma 优化。遇到问题先看 CUDA 文档和 CUTLASS 源码，它们是最好的老师。
