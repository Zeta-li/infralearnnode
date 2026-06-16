# TileLang 语法速查手册

> TileLang 是一种 Python 嵌入式 DSL，专为高性能 GPU/CPU 内核开发设计。本文档汇总所有核心语法，方便编码时快速查找。

---

## 目录

1. [程序结构](#1-程序结构)
2. [内存管理与缓冲区](#2-内存管理与缓冲区)
3. [计算原语](#3-计算原语)
4. [循环与控制流](#4-循环与控制流)
5. [数据类型](#5-数据类型)
6. [内建硬件原语](#6-内建硬件原语)
7. [布局标注与 Swizzle](#7-布局标注与-swizzle)
8. [Python 兼容性](#8-python-兼容性)
9. [完整示例](#9-完整示例)

---

## 1. 程序结构

### 1.1 `@tilelang.jit` 装饰器

标记一个 Python 函数为 TileLang 入口，处理 JIT 编译、缓存和后端选择。

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16"):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        # 内核逻辑
        ...
    return kernel
```

- **编译时参数**：形状、块大小、数据类型等在 `@tilelang.jit` 外层函数中定义
- **运行时参数**：张量/缓冲区在 `@T.prim_func` 内层函数中定义
- **懒编译**：首次调用时编译，或通过 `.compile()` 显式编译
- **内核缓存**：基于编译时参数自动缓存

### 1.2 `@T.prim_func` 装饰器

定义实际的内核实现，内部函数指定运行时参数及类型注解。

```python
@T.prim_func
def kernel_impl(
    A: T.Tensor((M, K), "float16"),  # 输入张量
    B: T.Tensor((K, N), "float16"),  # 输入张量
    C: T.Tensor((M, N), "float16"),  # 输出张量
):
    with T.Kernel(...) as (bx, by):
        ...
```

### 1.3 `T.Kernel` 上下文管理器

定义网格维度和线程块配置，是内核的入口点。

```python
# 一维网格 (如逐元素操作)
with T.Kernel(T.ceildiv(N, block), threads=256) as bx:
    for i in T.Parallel(block):
        C[bx * block + i] = A[bx * block + i] + B[bx * block + i]

# 二维网格 (如矩阵乘法)
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    ...

# 三维网格
with T.Kernel(gx, gy, gz, threads=128) as (bx, by, bz):
    ...
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `grid_x, grid_y, grid_z` | 网格维度，映射到 `blockIdx` |
| `threads` | 每个线程块的线程数，默认 128 |
| `cluster_dims` | 可选，Hopper+ 架构的 Cluster 维度 |

### 1.4 `T.macro` — 可复用代码块

类似 CUDA 的 `__device__` 函数，在编译时内联展开。

```python
@T.macro
def my_helper(buf, idx):
    buf[idx] = buf[idx] * 2.0
```

### 1.5 `T.const` — 编译时常量

```python
BLOCK_SIZE = T.const(128, "int32")
```

---

## 2. 内存管理与缓冲区

### 2.1 内存层次概览

```
全局内存 (global)     → 大容量(GB级)，高延迟(数百周期)，所有线程可访问
    ↓ T.copy() / TMA / cp.async
共享内存 (shared)     → 小容量(KB级)，低延迟(20-30周期)，线程块内共享
    ↓ ldmatrix / LDSM
Fragment (local.fragment) → 寄存器存储，布局感知，Tensor Core 操作数
    ↓
局部内存 (local)      → 线程私有，寄存器或 L1 缓存
```

### 2.2 缓冲区分配 API

| 函数 | 作用域 | 说明 | 示例 |
|------|--------|------|------|
| `T.alloc_shared(shape, dtype)` | `shared.dyn` | 共享内存，线程块内通信 | `A_shared = T.alloc_shared((BM, BK), "float16")` |
| `T.alloc_local(shape, dtype)` | `local` | 通用线程私有存储 | `buf = T.alloc_local((128,), "float32")` |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` | 寄存器存储，用于 Tensor Core | `C_local = T.alloc_fragment((BM, BN), "float")` |
| `T.alloc_var(dtype, init, scope)` | `local.var` | 单元素变量，可选初始值 | `val = T.alloc_var("float32", init=0.0)` |
| `T.alloc_tmem(shape, dtype)` | `shared.tmem` | Blackwell (SM100+) 专用张量内存 | `T.alloc_tmem((M, N), "float16")` |
| `T.alloc_barrier(arrive_count)` | `shared.barrier` | 硬件屏障，用于异步同步 | `T.alloc_barrier(128)` |
| `T.alloc_reducer(shape, dtype)` | — | 归约操作缓冲区 | `T.alloc_reducer((128,), "float32")` |

### 2.3 `T.Tensor` — 张量声明

在 `@T.prim_func` 参数中声明全局内存张量：

```python
A: T.Tensor((M, K), "float16")   # 二维张量
B: T.Tensor((K, N), "float16")
C: T.Tensor((M, N), "float16")
```

### 2.4 内存作用域标识符

| 作用域字符串 | 说明 |
|-------------|------|
| `"global"` | 全局内存，默认 |
| `"shared"` | 静态共享内存 |
| `"shared.dyn"` | 动态共享内存（`alloc_shared` 默认） |
| `"shared.tmem"` | Blackwell 张量内存 |
| `"shared.barrier"` | 硬件屏障内存 |
| `"local"` | 线程私有存储 |
| `"local.var"` | 单元素变量 |
| `"local.fragment"` | 布局感知的寄存器存储 |

---

## 3. 计算原语

### 3.1 `T.gemm` — 矩阵乘法

执行 C = A × B，自动选择硬件指令（WGMMA / TCGEN05 / MFMA）。

```python
T.gemm(A_shared, B_shared, C_local, trans_A=False, trans_B=False,
       policy=GemmWarpPolicy.kSquare)
```

| 参数 | 说明 |
|------|------|
| `A, B` | 输入矩阵（shared 或 fragment） |
| `C` | 累加矩阵（fragment） |
| `trans_A` | 是否转置 A |
| `trans_B` | 是否转置 B |
| `policy` | Warp 分配策略（kSquare 等） |

**指令自动选择：**
- Hopper (SM90) → `wgmma`
- Blackwell (SM100) → `tcgen05`
- AMD CDNA → `mfma`

### 3.2 `T.copy` — 数据搬运

在不同内存层次间传输数据，自动选择 SIMT 循环 / TMA / ldmatrix 等机制。

```python
# 全局内存 → 共享内存
T.copy(A[by * BM, ko * BK], A_shared)

# 共享内存 → 全局内存
T.copy(C_local, C[by * BM, bx * BN])
```

支持的搬运路径：
- `global` ↔ `shared`（TMA / cp.async / SIMT）
- `shared` ↔ `local.fragment`（ldmatrix / LDSM / STSM）

### 3.3 `T.reduce_*` — 归约操作

沿指定维度进行归约。

```python
T.reduce_sum(src, dst, axis=1)    # 求和
T.reduce_max(src, dst, axis=0)    # 最大值
T.reduce_min(src, dst, axis=0)    # 最小值
T.reduce_abssum(src, dst, axis=1) # 绝对值求和
T.reduce_absmax(src, dst, axis=1) # 绝对值最大
```

支持的归约类型：`sum`, `abssum`, `max`, `min`, `absmax`, `bitand`, `bitor`, `bitxor`

### 3.4 `T.fill` / `T.clear` — 填充操作

```python
T.fill(buffer, value)   # 用标量值填充缓冲区
T.clear(buffer)         # 清零缓冲区（T.fill 的特化）
```

### 3.5 `T.atomic_add` / `T.atomic_max` — 原子操作

```python
T.atomic_add(dst, src)  # 原子加法
T.atomic_max(dst, src)  # 原子最大值
```

支持 CUDA 内存序：`relaxed`, `release`, `acquire`, `acq_rel`

### 3.6 `T.im2col` — 卷积降维

```python
T.im2col(input, output, ...)  # 将图像转为列格式，用于卷积
```

### 3.7 `T.cumsum` — 累积求和

```python
T.cumsum(src, dst, axis=0)
```

### 3.8 计算原语与内存作用域对照

| 原语 | 允许的作用域 | 实现策略 |
|------|-------------|---------|
| `T.gemm` | `shared`, `shared.tmem`, `local.fragment` | Tensor Core (WGMMA/TCGEN05/MFMA) |
| `T.copy` | `global`, `shared`, `local.fragment` | SIMT Loop / TMA / Async Copy / LDSM |
| `T.fill` | `shared`, `local`, `local.fragment` | SIMT 并行循环 + 向量化 |
| `T.reduce_*` | `local.fragment` | Warp Shuffle / 共享内存 AllReduce |

---

## 4. 循环与控制流

### 4.1 循环类型总览

| Python API | 说明 | 主要用途 |
|-----------|------|---------|
| `T.Parallel(*extents)` | 并行循环，映射到 GPU 线程 | 逐元素操作，数据并行 |
| `T.Pipelined(extent, num_stages=N)` | 软件流水线循环 | 重叠数据搬运与计算 |
| `T.serial(start, stop, step)` | 串行循环 | 线程内顺序迭代 |
| `T.unroll(start, stop, step)` | 强制展开循环 | 常量范围循环，减少分支开销 |
| `T.vectorized(extent)` | 向量化循环 | SIMD 向量化访存与计算 |

### 4.2 `T.Parallel` — 并行循环

```python
# 单维并行
for i in T.Parallel(block_M):
    A_shared[i] = A[row + i]

# 多维并行
for i, j in T.Parallel(block_M, block_N):
    B_shared[i, j] = B[r + i, c + j]
```

**可选标注：**

| 标注 | 说明 |
|------|------|
| `coalesced_width` | 内存合并访问提示 |
| `parallel_loop_layout` | 附加 Fragment 布局 |
| `parallel_prefer_async` | 请求 cp.async 注入 |

**限制：** 访问 fragment/local 缓冲区的并行循环不能有符号化范围（必须为常量）。

### 4.3 `T.Pipelined` — 软件流水线

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)   # 阶段0: 数据搬运
    T.copy(B[ko * BK, bx * BN], B_shared)   # 阶段0: 数据搬运
    T.gemm(A_shared, B_shared, C_local)      # 阶段1: 计算
```

- `num_stages`：流水线级数（如 2=双缓冲，3=三缓冲）
- 编译器自动分配 `software_pipeline_stage` 和 `software_pipeline_order`
- 自动识别异步生产者（如 TMA copy）
- 自动多版本化缓冲区（如双缓冲）

### 4.4 `T.serial` — 串行循环

```python
# 基本用法
for i in T.serial(N):
    ...

# 带步长
for i in T.serial(0, N, 2):
    ...
```

### 4.5 `T.unroll` — 循环展开

```python
for i in T.unroll(16):
    ...
```

- 默认设置 `pragma_unroll_explicit=False`
- 访问 `local`/`warp` 作用域缓冲区时可能强制展开

### 4.6 `T.vectorized` — 向量化循环

```python
for i in T.vectorized(16):
    dst[i] = src[i]
```

- 编译器自动选择最优向量宽度（128/256 位）
- 支持混合精度类型转换（通过 `DecoupleTypeCast`）

### 4.7 条件语句

```python
# if/elif/else
if condition:
    ...
elif other_condition:
    ...
else:
    ...

# 三元表达式
x = a if cond else b

# 函数式条件
T.if_then_else(cond, true_val, false_val)
```

### 4.8 循环标注

| 标注 | 函数 | 说明 |
|------|------|------|
| `software_pipeline_stage` | 手动指定流水线阶段 | `T.Pipelined` 内使用 |
| `software_pipeline_order` | 手动指定发射顺序 | `T.Pipelined` 内使用 |
| `tl_pipelined_num_stages` | 流水线级数 | 流水线循环属性 |

---

## 5. 数据类型

### 5.1 标量类型

| 类别 | 类型 | Python 符号 | 位数 |
|------|------|-------------|------|
| **浮点** | Float16 | `T.float16` | 16 |
| | BFloat16 | `T.bfloat16` | 16 |
| | Float32 | `T.float32` | 32 |
| | Float64 | `T.float64` | 64 |
| **低精度** | FP8 E4M3 | `T.float8_e4m3fn` | 8 |
| | FP8 E5M2 | `T.float8_e5m2` | 8 |
| | FP8 E8M0 | `T.float8_e8m0fnu` | 8 |
| | FP4 E2M1 | `T.float4_e2m1fn` | 4 |
| **整数** | Int8 | `T.int8` | 8 |
| | Int16 | `T.int16` | 16 |
| | Int32 | `T.int32` | 32 |
| | Int64 | `T.int64` | 64 |
| | UInt8 | `T.uint8` | 8 |
| | UInt16 | `T.uint16` | 16 |
| | UInt32 | `T.uint32` | 32 |
| | UInt64 | `T.uint64` | 64 |
| **布尔** | Bool | `T.bool` | 1 |

### 5.2 向量类型

```python
T.float16x2(a, b)   # 打包的 float16x2
T.bfloat16x2(a, b)  # 打包的 bfloat16x2
```

### 5.3 类型转换

```python
# 标量转换
T.cast(value, "float16")

# 向量化转换（自动优化）
T.cast(buffer, dtype)  # 编译器自动选择最优向量化宽度
```

### 5.4 框架互操作

```python
# TileLang → PyTorch
dtype.as_torch()  # 返回对应的 torch.dtype

# NumPy → TileLang
# 内部通过 _NUMPY_DTYPE_TO_STR 映射
```

---

## 6. 内建硬件原语

### 6.1 TMA 操作 (Hopper SM90+)

| 原语 | 说明 |
|------|------|
| `create_tma_descriptor` | 创建 TMA 描述符 |
| `tma_load` | 异步从全局内存加载到共享内存 |
| `tma_load_multicast` | 多播异步加载到 Cluster |
| `tma_store` | 异步从共享内存存储到全局内存 |

### 6.2 全局内存加载/存储

| 原语 | 说明 |
|------|------|
| `ldg32` / `ldg64` / `ldg128` / `ldg256` | 指定位宽的全局内存加载 |
| `stg32` / `stg64` / `stg128` / `stg256` | 指定位宽的全局内存存储 |
| `__ldg` | 通过只读缓存加载 |

### 6.3 矩阵计算指令

| 原语 | 架构 | 说明 |
|------|------|------|
| `ptx_wgmma_ss` | SM90 | Hopper WGMMA (shared-shared) |
| `ptx_tcgen05_mma_ss` | SM100 | Blackwell TCGEN05 (shared-shared) |
| `ptx_tcgen05_mma_ts` | SM100 | Blackwell TCGEN05 (tmem-shared) |

### 6.4 Warp 级原语

```python
# Warp 投票
T.ballot(cond)         # Warp 级 ballot
T.all(cond)            # Warp 级 all
T.any(cond)            # Warp 级 any

# Warp 同步
T.warp_sync(mask)      # Warp 同步
```

### 6.5 同步与屏障

| 原语 | 说明 |
|------|------|
| `ptx_fence_barrier_init` | 初始化屏障 |
| `warpgroup_arrive` | 到达 Warpgroup 屏障 |
| `warpgroup_wait` | 等待 Warpgroup 屏障 |

### 6.6 快速数学函数

| 原语 | 说明 |
|------|------|
| `__exp` | 快速指数 |
| `__log` | 快速对数 |
| `__sin` | 快速正弦 |

### 6.7 打包向量运算

| 原语 | 说明 |
|------|------|
| `add2` | 打包向量加法 (float16x2 / bfloat16x2) |
| `fma2` | 打包融合乘加 |
| `abs2` | 打包绝对值 |

### 6.8 编译配置键

| 配置键 | 说明 |
|--------|------|
| `tl.disable_tma_lower` | 禁止自动将 T.copy 降级为 TMA |
| `tl.enable_fast_math` | 启用快速数学内建 |
| `tl.enable_lower_ldgstg` | 将全局访问降级为 ldg/stg |
| `tl.disable_wgmma` | 禁用 WGMMA 指令选择 |
| `tl.disable_thread_storage_sync` | 禁用自动屏障插入 |

---

## 7. 布局标注与 Swizzle

### 7.1 `T.use_swizzle` — 线程块 Swizzle

改善 L2 缓存局部性，用于线程块光栅化。

```python
T.use_swizzle(panel_size=10, enable=True)
# 或指定顺序
T.use_swizzle(panel_size=10, order="row")
```

### 7.2 `T.annotate_layout` — 布局标注

直接为缓冲区附加 Layout 或 Fragment，绕过自动布局推断。

```python
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

### 7.3 其他标注

```python
T.annotate_safe_value({buf: 0.0})          # 标注安全值（越界访问时使用）
T.annotate_l2_hit_ratio({buf: 1.0})        # 标注 L2 缓存命中率
T.annotate_restrict_buffers(buf1, buf2)     # 标注不重叠缓冲区
T.annotate_min_blocks_per_sm(n)             # 最小每 SM 线程块数
```

### 7.4 Swizzle 布局工具 (`tilelang.layout`)

| 函数 | 架构 | 说明 |
|------|------|------|
| `make_swizzled_layout` | 通用 | 可配置粒度的 XOR swizzle |
| `make_volta_swizzled_layout` | SM70 | Volta ldmatrix 模式 |
| `make_wgmma_swizzled_layout` | SM90 | Hopper WGMMA 要求 |
| `make_tcgen05mma_swizzled_layout` | SM100 | Blackwell TCGEN05 MMA |
| `make_full_bank_swizzled_layout` | 通用 | 128 字节 bank swizzle |
| `make_half_bank_swizzled_layout` | 通用 | 64 字节 bank swizzle |
| `make_quarter_bank_swizzled_layout` | 通用 | 32 字节 bank swizzle |
| `make_fully_replicated_layout_fragment` | 通用 | 每个线程持有完整副本 |

### 7.5 Layout 操作

```python
Layout.repeat(dim, factor)           # 沿维度重复布局
Layout.expand(leading_shape)         # 添加前导维度
Layout.reshape(shape)                # 改变逻辑形状
Fragment.replicate(factor)           # 跨线程复制 Fragment
```

---

## 8. Python 兼容性

### 8.1 控制流

| Python 特性 | 支持 | 说明 / 替代 |
|------------|------|------------|
| `for i in range(n)` | ✅ | 映射到 `T.serial(n)` |
| `for i in range(a, b, s)` | ✅ | 映射到 `T.serial(a, b, s)` |
| `for x in list` | ❌ | 使用索引循环 |
| `while condition` | ✅ | — |
| `if` / `elif` / `else` | ✅ | — |
| `x if cond else y` | ✅ | 三元表达式 |
| `break` / `continue` | ✅ | — |
| `enumerate()` / `zip()` | ❌ | — |

### 8.2 数据访问

| Python 特性 | 支持 | 说明 |
|------------|------|------|
| `a[i]` 索引 | ✅ | 支持多维：`a[i, j, k]` |
| `a[i:j]` 切片 | ✅ | 创建 BufferRegion |
| `a[-1]` 负索引 | ✅ | — |

### 8.3 赋值与运算

| Python 特性 | 支持 | 说明 |
|------------|------|------|
| `x = expr` | ✅ | — |
| `+`, `-`, `*`, `/`, `%` | ✅ | 映射到设备端运算 |
| `+=`, `-=`, `*=` 等 | ✅ | 增强赋值 |
| `a = b = c` | ❌ | 使用分开赋值 |

### 8.4 函数与类

- **不支持** Python 函数和类
- 使用 `@T.macro` 定义可复用代码块（编译时内联）

### 8.5 语句与内建函数

| Python 特性 | 支持 | 说明 |
|------------|------|------|
| `with` | ⚠️ | 仅支持 `T.Kernel`, `T.ws` |
| `assert` | ⚠️ | 使用 `T.device_assert` 或 `T.assert` |
| `print()` | ⚠️ | 使用 `T.print()` |
| `len()` | ❌ | 使用 `buffer.shape[dim]` |
| `type()` / `isinstance()` | ❌ | — |

---

## 9. 完整示例

### 9.1 矩阵乘法 (GEMM)

```python
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=32,
           dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 可选: 布局标注
            T.annotate_layout({
                A_shared: make_mma_swizzle_layout(A_shared),
                B_shared: make_mma_swizzle_layout(B_shared),
            })

            # 可选: Swizzle 光栅化
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### 9.2 向量加法 (Elementwise)

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def vector_add(N, block_size=256, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                C[idx] = A[idx] + B[idx]

    return main
```

### 9.3 Flash Attention (简化版)

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def flash_attention(B, H, M, N, block_M=64, block_N=64,
                    dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        Q: T.Tensor((B, H, M, 64), dtype),
        K: T.Tensor((B, H, N, 64), dtype),
        V: T.Tensor((B, H, N, 64), dtype),
        O: T.Tensor((B, H, M, 64), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), H, B, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, 64), dtype)
            K_shared = T.alloc_shared((block_N, 64), dtype)
            V_shared = T.alloc_shared((block_N, 64), dtype)
            S_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            O_local  = T.alloc_fragment((block_M, 64), accum_dtype)

            T.clear(O_local)
            m_i = T.alloc_var(accum_dtype, init=T.infinity(accum_dtype) * -1)
            l_i = T.alloc_var(accum_dtype, init=0.0)

            # 加载 Q tile
            T.copy(Q[bz, by, bx * block_M, 0], Q_shared)

            for ko in T.serial(T.ceildiv(N, block_N)):
                T.copy(K[bz, by, ko * block_N, 0], K_shared)
                T.gemm(Q_shared, K_shared, S_local, trans_B=True)

                # Softmax 在线更新
                # ... (省略详细 softmax 逻辑)

                T.copy(V[bz, by, ko * block_N, 0], V_shared)
                T.gemm(S_local, V_shared, O_local)

            T.copy(O_local, O[bz, by, bx * block_M, 0])

    return main
```

---

## 快速参考卡

```
# 导入
import tilelang
import tilelang.language as T

# 程序结构
@tilelang.jit → @T.prim_func → with T.Kernel(...) as (...):

# 内存分配
T.alloc_shared(shape, dtype)      # 共享内存
T.alloc_fragment(shape, dtype)    # 寄存器 Fragment
T.alloc_local(shape, dtype)       # 线程私有
T.alloc_var(dtype, init)          # 标量变量

# 计算
T.gemm(A, B, C)                   # 矩阵乘法
T.copy(src, dst)                   # 数据搬运
T.reduce_sum/max/min(src, dst)    # 归约
T.fill(buf, val) / T.clear(buf)   # 填充/清零
T.atomic_add/dst, src)            # 原子操作

# 循环
T.Parallel(M, N)                   # 并行
T.Pipelined(K, num_stages=3)      # 流水线
T.serial(N)                        # 串行
T.unroll(N)                        # 展开
T.vectorized(N)                    # 向量化

# 标注
T.annotate_layout({...})           # 布局标注
T.use_swizzle(panel_size=N)       # Swizzle
T.annotate_safe_value({...})       # 安全值

# 工具
T.ceildiv(a, b)                    # 向上取整除法
T.cast(val, dtype)                 # 类型转换
T.if_then_else(cond, a, b)        # 条件选择
T.infinity(dtype)                  # 无穷大
```

---

> 参考来源：[TileLang GitHub](https://github.com/tile-ai/tilelang) | [TileLang 官方文档](https://tilelang.tile-ai.cn/) | [DeepWiki TileLang](https://deepwiki.com/tile-ai/tilelang)
