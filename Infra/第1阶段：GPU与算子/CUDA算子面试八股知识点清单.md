# CUDA 算子面试八股知识点清单

---

## 模块一：GPU 硬件架构基础

### 1. SM（Streaming Multiprocessor）

- **定义**：SM 是 GPU 的核心计算单元，类似 CPU 的核心但面向吞吐量设计。每个 SM 包含多组 CUDA 核心（SP）、共享内存、L1 缓存、warp 调度器、寄存器文件、特殊功能单元（SFU）和加载/存储单元（LD/ST）。A100 每个SM有64个FP32 CUDA核心，H100 每个SM有128个FP32核心。
- **追问**：为什么 SM 要设计成这种多核心+共享内存的结构？→ 面向 SIMT 执行模型，用大量核心掩盖延迟，共享内存提供低延迟数据共享。
- **对比**：CPU 核心强调单线程延迟，有深度流水线和分支预测；SM 强调吞吐量，靠大量轻量核心并发执行。

### 2. SP（Streaming Processor / CUDA Core）

- **定义**：SP 是 SM 内最小的计算单元，每个 SP 可执行一个线程的浮点/整数运算。Kepler 开始一个 SP 对应一个 FP32 单元；Volta 之后引入了 FP16/FP32/INT32 混合精度单元。
- **追问**：Volta 之后为什么 FP32 和 INT32 可以并发执行？→ 因为它们使用不同的执行端口，可以同时发射。
- **代码**：`// 一个 SM 的理论峰值 FLOPS = SP数量 × 频率 × 2(乘加) × 每时钟发射次数`

### 3. Warp

- **定义**：Warp 是 SM 中线程调度的基本单位，由 32 个连续线程组成。同一 warp 内的线程以锁步（lockstep）方式执行相同指令（SIMT），但各自有独立的寄存器和执行路径。
- **追问**：为什么 warp 大小是 32？→ 硬件设计权衡：32 足够掩盖流水线延迟，又不至于分支代价过大。AMD CDNA 的 wavefront 是 64。
- **对比**：OpenCL 中对应概念为 wavefront/wave，AMD 为 64，Intel GPU 为 16/32。

### 4. GPU 内存层次

- **定义**：从快到慢依次为：寄存器 → 共享内存(L1) → L2 缓存 → 全局内存(GDDR/HBM)。此外还有常量内存（通过常量缓存）、纹理内存（通过纹理缓存）。
- **追问**：各层级延迟量级？→ 寄存器 ~1 周期，共享内存 ~20-30 周期，L2 ~200 周期，全局内存 ~400-800 周期（具体因架构而异）。
- **公式**：`有效带宽(GB/s) = 数据量(GB) / 传输时间(s)`

### 5. L2 Cache

- **定义**：所有 SM 共享的片上缓存，位于全局内存和 SM 之间，缓存全局内存访问。A100 L2 大小为 40MB，H100 为 50MB。
- **追问**：如何利用 L2 缓存提高性能？→ 数据局部性、`cudaFuncSetAttribute` 设置 L2 持久化策略（`cudaAccessPolicyWindow`）。
- **代码**：
  ```c
  cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  ```

### 6. Tensor Core

- **定义**：Volta 架构引入的矩阵乘加加速单元，可在单个时钟周期完成 `m×n×k` 的 MMA（Matrix Multiply-Accumulate）操作。A100 支持 MMA 16×8×16（FP16），H100 支持 MMA 16×8×32（FP16）。
- **追问**：Tensor Core 与 CUDA Core 的区别？→ Tensor Core 专用矩阵运算，效率远高于 CUDA Core 逐元素乘加；CUDA Core 是标量运算单元。
- **代码**：
  ```c
  // WMMA API (Volta+)
  #include <mma.h>
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
  nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, 16);
  ```

### 7. GPU 架构演进关键节点

- **定义**：
  - **Kepler (K80)**：引入动态并行、Hyper-Q
  - **Maxwell (GTX 900)**：统一共享内存/L1，能效提升
  - **Pascal (P100)**：HBM2 首次引入，NVLink
  - **Volta (V100)**：Tensor Core，独立线程调度
  - **Turing (T4)**：混合精度 Tensor Core，RT Core
  - **Ampere (A100)**：3代 Tensor Core，稀疏化，结构化稀疏
  - **Hopper (H100)**：4代 Tensor Core，TMA，分布式共享内存，DPX
- **追问**：Volta 的独立线程调度（Independent Thread Scheduling）带来了什么？→ 同一 warp 内不同线程可真正独立调度，减少了分支发散的性能损失，但增加了同步复杂度。

---

## 模块二：CUDA 编程模型

### 1. Kernel 启动语法

- **定义**：`kernel<<<grid, block, shared_mem, stream>>>(args)`，其中 grid 为网格维度，block 为块维度，shared_mem 为动态共享内存大小，stream 为执行流。
- **追问**：grid 和 block 的维度限制？→ block 维度最大 1024 线程（x×y×z ≤ 1024），grid 维度 x/y 最大 2^31-1，z 最大 65535（具体看 compute capability）。
- **代码**：
  ```c
  // 一维：256 threads/block, N/256 blocks
  kernel<<<(N + 255) / 256, 256>>>(d_data, N);
  // 二维：16x16 block
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16);
  kernel<<<grid, block>>>(d_data, W, H);
  ```

### 2. 内置变量

- **定义**：
  - `threadIdx.{x,y,z}`：当前线程在 block 内的索引
  - `blockIdx.{x,y,z}`：当前 block 在 grid 内的索引
  - `blockDim.{x,y,z}`：block 的维度大小
  - `gridDim.{x,y,z}`：grid 的维度大小
- **追问**：如何计算全局一维索引？→ `int idx = blockIdx.x * blockDim.x + threadIdx.x`
- **代码**：
  ```c
  // 2D 全局索引
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = row * width + col;
  ```

### 3. 线程层次与限制

- **定义**：Thread → Warp(32) → Block → Grid。Block 内线程可同步（`__syncthreads()`）并共享共享内存；不同 Block 之间无法直接同步。
- **追问**：为什么跨 Block 同步困难？→ Block 可在任意 SM 上调度，执行顺序不确定。跨 Block 同步只能通过 kernel 结束或原子操作+自旋等待（不推荐）实现。
- **限制**：每个 Block 最大 1024 线程，最大维度 (1024,1024,64)；每个 SM 同时驻留的 Block 数受寄存器和共享内存约束。

### 4. CUDA 核函数限定符

- **定义**：
  - `__global__`：在 GPU 上执行，从 CPU 调用（或通过动态并行从 GPU 调用），返回 void
  - `__device__`：在 GPU 上执行，只能从 GPU 调用
  - `__host__`：在 CPU 上执行，只能从 CPU 调用（默认）
  - `__host__ __device__`：同时编译 CPU 和 GPU 版本
- **追问**：`__global__` 函数为什么必须返回 void？→ kernel 是异步启动的，没有机制将返回值传回 host。
- **代码**：
  ```c
  __global__ void vecAdd(float *a, float *b, float *c, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) c[i] = a[i] + b[i];
  }
  ```

### 5. Compute Capability

- **定义**：描述 GPU 硬件功能的版本号，如 7.0（V100）、8.0（A100）、9.0（H100）。主版本号对应架构代际，次版本号区分同代不同型号。
- **追问**：编译时如何指定？→ `nvcc -arch=sm_80` 或 `-gencode=arch=compute_80,code=sm_80`
- **对比**：Compute Capability 决定了可用的硬件特性（如 warp reduce 指令从 Kepler 开始，Cooperative Groups 从 Pascal+）。

---

## 模块三：内存模型与优化

### 1. 全局内存（Global Memory）

- **定义**：所有 SM 共享的大容量显存（A100: 40/80GB HBM2e），所有线程均可读写，延迟最高（~400-800 周期），带宽最大但需要合并访问才能充分利用。
- **追问**：HBM2e 和 GDDR6 的区别？→ HBM 使用硅通孔（TSV）实现超宽位宽（3072-5120 bit），单 pin 带宽较低但总带宽极高；GDDR6 位宽较窄（384 bit）但单 pin 频率更高。
- **代码**：
  ```c
  float *d_data;
  cudaMalloc(&d_data, N * sizeof(float));  // 分配全局内存
  cudaFree(d_data);                         // 释放
  ```

### 2. 合并访问（Coalesced Access）

- **定义**：当同一 warp 内线程访问的地址连续且对齐时，硬件将这些访问合并为尽可能少的事务（transaction），极大提高带宽利用率。理想情况：warp 内线程 i 访问地址 `base + i * sizeof(T)`。
- **追问**：不合并访问的代价？→ 一次 128 字节的 cache line 可能只为 1 个线程服务，带宽利用率降至 1/32。L1 cache line 为 128B，L2 segment 为 32B。
- **代码**：
  ```c
  // ✅ 合并访问：连续线程访问连续地址
  float val = data[threadIdx.x + blockIdx.x * blockDim.x];
  // ❌ 非合并访问：转置访问模式
  float val = data[threadIdx.y * width + threadIdx.x]; // 当 width 不是 32 的倍数
  ```

### 3. 共享内存（Shared Memory）

- **定义**：SM 片上高速存储，同一 Block 内线程共享，可编程管理。延迟约 20-30 周期，带宽约 19TB/s（A100），远高于全局内存。通过 `__shared__` 声明。
- **追问**：共享内存和 L1 Cache 的关系？→ Kepler 开始共享内存和 L1 共享同一物理存储（64KB/128KB 可配分），Ampere 开始默认 128KB 可按比例划分。
- **代码**：
  ```c
  __global__ void kernel(...) {
      __shared__ float tile[256];          // 静态共享内存
      extern __shared__ float dyn_tile[];  // 动态共享内存
      tile[threadIdx.x] = global_data[idx];
      __syncthreads();
  }
  ```

### 4. 存储体冲突（Bank Conflict）

- **定义**：共享内存被划分为 32 个存储体（bank），每个 bank 宽 4 字节。当同一 warp 内两个及以上线程同时访问同一 bank 的不同地址时，发生冲突，访问被序列化（广播除外）。n-way 冲突将访问时间增加 n 倍。
- **追问**：如何解决？→ Padding（填充使地址错开）、改变访问模式、使用 `__ldg` 或 shuffle 指令。同一地址的广播读取不冲突。
- **代码**：
  ```c
  // ❌ 2-way bank conflict：相邻线程跨 stride=32 访问
  float val = shared[threadIdx.x * 32]; // 每32个float属于同一bank
  // ✅ Padding 解决
  __shared__ float tile[32][33]; // 33 列，避免 bank 冲突
  ```

### 5. 寄存器（Register）

- **定义**：每个线程私有的最快存储，延迟 ~1 周期。每个 SM 有固定数量的寄存器（A100: 65536个），所有驻留线程平分。寄存器溢出（spill）会降低性能。
- **追问**：寄存器分配如何影响占用率？→ 每线程寄存器越多 → 单 SM 驻留线程越少 → 占用率下降 → 延迟掩盖能力减弱。
- **代码**：
  ```c
  // 编译时限制每线程最大寄存器数
  __global__ void __launch_bounds__(256, 2) kernel(...) { ... }
  // 或启动时
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxRegistersPerBlock, 32);
  ```

### 6. 常量内存（Constant Memory）

- **定义**：大小 64KB，只读，通过常量缓存访问，对所有线程可见。当同一 warp 内所有线程读取相同地址时，一次广播即可，效率极高；若线程读取不同地址，则串行化。
- **追问**：常量内存的适用场景？→ 所有线程读取同一常量参数（如卷积核权重、查找表）。不适合线程读取不同地址的场景。
- **代码**：
  ```c
  __constant__ float const_data[256]; // 声明
  cudaMemcpyToSymbol(const_data, h_data, size); // 从 host 拷贝
  ```

### 7. 纹理内存（Texture Memory）

- **定义**：通过纹理缓存访问全局内存，对 2D 空间局部性有优化（缓存相邻像素），支持硬件插值（线性插值）、边界处理（clamp/wrap）和类型转换。
- **追问**：纹理内存 vs 共享内存？→ 纹理内存是只读缓存的，不需要手动管理；共享内存可读写但需手动管理同步。
- **代码**：
  ```c
  texture<float, 2, cudaReadModeElementType> tex;
  cudaBindTexture2D(...);
  float val = tex2D(tex, x, y);
  ```

### 8. Local Memory

- **定义**：逻辑上每线程私有，物理上位于全局内存（寄存器溢出时使用）。访问延迟与全局内存相同，但可通过 L1/L2 缓存。
- **追问**：什么情况下编译器会将变量放到 local memory？→ 寄存器不足、数组下标为变量（编译器无法确定偏移）、大型结构体。
- **优化**：检查 `--ptxas-options=-v` 输出的 `ptxas info : Used xxx registers, xxx bytes stack frame` 判断是否溢出。

### 9. 统一内存（Unified Memory / Managed Memory）

- **定义**：`cudaMallocManaged` 分配的内存在 CPU 和 GPU 间共享，由驱动自动按页迁移。简化编程，但可能引入页面迁移开销。
- **追问**：UM 的页面迁移机制？→ 按需迁移（demand paging），首次访问触发缺页中断和 PCIe 传输。Pascal+ 支持硬件页面迁移和预取。
- **代码**：
  ```c
  float *data;
  cudaMallocManaged(&data, N * sizeof(float));
  cudaMemPrefetchAsync(data, N * sizeof(float), deviceId, stream); // 预取
  ```

---

## 模块四：线程调度与并行模式

### 1. Warp 分支发散（Warp Divergence）

- **定义**：同一 warp 内线程走不同分支时，硬件串行执行各分支路径，不活跃线程被遮罩（mask）。例如 if-else 导致约 2 倍执行时间。
- **追问**：Volta+ 的独立线程调度如何改善？→ 允许同一 warp 内线程在不同分支间切换，但总执行时间仍为各路径之和。不过其他 warp 可以利用空闲的执行单元。
- **代码**：
  ```c
  // ❌ Warp divergence
  if (threadIdx.x < 16) { path_a(); } else { path_b(); }
  // ✅ 避免分歧：重排数据使同一 warp 走同一分支
  ```

### 2. 占用率（Occupancy）

- **定义**：单个 SM 上活跃 warp 数与最大支持 warp 数的比值。受每个线程的寄存器数量、每个 Block 的共享内存量、Block 大小等因素限制。
- **追问**：高占用率一定好吗？→ 不一定。高占用率有助于掩盖延迟，但可能迫使每线程寄存器减少，导致溢出。需要在占用率和寄存器使用量之间平衡。
- **公式**：
  ```
  Occupancy = active_warps / max_warps_per_SM
  限制因素：max(寄存器限制, 共享内存限制, Block数限制)
  // A100: max_warps_per_SM = 64
  ```

### 3. 同步机制

- **定义**：
  - `__syncthreads()`：Block 内所有线程同步，保证共享内存可见性。必须被 Block 内所有线程执行，否则死锁。
  - `__syncwarp(mask)`：warp 内指定线程同步（Volta+ 推荐）。
  - Cooperative Groups：灵活的同步抽象，支持跨 Block 同步（需 `cudaLaunchCooperativeKernel`）。
- **追问**：`__syncthreads()` 放在条件分支内会怎样？→ 如果不是所有线程都走同一分支，会死锁。
- **代码**：
  ```c
  __shared__ float tile[256];
  tile[threadIdx.x] = data[idx];
  __syncthreads(); // 确保所有线程写入完成
  float val = tile[255 - threadIdx.x]; // 安全读取
  ```

### 4. 原子操作（Atomic Operations）

- **定义**：对全局或共享内存的不可分割的读-改-写操作。支持 `atomicAdd`, `atomicSub`, `atomicExch`, `atomicCAS`, `atomicMin`, `atomicMax` 等。
- **追问**：原子操作的性能问题？→ 多个线程竞争同一地址会导致序列化，严重影响性能。全局内存原子延迟高，共享内存原子好一些但仍有竞争。
- **代码**：
  ```c
  // 全局原子加
  atomicAdd(&result, value);
  // CAS 实现自定义原子
  int old, assumed;
  do {
      assumed = old = result;
  } while (atomicCAS(&result, assumed, assumed + value) != assumed);
  ```

### 5. Warp-level 原语（Warp Shuffle）

- **定义**：允许同一 warp 内线程直接交换寄存器数据，无需通过共享内存，延迟极低。包括 `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`。
- **追问**：shuffle 和共享内存的区别？→ shuffle 无需额外存储，无 bank conflict，但仅限 warp 内。共享内存支持 block 内，但需同步且有 bank conflict。
- **代码**：
  ```c
  // Warp reduce sum
  float val = data[threadIdx.x];
  for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }
  // 最终 val 只在 threadIdx.x==0 有效
  ```

### 6. 常见并行模式

- **Map**：一一映射，最简单。每个线程处理一个元素。
- **Reduce**：树形归约，需要同步。O(logN) 步。
- **Scan/Prefix Sum**：Blelloch 或 Kogge-Stone 算法，O(logN) 步。
- **Histogram**：原子操作或共享内存局部统计再合并。
- **Transpose**：利用共享内存避免非合并访问。
- **Tiled**：分块计算，经典如矩阵乘法。

---

## 模块五：性能分析与调优

### 1. CUDA Profiling 工具

- **定义**：
  - **Nsight Compute (ncu)**：单 kernel 级深度分析，展示 SM 利用率、内存带宽、指令吞吐、占用率等。
  - **Nsight Systems (nsys)**：系统级时间线分析，展示 kernel 执行、内存传输、API 调用的时序关系。
- **追问**：如何用 ncu 分析 kernel？→ `ncu --set full -o report ./app`，关注 SM Busy、Memory Throughput、Occupancy、Stall Reason。
- **代码**：
  ```bash
  ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum ./app
  nsys profile -o report ./app
  ```

### 2. Roofline 模型

- **定义**：以算术强度（FLOP/Byte）为横轴、性能（FLOPS）为纵轴的模型。屋顶线由峰值计算能力和峰值带宽决定，分为计算受限区和带宽受限区。
- **追问**：如何判断 kernel 受限于什么？→ 算术强度低 → 带宽受限（如 element-wise 操作）；算术强度高 → 计算受限（如大矩阵乘法）。
- **公式**：
  ```
  算术强度 AI = FLOPs / Bytes_accessed
  峰值性能 = min(峰值FLOPS, 带宽 × AI)
  ```

### 3. 指令级并行（ILP）

- **定义**：通过让每个线程执行多条独立指令，利用流水线并行性掩盖延迟。例如每个线程处理 4 个元素，循环展开。
- **追问**：ILP 与 TLP（线程级并行）的关系？→ ILP 增加单线程工作量，TLP 增加并发线程数。两者都可掩盖延迟，但 ILP 增加寄存器压力。
- **代码**：
  ```c
  // 循环展开，增加 ILP
  float v0 = data[i], v1 = data[i+1], v2 = data[i+2], v3 = data[i+3];
  v0 = v0 * alpha; v1 = v1 * alpha; v2 = v2 * alpha; v3 = v3 * alpha;
  out[i] = v0; out[i+1] = v1; out[i+2] = v2; out[i+3] = v3;
  ```

### 4. 带宽优化策略

- **合并访问**：确保 warp 内连续线程访问连续地址。
- **对齐**：起始地址 128B 对齐，`cudaMalloc` 自动对齐。
- **向量化读写**：使用 `float4`/`int4` 等宽类型，一次传输 16 字节。
- **减少冗余传输**：能就地计算就不搬运。
- **Pinned Memory**：使用 `cudaMallocHost` 分配页锁定内存，提高 DMA 传输效率。
- **代码**：
  ```c
  // 向量化加载
  float4 val = reinterpret_cast<float4*>(data)[idx]; // 一次读4个float
  ```

### 5. 延迟隐藏

- **定义**：GPU 不依赖复杂预测/大缓存隐藏延迟，而是依赖大量并发线程。当某些 warp 等待内存时，调度器切换到其他就绪 warp。
- **追问**：需要多少 warp 才能隐藏延迟？→ 经验公式：`所需warp数 ≈ 延迟周期 / 指令吞吐周期`。实际需考虑依赖链长度。
- **关键**：提高占用率 → 更多活跃 warp → 更好延迟隐藏。

### 6. 流水线化（Software Pipelining）

- **定义**：将数据加载和计算重叠，在处理当前块数据的同时预取下一块数据。使用双缓冲（double buffering）技术。
- **代码**：
  ```c
  __shared__ float tile[2][TILE_SIZE]; // 双缓冲
  // Iteration 0: load tile[0]
  for (int t = 0; t < T; t++) {
      // 计算当前块 tile[t%2]
      // 异步加载下一块 tile[(t+1)%2]
      cp_async(&tile[(t+1)%2], &global[...]); // Hopper TMA
      compute(tile[t%2]);
      cp_async_commit();
      cp_async_wait_group<0>();
  }
  ```

### 7. 常见性能瓶颈指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| SM Occupancy | 活跃 warp 占比 | >50% |
| SM Busy | SM 计算单元利用率 | 越高越好 |
| Memory Throughput | 实际内存带宽/峰值 | 越高越好 |
| Warp Execution Efficiency | 非发散活跃线程占比 | >80% |
| L2 Hit Rate | L2 缓存命中率 | 越高越好 |

---

## 模块六：常见算子实现

### 1. Reduce（归约求和）

- **核心思路**：树形归约，每轮线程数减半。需处理 bank conflict 和线程同步。
- **追问**：如何优化 reduce？→ (1) 解决 bank conflict（加 padding 或交错寻址）(2) 减少空闲线程 (3) 最后一个 warp 不需要 `__syncthreads()`（warp 内锁步）(4) 循环展开最后一个 warp。
- **代码**：
  ```c
  __global__ void reduce(float *in, float *out, int n) {
      __shared__ float sdata[256];
      int tid = threadIdx.x;
      int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
      sdata[tid] = (i < n ? in[i] : 0) + (i + blockDim.x < n ? in[i + blockDim.x] : 0);
      __syncthreads();
      for (int s = blockDim.x / 2; s > 32; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
      }
      // Last warp: no sync needed
      if (tid < 32) {
          volatile float *vs = sdata;
          vs[tid] += vs[tid + 32]; vs[tid] += vs[tid + 16];
          vs[tid] += vs[tid + 8];  vs[tid] += vs[tid + 4];
          vs[tid] += vs[tid + 2];  vs[tid] += vs[tid + 1];
      }
      if (tid == 0) out[blockIdx.x] = sdata[0];
  }
  ```

### 2. SGEMM（单精度矩阵乘法）

- **核心思路**：分块（Tiling）+ 共享内存 + 寄存器累加 + 向量化加载。经典算法：C = A×B，将矩阵分为 tile，每次从全局内存加载一块到共享内存，在寄存器中累加部分和。
- **追问**：如何进一步优化？→ (1) 调整 tile 大小（如 128×128 block, 32×32 warp tile, 16×16 MMA tile）(2) 双缓冲预取 (3) 使用 Tensor Core WMMA/cuBLASLt (4) 避免共享内存 bank conflict（加 padding）(5) 计算与访存重叠。
- **代码**：
  ```c
  __global__ void sgemm(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
      __shared__ float As[TILE][TILE+1]; // +1 padding 避免 bank conflict
      __shared__ float Bs[TILE][TILE+1];
      float accum[TILE_X][TILE_Y] = {0};
      int tx = threadIdx.x, ty = threadIdx.y;
      int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
      for (int k = 0; k < K; k += TILE) {
          As[ty][tx] = A[row * K + k + tx];
          Bs[ty][tx] = B[(k + ty) * N + col];
          __syncthreads();
          for (int i = 0; i < TILE; i++)
              for (int j = 0; j < TILE_X; j++)
                  for (int l = 0; l < TILE_Y; l++)
                      accum[j][l] += As[ty * TILE_X + j][i] * Bs[i][tx * TILE_Y + l];
          __syncthreads();
      }
      // 写回结果
  }
  ```

### 3. Softmax

- **核心思路**：三遍扫描或两遍扫描：(1) 求 max (2) 求 exp(x-max) 的 sum (3) 归一化。Online softmax 可两遍完成。
- **追问**：数值稳定性？→ 必须减去最大值防止 exp 溢出。FlashAttention 的 online softmax 技巧。
- **代码**：
  ```c
  __global__ void softmax(float *x, float *out, int n) {
      __shared__ float s_max, s_sum;
      int tid = threadIdx.x;
      int i = blockIdx.x * n; // 每个block处理一行
      // Step 1: max
      float local_max = -FLT_MAX;
      for (int j = tid; j < n; j += blockDim.x)
          local_max = fmaxf(local_max, x[i + j]);
      // warp reduce max...
      if (tid == 0) s_max = local_max;
      __syncthreads();
      // Step 2: sum(exp)
      float local_sum = 0;
      for (int j = tid; j < n; j += blockDim.x)
          local_sum += expf(x[i + j] - s_max);
      // warp reduce sum...
      if (tid == 0) s_sum = local_sum;
      __syncthreads();
      // Step 3: normalize
      for (int j = tid; j < n; j += blockDim.x)
          out[i + j] = expf(x[i + j] - s_max) / s_sum;
  }
  ```

### 4. LayerNorm

- **核心思路**：对每个样本沿特征维度求均值和方差，然后归一化。与 Softmax 类似但需计算方差（两遍或一遍 Welford 算法）。
- **追问**：Welford 在线算法 vs 两遍算法？→ Welford 一遍扫描数值更稳定，两遍算法简单但需两次遍历。
- **公式**：
  ```
  μ = (1/D) Σ x_i
  σ² = (1/D) Σ (x_i - μ)²
  y_i = γ_i * (x_i - μ) / √(σ² + ε) + β_i
  ```
- **代码**：
  ```c
  // Welford online
  float mean = 0, m2 = 0;
  for (int j = tid; j < D; j += blockDim.x) {
      float delta = x[j] - mean;
      mean += delta / (j + 1);
      float delta2 = x[j] - mean;
      m2 += delta * delta2;
  }
  // warp reduce mean and m2...
  float var = m2 / D;
  float inv_std = rsqrtf(var + eps);
  ```

### 5. Flash Attention

- **核心思路**：分块计算注意力，将 Q、K、V 分成块（tile），逐块计算 QK^T 和 attention 输出，避免实例化完整 N×N 注意力矩阵。使用 online softmax 技巧逐步修正累加结果。
- **追问**：为什么 Flash Attention 能省显存？→ 不需要存储 N×N 的 attention 矩阵，只存储 O(N×d) 的输出和统计量。内存复杂度从 O(N²) 降到 O(N)。
- **核心公式（Online Softmax 修正）**：
  ```
  新 m' = max(m_old, m_new)
  修正因子 l' = l_old * exp(m_old - m') + l_new * exp(m_new - m')
  修正 O = O * (l_old * exp(m_old - m') / l') + (new_attn @ V) * (l_new * exp(m_new - m') / l')
  ```

### 6. Embedding / Gather

- **核心思路**：根据索引从权重表中查找行向量。本质是随机读取，难以合并访存。
- **追问**：如何优化？→ (1) 缓存友好：按索引排序 (2) 请求合并：batch 内相同索引的线程合并读取 (3) L2 持久化缓存热门 embedding。
- **代码**：
  ```c
  __global__ void embedding(float *weight, int *indices, float *out, int D) {
      int idx = blockIdx.x;
      int tid = threadIdx.x;
      float *row = weight + indices[idx] * D;
      if (tid < D) out[idx * D + tid] = row[tid]; // 同一 block 线程连续读取同一行
  }
  ```

### 7. Reduce-Scan（Prefix Sum）

- **核心思路**：Kogge-Stone（work-inefficient 但并行度高）或 Blelloch（work-efficient 两阶段）。常见实现用共享内存 + warp shuffle。
- **代码**（Blelloch Up-Sweep + Down-Sweep）：
  ```c
  // Up-sweep (reduce phase)
  for (int d = 1; d < n; d *= 2) {
      if ((tid + 1) % (2*d) == 0) sdata[tid] += sdata[tid - d];
      __syncthreads();
  }
  sdata[n-1] = 0; __syncthreads();
  // Down-sweep
  for (int d = n/2; d >= 1; d /= 2) {
      if ((tid + 1) % (2*d) == 0) {
          float tmp = sdata[tid - d];
          sdata[tid - d] = sdata[tid];
          sdata[tid] += tmp;
      }
      __syncthreads();
  }
  ```

---

## 模块七：CUDA 高级特性

### 1. CUDA Stream（流）

- **定义**：Stream 是 GPU 上的命令队列，同一流内操作串行执行，不同流间可并行。用于实现 kernel 执行与数据传输的重叠。
- **追问**：默认流（0号流）的特殊行为？→ 默认流与所有非默认流同步（legacy 模式）；使用 `--default-stream per-thread` 可改为非阻塞行为。
- **代码**：
  ```c
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
  kernel<<<grid, block, 0, stream2>>>(d_b);
  ```

### 2. CUDA Event（事件）

- **定义**：Event 是流中的标记点，用于精确计时和流间同步。`cudaEventRecord` 记录时间戳，`cudaEventSynchronize` 等待完成。
- **追问**：Event 和 Stream Synchronize 的区别？→ Event 可跨流同步（`cudaStreamWaitEvent`），Stream Synchronize 只能等待单流完成。
- **代码**：
  ```c
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  kernel<<<grid, block, 0, stream>>>(...);
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float ms; cudaEventElapsedTime(&ms, start, stop);
  ```

### 3. 重叠传输与计算

- **定义**：使用双缓冲+多流，在一个流执行计算时，另一个流传输数据。需 Pinned Memory。
- **代码**：
  ```c
  for (int i = 0; i < N; i++) {
      int cur = i % 2;
      cudaStream_t s = streams[cur];
      cudaMemcpyAsync(d_buf[cur], h_buf[cur], size, cudaMemcpyHostToDevice, s);
      kernel<<<grid, block, 0, s>>>(d_buf[cur], d_out, ...);
      cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, s);
  }
  ```

### 4. CUDA Graph（图执行）

- **定义**：将多个 kernel 和内存操作构建为有向无环图，一次性提交执行。减少 CPU 端 launch 开销，适合重复执行的工作流。
- **追问**：Graph 与 Stream 的性能差异？→ Stream 每次启动 kernel 需 CPU → GPU 命令提交（~5-20μs），Graph 一次实例化后可重复执行，开销降至 ~1-2μs。
- **代码**：
  ```c
  cudaGraph_t graph;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel1<<<...>>>(...); kernel2<<<...>>>(...);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphLaunch(instance, stream); // 重复执行
  ```

### 5. MPS（Multi-Process Service）

- **定义**：NVIDIA 提供的多进程 GPU 共享服务，允许多个 CPU 进程的 kernel 并发在同一 GPU 上执行，提高小 kernel 的利用率。
- **追问**：MPS vs MIG？→ MPS 是软件级共享，进程间无隔离（地址空间共享）；MIG（A100+）是硬件级隔离，将 SM/内存物理分区，完全隔离。
- **使用**：`nvidia-cuda-mps-control` 启动守护进程，客户端通过 `CUDA_VISIBLE_DEVICES` 自动路由。

### 6. 动态并行（Dynamic Parallelism）

- **定义**：GPU 线程可以启动新的 kernel（子 kernel），无需返回 CPU。子 kernel 与父 kernel 在同一设备上执行。
- **追问**：动态并行的开销？→ 子 kernel 启动有额外延迟和资源消耗（需要独立的 launch setup），不如 CPU 端启动高效。适用于递归/不规则并行。
- **代码**：
  ```c
  __global__ void parent(...) {
      if (need_more_work) child<<<grid, block>>>(...);
      cudaDeviceSynchronize(); // 等待子 kernel（设备端同步）
  }
  ```

### 7. Cooperative Groups

- **定义**：CUDA 9 引入的灵活同步抽象，支持 thread block、warp、多 block 乃至多 GPU 的协同。替代了 `__syncthreads()` 和 warp shuffle 的硬编码用法。
- **追问**：CG 的优势？→ 类型安全的同步、可组合的同步范围、跨 block 同步（需 cooperative launch）。
- **代码**：
  ```c
  namespace cg = cooperative_groups;
  __global__ void kernel(...) {
      auto block = cg::this_thread_block();
      auto warp = cg::tiled_partition<32>(block);
      // block 级同步
      cg::sync(block);
      // warp 级 reduce
      float val = warp.shfl_down(val, 16);
  }
  ```

### 8. PTX 与 SASS

- **定义**：
  - **PTX**（Parallel Thread Execution）：NVIDIA 的中间指令集 ISA，类似汇编但与具体架构无关，由驱动 JIT 编译为 SASS。
  - **SASS**：特定 GPU 架构的机器码，实际硬件执行。
- **追问**：什么时候需要写 PTX？→ 极端性能优化（如手写 Tensor Core MMA 指令）、内联汇编（`asm volatile`）。绝大多数情况不需要。
- **代码**：
  ```c
  // 内联 PTX：warp reduce
  asm volatile("reduce.add.f32 %0, %0, %1;" : "+f"(val) : "f"(other));
  // 查看 SASS
  // cuobjdump -sass app.sm_80.cubin
  ```

### 9. Async Copy / TMA（Hopper+）

- **定义**：Ampere 引入 `cp.async` 指令异步从全局内存拷贝到共享内存；Hopper 引入 TMA（Tensor Memory Accelerator），硬件级异步批量张量传输，支持多维寻址。
- **追问**：TMA 的优势？→ 减轻线程负担（不需要所有线程参与加载），支持 swizzle 布局自动变换，支持边界处理。
- **代码**：
  ```c
  // Ampere async copy
  asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" \
      :: "r"(smem_ptr), "l"(gmem_ptr), "n"(16));
  cp.async.commit_group;
  cp.async.wait_group<0>;
  ```

---

## 模块八：常见面试陷阱与典型编程错误

### 1. 未检查 CUDA 错误

- **问题**：CUDA API 调用失败时默认不报错，静默返回错误码，导致后续结果全错。
- **正确做法**：所有 CUDA 调用后检查返回值，kernel 启动后用 `cudaGetLastError()` + `cudaDeviceSynchronize()` 检查。
- **代码**：
  ```c
  #define CUDA_CHECK(call) do { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
          fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
          exit(1); \
      } \
  } while(0)
  
  CUDA_CHECK(cudaMalloc(&d_data, size));
  kernel<<<grid, block>>>(...);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  ```

### 2. 忘记同步

- **问题**：共享内存写入后未 `__syncthreads()` 就读取，导致读到未更新的数据（数据竞争）。
- **追问**：Volta 的独立线程调度使得此问题更易出现？→ 是的，Volta 之前 warp 内 lockstep 隐式同步掩盖了部分 bug；Volta 后线程可独立调度，bug 暴露。
- **规则**：共享内存写后读必须同步；`__syncthreads()` 不能放在条件分支内（除非所有线程都走同一分支）。

### 3. 未释放 GPU 内存

- **问题**：`cudaMalloc` 后忘记 `cudaFree`，导致显存泄漏。长时间运行的训练任务尤其危险。
- **建议**：RAII 封装，使用智能指针或 cuBLAS/cuDNN 的 wrapper。

### 4. 越界访问

- **问题**：未做边界检查 `if (idx < N)`，当 N 不是 block 大小的整数倍时，多余线程越界读写。
- **代码**：
  ```c
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) out[idx] = a[idx] + b[idx]; // ✅ 必须
  ```

### 5. 错误的共享内存大小声明

- **问题**：动态共享内存通过 `extern __shared__` 声明，大小在 kernel 启动时指定；忘记指定或指定错误。
- **代码**：
  ```c
  __global__ void kernel(...) {
      extern __shared__ float sdata[];
  }
  // 启动时指定字节大小
  kernel<<<grid, block, N * sizeof(float), stream>>>(...);
  ```

### 6. 混淆 host 和 device 指针

- **问题**：在 host 代码解引用 device 指针，或在 kernel 中解引用 host 指针。统一内存可部分缓解，但非 UM 场景仍需小心。
- **代码**：
  ```c
  float *h_data = (float*)malloc(N * sizeof(float));  // host
  float *d_data; cudaMalloc(&d_data, N * sizeof(float)); // device
  // ❌ h_data[0] = d_data[0]; // 不能直接访问
  cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost); // ✅
  ```

### 7. 低效的原子操作使用

- **问题**：对热门地址（如 histogram 单 bin）大量原子操作导致严重序列化。
- **优化**：(1) 共享内存局部聚合再全局原子 (2) warp 聚合（warp 内用 ballot/shuffle 合并相同 key）(3) 使用 `atomicAdd` 的 block-level 聚合。

### 8. 忽略 CUDA 核函数的异步性

- **问题**：kernel 启动是异步的，`cudaMemcpy`（非 Async 版本）会隐式同步，但 `cudaMemcpyAsync` 不会。混用导致数据不一致。
- **规则**：理解哪些 API 是同步的（`cudaMemcpy`、`cudaDeviceSynchronize`），哪些是异步的（`cudaMemcpyAsync`、kernel launch）。

### 9. Grid Stride Loop 模式

- **最佳实践**：当数据量远大于线程数时，使用 grid stride loop 而非仅一个元素/线程。
- **代码**：
  ```c
  __global__ void kernel(float *data, int N) {
      for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
          data[i] = data[i] * 2.0f;
      }
  }
  ```

### 10. 忽视 float 精度问题

- **问题**：`float` 只有 7 位有效数字，大数加小数会丢失精度（如 reduce 求和）。GPU 上 double 慢且 Tensor Core 不支持。
- **建议**：使用 Kahan summation、分块求和后再归约、或分层累加。

---

## 附录：常见面试高频题速查

| 题目 | 关键考点 |
|------|----------|
| 实现 reduce 并逐步优化 | 合并访问、bank conflict、warp 同步优化 |
| 实现 SGEMM | 分块、共享内存、寄存器分块、bank conflict |
| 实现 Softmax | 数值稳定性、online softmax |
| Flash Attention 原理 | 分块计算、online softmax 修正、IO 复杂度 |
| 为什么 GPU 适合深度学习 | 大规模并行、高带宽、Tensor Core |
| 共享内存 bank conflict 如何解决j mm | Padding、交错访问、shuffle |
| 如何分析 kernel 性能瓶颈 | Roofline、ncu、SM Busy vs Memory Throughput |
| CUDA Stream 和 Event 的作用 | 异步执行、重叠传输计算、精确计时 |
| 占用率的影响因素 | 寄存器数、共享内存、block 大小 |
| GPU 和 CPU 的根本区别 | 吞吐量 vs 延迟、SIMT vs SMT、内存层次 |
