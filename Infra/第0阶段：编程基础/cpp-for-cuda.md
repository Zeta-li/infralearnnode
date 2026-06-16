# C++ 基础知识系统学习：面向 CUDA 算子开发

> **目标读者**：准备学习 CUDA Core / Tensor Core / CUTLASS 的开发者，已完成 C 语言入门。
> **文档定位**：按 CUDA 开发的实际需求组织知识点，每个章节标注「CUDA 相关度」并给出 GPU 编程场景示例。

---

## 目录

1. [前言：C++ 与 CUDA 的关系](#1-前言c-与-cuda-的关系)
2. [基础语法速览](#2-基础语法速览)
3. [指针、引用与内存模型](#3-指针引用与内存模型)
4. [类、结构体与 RAII](#4-类结构体与-raii)
5. [模板基础：从函数模板到类模板](#5-模板基础从函数模板到类模板)
6. [模板元编程进阶](#6-模板元编程进阶)
7. [constexpr 与编译期计算](#7-constexpr-与编译期计算)
8. [Lambda 表达式与函数对象](#8-lambda-表达式与函数对象)
9. [运算符重载](#9-运算符重载)
10. [内存布局、对齐与类型转换](#10-内存布局对齐与类型转换)
11. [继承、多态与虚函数](#11-继承多态与虚函数)
12. [智能指针与 Host 侧资源管理](#12-智能指针与-host-侧资源管理)
13. [编译、链接与构建系统](#13-编译链接与构建系统)
14. [C++17/20 关键特性速览](#14-c1720-关键特性速览)
15. [综合练习与学习路径建议](#15-综合练习与学习路径建议)

---

## 1. 前言：C++ 与 CUDA 的关系

### 1.1 为什么 CUDA 需要 C++

| CUDA 组件 | 依赖的 C++ 特性 |
|-----------|----------------|
| **CUDA Runtime API** | 类、模板、异常处理 (host side) |
| **Kernel 函数** | 函数模板、lambda（`--extended-lambda`） |
| **Tensor Core (wmma/mma)** | 模板参数、类型萃取、`constexpr` |
| **CUTLASS** | 全量模板元编程：SFINAE、变参模板、CRTP、类型萃取、编译期计算 |
| **Thrust / CUB** | 模板、迭代器、函数对象、lambda |
| **自定义内存分配器** | RAII、智能指针、`new`/`delete` 重载 |

### 1.2 学习策略

- **Host 侧**：完整 C++17，与你日常写 C++ 一致
- **Device 侧**：C++ 子集（无异常、无 RTTI、无 `std::` 大部分容器、`__device__` 限定）
- **元编程**：几乎全部发生在编译期，host/device 通用

> 本文所有代码示例在标注 `// host` 时仅 host 可用，标注 `// device` 时需加上 `__device__`，未标注则 host/device 通用。

---

## 2. 基础语法速览

> **CUDA 相关度**：★★☆☆☆（基础，但必须熟练掌握）

### 2.1 命名空间

```cpp
#include <iostream>

// 自定义命名空间避免符号冲突
namespace cuda_utils {
    void check_error(cudaError_t err) { /* ... */ }
}

// 使用
using cuda_utils::check_error;
// 或
using namespace cuda_utils;  // 谨慎使用

// CUDA 场景：CUTLASS 大量使用嵌套命名空间
// cutlass::gemm::kernel::DefaultGemm<uint8_t, ...>
```

### 2.2 auto 类型推导

```cpp
auto a = 3.14f;            // float
auto b = 42;               // int
auto& c = a;               // float&
const auto& d = a;         // const float&

// CUDA 场景：避免手写冗长的迭代器/模板类型
auto ptr = thrust::device_pointer_cast(dev_ptr);
auto policy = thrust::cuda::par.on(stream);

// 注意：kernel 内 auto 可用（SM 3.5+）
__global__ void kernel(float* data, int N) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;  // int
}
```

const auto* a❌ 改*a的值✅ 改a的指向

auto* const a✅ 改*a的值❌ 改a的指向

| 写法            | 结果   | 含义                                  |
| --------------- | ------ | ------------------------------------- |
| `const auto& a` | ✅ 正确 | **指向常量的引用**（不能通过 a 改值） |
| `auto& const a` | ❌ 错误 | 语法非法，引用不能加 const            |

引用一旦绑定，就永远不能再绑定到别的变量

### 2.3 范围 for 循环

```cpp
std::vector<float> vec = {1.0f, 2.0f, 3.0f};
for (const auto& v : vec) { /* ... */ }  // host only

// CUDA 场景：host 侧遍历配置参数
std::vector<cudaStream_t> streams(4);
for (auto& s : streams) { cudaStreamCreate(&s); }
```

### 2.4 结构化绑定（C++17）

```cpp
std::pair<int, float> p{42, 3.14f};
auto [x, y] = p;           // x = 42, y = 3.14f

struct KernelConfig { int grid; int block; size_t smem; };
KernelConfig cfg{128, 256, 49152};
auto [grid, block, smem] = cfg;  // 解包 kernel 配置

// CUDA 场景：解包多返回值配置
```

### 2.5 枚举类（enum class）

```cpp
// 推荐：类型安全的枚举
enum class DataType { FP32, FP16, BF16, INT8, INT4 };
enum class Layout   { RowMajor, ColMajor, TensorCore };

// CUDA 场景：CUTLASS 大量使用 enum class 控制模板参数
template <DataType D, Layout L>
struct GemmConfig { /* ... */ };
```

---

## 3. 指针、引用与内存模型

> **CUDA 相关度**：★★★★★（核心中的核心）

### 3.1 指针基础回顾

```cpp
int val = 100;
int* ptr = &val;         // ptr 存储 val 的地址
*ptr = 200;              // 通过指针修改 val
int** dptr = &ptr;       // 二级指针

// CUDA 场景：指针是 GPU 编程的"通用语言"
float* d_A;              // device 指针
cudaMalloc(&d_A, N * sizeof(float));  // &d_A 是 float**（二级指针）
cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
```

### 3.2 指针算术

```cpp
float arr[4] = {1.0f, 2.0f, 3.0f, 4.0f};
float* p = arr;
float v1 = *(p + 2);     // arr[2] = 3.0f——偏移 2 个元素
float v2 = p[2];         // 等价写法

// CUDA 场景：kernel 内工作集划分
template <typename T>
__global__ void process_tile(T* data, int N, int tile_size) {
    // 指针偏移到当前 tile
    T* tile = data + blockIdx.x * tile_size;  // 指针算术 = 块内数据起始地址
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        tile[i] = ...;
    }
}
```

C++ 函数传参默认是**值拷贝**。

如果你想在函数里**修改外面的指针指向哪里**，必须用**二级指针**。

```c++
void changePtr(int** pp) {
    // 修改外面的指针 p
    *pp = new int(999);
}

int main() {
    int* p = nullptr;
    changePtr(&p);  // 必须传指针的地址
    cout << *p;     // 999 ✅
}
```

```
1. 常量指针 (Pointer to Constant)
定义：const int *p; 或 int const *p;（const 在 * 左侧）

含义：p 是指向一个整型常量的指针。即 *p 是常量，不能修改，但 p 可以指向其他变量。


2. 指针常量 (Constant Pointer)
定义：int * const p;（const 在 * 右侧）

含义：p 本身是常量，必须初始化，之后不能再指向其他地址，但可以通过 p 修改指向的变量的值。


常量指针：const int *p → 指向的值不能改，但指针可指向别处。

指针常量：int * const p → 指针本身不能改，但指向的值可改。

引用在行为上非常像一个指针常量，没有自己的地址
```



### 3.3 引用（Reference）

```cpp
// 引用是别名，没有空引用，语法更安全
int x = 10;
int& ref = x;            // ref 就是 x
ref = 20;                // x 也变成 20

// CUDA 场景：host 侧包装类
void launch_kernel(const KernelConfig& cfg) {  // 避免拷贝，保证不为空
    my_kernel<<<cfg.grid, cfg.block, cfg.smem>>>(...);
}

// const 引用：只读 + 避免拷贝
void check_config(const std::vector<KernelConfig>& configs) { /* ... */ }
```

### 3.4 指针 vs 引用 对比

| 特性 | 指针 `T*` | 引用 `T&` |
|------|----------|----------|
| 可为空 | `nullptr` | 否 |
| 可重新绑定 | 是 | 否（终身绑定） |
| 语法 | `*p`, `p->` | 直接使用 |
| CUDA 通信 | 设备指针（核心） | Host 侧参数传递 |
| 适用场景 | 显存地址、数组遍历 | 参数传递、返回值优化 |

```cpp
// 典型 CUDA 混合使用
void my_function(const int& N,        // 引用：只读，不为空
                 float* d_output)     // 指针：device 地址，可能为空需检查
{
    if (d_output == nullptr) { /* error handling */ }
}
```

### 3.5 动态内存分配（C++ 风格）

```cpp
// C++ 方式（推荐用于 host 侧）
float* h_A = new float[N];          // 分配
delete[] h_A;                        // 释放

// C 方式（CUDA kernel 常用，因为 cudaMalloc 遵循 C 接口）
float* d_A;
cudaMalloc((void**)&d_A, N * sizeof(float));  // C 风格
cudaFree(d_A);

// RAII 封装（见第 4 章 + 第 12 章）
```

### 3.6 this 指针

```cpp
struct Tensor {
    float* data;
    int size;

    // this 是隐式指针，指向调用对象
    void fill(float val) {
        for (int i = 0; i < this->size; ++i)  // this-> 可省略
            this->data[i] = val;
    }
};

// CUDA 场景：CUTLASS 中每个算子类都有大量 this-> 引用成员
```

---

## 4. 类、结构体与 RAII

> **CUDA 相关度**：★★★★☆

### 4.1 struct vs class

```cpp
// 唯一区别：默认访问权限
struct KernelConfig {    // 默认 public
    int grid;
    int block;
};

class CudaArray {        // 默认 private
    float* data_;
public:
    CudaArray(int N) { cudaMalloc(&data_, N * sizeof(float)); }
    ~CudaArray() { cudaFree(data_); }
};

// CUDA 场景：POD struct 用于 kernel 参数（struct 更自然）
struct __align__(16) KernelParams {
    const float* __restrict__ input;
    float* __restrict__ output;
    int N;
    float alpha;
};
```

### 4.2 构造函数与析构函数

```cpp
class DeviceBuffer {
    float* data_;
    size_t size_;
public:
    // 默认构造
    DeviceBuffer() : data_(nullptr), size_(0) {}

    // 带参构造
    explicit DeviceBuffer(size_t N) : size_(N) {
        cudaMalloc(&data_, N * sizeof(float));
    }

    // 析构：自动释放资源
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }

    // 禁止拷贝（GPU 资源不应浅拷贝）
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // 移动构造（见 4.5 节）
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
};
```

### 4.3 初始化列表

```cpp
// 初始化列表比构造函数体内赋值更高效（直接构造）
class GemmConfig {
    int M_, N_, K_;
public:
    // 初始化列表
    GemmConfig(int M, int N, int K) : M_(M), N_(N), K_(K) {}

    // 不要这样写（先默认构造再赋值，const/引用成员无法这样初始化）
    // GemmConfig(int M, int N, int K) { M_ = M; N_ = N; K_ = K; }
};
```

### 4.4 RAII（资源获取即初始化）

CUDA 编程中最核心的 C++ 惯用法：

```cpp
// 手动管理：容易因为提前 return / 异常而泄漏
void manual_way() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));  // 若这行失败，d_A 泄漏
    cudaMalloc(&d_C, N * sizeof(float));

    // ... kernel launch ...

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  // 若中间 return 了，这些都不会执行
}

// RAII 方式：资源生命周期绑定到对象生命周期
class CudaMemory {
    void* ptr_;
public:
    CudaMemory(size_t bytes) { cudaMalloc(&ptr_, bytes); }
    ~CudaMemory() { if (ptr_) cudaFree(ptr_); }
    void* get() { return ptr_; }
    // 禁止拷贝 ...
};

void safe_way() {
    CudaMemory d_A(N * sizeof(float));   // 分配
    CudaMemory d_B(N * sizeof(float));
    CudaMemory d_C(N * sizeof(float));

    // ... kernel launch ...

    // 无论函数如何退出，析构函数自动释放所有显存
}  // d_A, d_B, d_C 出作用域时自动析构
```

### 4.5 移动语义

```cpp
// 移动构造函数：转移资源所有权（而非拷贝）
DeviceBuffer buf1(1024);                // 分配 1024 个 float
DeviceBuffer buf2(std::move(buf1));     // buf1 所有权转移给 buf2
// buf1 现在为空，buf2 持有资源

// CUDA 场景：容器中存储 device buffer
std::vector<DeviceBuffer> buffers;
buffers.push_back(DeviceBuffer(4096));  // 触发移动构造，避免 GPU 内存拷贝

// 工厂函数返回大型对象
DeviceBuffer create_buffer(size_t N) {
    DeviceBuffer buf(N);                // 局部对象
    // 赋值数据...
    return buf;                         // C++11+：自动移动（NRVO 通常优化掉移动）
}
```

### 4.6 静态成员

```cpp
class CudaUtils {
public:
    static constexpr int WARP_SIZE = 32;    // 编译期常量

    static int get_device_count() {         // 静态方法，无需实例
        int count;
        cudaGetDeviceCount(&count);
        return count;
    }
};

// 使用
int warp = CudaUtils::WARP_SIZE;            // 不创建对象直接访问
int devs = CudaUtils::get_device_count();
```

---

1. **`unique_ptr`**：独生子，只能移动不能拷贝。开销小，优先使用。
2. **`shared_ptr`**：多人共享，引用计数。注意循环引用 → 用 `weak_ptr` 打破。
3. **`weak_ptr`**：给 `shared_ptr` 当小弟，不增加计数，用来解决循环依赖。
4. `shared_ptr` 的引用计数机制无法感知循环：A 引用 B，B 引用 A，形成一个闭环，谁也等不到对方先释放。
5. `weak_ptr` **不增加引用计数**，只是对资源的**弱引用**。它不阻止对象销毁，因此闭环中的某个环节可以自然消失。

## 5. 模板基础：从函数模板到类模板

> **CUDA 相关度**：★★★★★（CUDA/CUTLASS 的灵魂——几乎所有 API 都是模板化的）

### 5.1 函数模板

```cpp
// 问题：一个 reduce kernel 要支持 float/half/int8_t 怎么办？
// 方案：函数模板——由编译器自动生成多个版本

template <typename T>
__global__ void sum_kernel(const T* input, T* output, int N) {
    T sum = T(0);               // 零值的通用写法
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += input[i];
    }
    // ... warp reduce ...
}

// 实例化：编译器自动生成 float 版和 half 版
sum_kernel<float><<<1, 256>>>(d_in, d_out, N);
sum_kernel<half><<<1, 256>>>(d_in, d_out, N);

// ⚠️ PTX 阶段才生成具体代码，语法错误可能到模板实例化时才暴露
```

### 5.2 非类型模板参数（NTTP）

```cpp
// 将编译期常量作为模板参数——CUDA kernel 配置的常见模式
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void reduce_kernel(const float* input, float* output, int N) {
    // BLOCK_SIZE 和 ITEMS_PER_THREAD 是编译期常量 → 展开循环、寄存器分配优化
    float items[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        items[i] = input[...];
    }
    // ...
}

// 不同配置生成不同 kernel
reduce_kernel<256, 4><<<grid, 256>>>(d_in, d_out, N);
reduce_kernel<128, 8><<<grid, 128>>>(d_in, d_out, N);
```

### 5.3 类模板

```cpp
// CUTLASS 的核心模式：操作语义 → 模板参数
template <typename T, int Alignment>
class AlignedArray {
    T* data_;
public:
    // Alignment 编译期已知 → 编译器可向量化加载
    __device__ void load(const T* global_ptr) {
        // 若 T=float, Alignment=4 → 128-bit 向量加载（单指令完成）
    }
};

// 实例化
AlignedArray<float, 4> arr;    // 128-bit 对齐加载
AlignedArray<half, 8>  arr2;  // 同样是 128-bit
```

### 5.4 模板特化（Template Specialization）

```cpp
// 通用模板
template <typename T>
struct TypeTraits {
    static constexpr bool is_floating = false;
};

// 完全特化：为 float 定制
template <>
struct TypeTraits<float> {
    static constexpr bool is_floating = true;
    static constexpr int  precision = 32;
};

// 部分特化：为一组类型定制
template <typename T>
struct TypeTraits<const T> {
    static constexpr bool is_floating = TypeTraits<T>::is_floating;
    static constexpr bool is_const     = true;
};

// CUDA 场景量化
template <>
struct TypeTraits<int8_t> {
    static constexpr bool is_floating = false;
    static constexpr bool is_signed   = true;
    static constexpr int  bit_width   = 8;
};
```

### 5.5 默认模板参数

```cpp
template <typename T,           // 数据类型
          int BLOCK = 256,      // 默认 block 大小
          int ITEMS = 4>        // 默认每线程处理数
__global__ void generic_kernel(const T* in, T* out, int N) { /* ... */ }

generic_kernel<float><<<grid, 256>>>(d_in, d_out, N);           // BLOCK=256, ITEMS=4
generic_kernel<float, 128, 8><<<grid, 128>>>(d_in, d_out, N);  // BLOCK=128, ITEMS=8
```

---

## 6. 模板元编程进阶

> **CUDA 相关度**：★★★★★（CUTLASS 的核心语言——如果你看不懂 CUTLASS 源码，问题通常在这里）

### 6.1 类型萃取（Type Traits）

```cpp
#include <type_traits>

// 编译期检查类型属性
static_assert(std::is_same<float, float>::value, "must be same");
static_assert(std::is_floating_point<float>::value, "");
static_assert(std::is_integral<int>::value, "");

// CUDA 场景：kernel 内部根据类型选择不同算法路径
template <typename T>
__global__ void compute_kernel(T* data, int N) {
    // if constexpr：编译期分支，不增加运行时开销
    if constexpr (std::is_same_v<T, half>) {
        // half 专用路径：使用 __hfma、__hadd 等 intrinsic
    } else if constexpr (std::is_same_v<T, float>) {
        // float 专用路径：使用 __fmaf_rn 等
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}
```

**CUTLASS 中类型萃取的典型用法**：

```cpp
// 检查某类型是否能用于 Tensor Core
template <typename T>
struct is_tensorcore_eligible : std::false_type {};

template <>
struct is_tensorcore_eligible<half> : std::true_type {};

template <>
struct is_tensorcore_eligible<tfloat32_t> : std::true_type {};

// 使用
template <typename T,
          typename = std::enable_if_t<is_tensorcore_eligible<T>::value>>
struct TensorCoreGemm { /* ... */ };
```

### 6.2 SFINAE 与 enable_if

**SFINAE** = Substitution Failure Is Not An Error：模板替换失败不报错，而是从候选集中移除。

```cpp
// 方案 1：enable_if 条件启用模板
// 仅当 T 是浮点类型时启用此重载
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>, void>
launch_gemm(T alpha, /* ... */) {
    // 浮点 GEMM
}

// 仅当 T 是整型时启用此重载
template <typename T>
std::enable_if_t<std::is_integral_v<T>, void>
launch_gemm(T alpha, /* ... */) {
    // 整型 GEMM（量化推理场景）
}

// 方案 2：if constexpr（C++17，更推荐）
template <typename T>
void launch_gemm(T alpha, /* ... */) {
    if constexpr (std::is_floating_point_v<T>) {
        // 浮点路径
    } else {
        // 整型路径
    }
}
```

### 6.3 变参模板（Variadic Templates）

```cpp
// 递归展开：编译期计算所有参数的乘积（用于计算 tile 元素总数）
template <int... Dims>
struct Product;

// 基础情况：空包 = 1
template <>
struct Product<> : std::integral_constant<int, 1> {};

// 递归情况
template <int First, int... Rest>
struct Product<First, Rest...>
    : std::integral_constant<int, First * Product<Rest...>::value> {};

// C++17 折叠表达式简化版
template <int... Dims>
constexpr int product_v = (... * Dims);

// CUDA 场景：多维 tile 定义
static_assert(product_v<16, 8, 8> == 1024);   // 16×8×8 tile

// CUTLASS 场景：多维坐标
template <int... Dims>
struct Coord {
    static constexpr int rank = sizeof...(Dims);
};

using TileShape = Coord<128, 128, 32>;  // M-N-K tile
```

### 6.4 CRTP（奇异递归模板模式）

CRTP 是 CUTLASS 实现静态多态的核心模式。

```cpp
// 基类模板：参数是派生类自身
template <typename Derived>
class OperatorBase {
public:
    // 静态多态：编译期绑定，无虚函数开销
    __device__ float apply(float a, float b) const {
        return static_cast<const Derived*>(this)->apply_impl(a, b);
    }

    float host_apply(float a, float b) const {
        return static_cast<const Derived*>(this)->apply_impl(a, b);
    }
};

// 派生类：通过模板参数注入实现
class AddOp : public OperatorBase<AddOp> {
public:
    __device__ __host__ float apply_impl(float a, float b) const {
        return a + b;  // 此处可以用 __fadd_rn 等 intrinsic
    }
};

class MulOp : public OperatorBase<MulOp> {
public:
    __device__ __host__ float apply_impl(float a, float b) const {
        return a * b;
    }
};

// 使用：零虚函数开销的"多态"
template <typename Op>
__global__ void elementwise_kernel(const float* A, const float* B,
                                    float* C, int N, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = op.apply(A[idx], B[idx]);  // 编译期静态分发
    }
}
```

### 6.5 using 别名与模板别名

```cpp
// 传统 typedef（不推荐用于模板）
typedef unsigned int uint32_t;

// C++11 using 别名（推荐，支持模板）
template <typename T>
using DeviceVector = thrust::device_vector<T>;

using FloatVector = DeviceVector<float>;  // 等价于 thrust::device_vector<float>

// CUTLASS 场景：减少模板参数传递的重复
template <typename T, int M, int N, int K>
class Gemm;

template <typename T>
using Gemm128x128x32 = Gemm<T, 128, 128, 32>;

template <typename T>
using Gemm256x128x64 = Gemm<T, 256, 128, 64>;
```

---

## 7. constexpr 与编译期计算

> **CUDA 相关度**：★★★★☆（优化 kernel 配置、tile size 计算、寄存器预估）

### 7.1 constexpr 变量

```cpp
// 编译期常量——kernel 配置的黄金搭档
constexpr int WARP_SIZE     = 32;
constexpr int BLOCK_SIZE    = 256;
constexpr int WARPS_PER_BLK = BLOCK_SIZE / WARP_SIZE;  // = 8

// const vs constexpr
const     int runtime_val = argc;             // 运行时确定
constexpr int compile_val = 256 * 32;         // 编译期确定（可用于数组大小/模板参数）

// CUDA 场景
template <int BLOCK>
__global__ void kernel() {
    __shared__ float smem[BLOCK];              // 必须编译期确定大小
}
```

### 7.2 constexpr 函数

```cpp
// 在编译期计算 tile 的 shared memory 大小
constexpr size_t smem_size(int tile_m, int tile_n, int type_size) {
    return tile_m * tile_n * type_size;
}

// 用于 kernel 启动配置时完全在编译期求值
constexpr size_t SMEM = smem_size(128, 128, sizeof(half));  // = 32768 bytes

my_kernel<<<grid, block, SMEM>>>(...);

// 更复杂的例子：计算 pad 后的 shared memory 大小（避免 bank conflict）
constexpr int padded_dim(int dim, int bank_bytes) {
    return dim + (bank_bytes / sizeof(float)) - (dim % (bank_bytes / sizeof(float)));
}
```

### 7.3 constexpr if（C++17）

```cpp
// 优于 #ifdef，类型安全且语法友好
template <typename T>
__global__ void flexible_kernel(T* data, int N) {
    __shared__ union {
        float f_data[256];
        half  h_data[256];
        int8_t i8_data[256];
    } smem;

    if constexpr (std::is_same_v<T, float>) {
        smem.f_data[threadIdx.x] = data[threadIdx.x + blockIdx.x * blockDim.x];
    } else if constexpr (std::is_same_v<T, half>) {
        smem.h_data[threadIdx.x] = data[threadIdx.x + blockIdx.x * blockDim.x];
    }
    // 编译器只生成匹配的分支代码，其他分支完全消失
}
```

---

## 8. Lambda 表达式与函数对象

> **CUDA 相关度**：★★★★☆（Kernel fusion、Thrust 算法、自定义算子）

### 8.1 Lambda 基础语法

```cpp
// [捕获] (参数) -> 返回类型 { 函数体 }

auto add = [](float a, float b) -> float { return a + b; };
auto mul = [](float a, float b) { return a * b; };  // 返回类型可推导

// 捕获
float alpha = 2.0f;
auto scale = [alpha](float x) { return alpha * x; };    // 值捕获
auto scale_ref = [&alpha](float x) { return alpha * x; }; // 引用捕获

// CUDA 场景：Thrust 算法
thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(),
    [] __device__ (float a, float b) {
        return a * b + 1.0f;  // FMA 操作
    }
);
```

### 8.2 Lambda 与函数对象等价性

```cpp
// Lambda 本质上是匿名函数对象（functor）的语法糖
auto lambda = [alpha](float x) { return alpha * x; };

// 等价于：
struct AnonymousFunctor {
    float alpha;
    AnonymousFunctor(float a) : alpha(a) {}
    __host__ __device__ float operator()(float x) const { return alpha * x; }
};
```

### 8.3 Device Lambda（CUDA C++14+）

```cpp
#include <cuda/std/functional>

// __device__ lambda：在 GPU 上执行的 lambda
auto device_op = [] __device__ (float a, float b) -> float {
    return fmaxf(a, b);
};

// 用于自定义 kernel fusion（如 element-wise 操作组合）
template <typename Func>
__global__ void transform_kernel(float* data, int N, Func op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { data[idx] = op(data[idx]); }
}

float scale = 2.0f;
transform_kernel<<<grid, block>>>(d_data, N,
    [scale] __device__ (float x) { return scale * x + 1.0f; }
);
```

---

## 9. 运算符重载

> **CUDA 相关度**：★★★☆☆（half/bf16 类型的算术操作、矩阵类）

### 9.1 成员函数 vs 全局函数重载

```cpp
// 场景：给自定义 half2 类型添加运算符（CUDA 中 half 自身不支持算术运算）

struct Vec3 {
    float x, y, z;

    // 成员函数重载（二元运算符）
    Vec3 operator+(const Vec3& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z};
    }

    Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }

    // 一元运算符
    Vec3 operator-() const {
        return {-x, -y, -z};
    }
};

// 全局函数重载（便于左操作数隐式转换）
Vec3 operator*(float scalar, const Vec3& v) {
    return {scalar * v.x, scalar * v.y, scalar * v.z};
}

// 流输出重载（常用）
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}
```

### 9.2 常用可重载的运算符

| 类别 | 运算符 | CUDA 场景 |
|------|--------|----------|
| 算术 | `+ - * / %` | 矩阵/向量运算 |
| 复合赋值 | `+= -= *= /=` | 梯度累积 |
| 比较 | `== != < > <= >=` | 排序、搜索 |
| 下标 | `[]` | `Tensor<float>[i][j]` |
| 函数调用 | `()` | 自定义激活函数（functor） |
| 类型转换 | `operator T()` | `half` → `float` 隐式转换 |

### 9.3 函数调用运算符（Functor）

```cpp
// CUTLASS 中 epilogue 算子通过 operator() 定义

struct Relu {
    template <typename T>
    __device__ T operator()(T x) const {
        return (x > T(0)) ? x : T(0);
    }
};

struct Gelu {
    template <typename T>
    __device__ T operator()(T x) const {
        // GELU 近似计算
        constexpr T c = T(0.044715);
        T x3 = x * x * x;
        return T(0.5) * x * (T(1.0) + tanh(T(0.7978845608) * (x + c * x3)));
    }
};

// 将激活函数作为模板参数注入 GEMM
template <typename EpilogueOp>
__global__ void gemm_with_activation(/* ... */, EpilogueOp op) {
    // ... GEMM 计算 ...
    float acc = /* ... */;
    result = op(acc);  // 编译期多态，零开销
}

gemm_with_activation<<<...>>>(/* ... */, Relu{});
gemm_with_activation<<<...>>>(/* ... */, Gelu{});
```

---

## 10. 内存布局、对齐与类型转换

> **CUDA 相关度**：★★★★★（访存合并、bank conflict、向量化加载）

### 10.1 内存对齐

```cpp
#include <cstddef>

// alignof / alignas (C++11)
struct Unaligned {
    char   a;    // 1 byte
    int    b;    // 4 bytes (需要 4 字节对齐)
    double c;    // 8 bytes (需要 8 字节对齐)
};
// sizeof(Unaligned) = 24 (填充了 3+4 字节)

struct alignas(16) Aligned16 {
    char c;
};
// sizeof(Aligned16) = 16 (强制 16 字节对齐)

// CUDA 场景：128-bit 向量化加载需要 16 字节对齐
struct alignas(16) float4_aligned {
    float x, y, z, w;
};  // sizeof = 16

// 使用 float4/int4 进行向量化全局内存访问
__global__ void vectorized_copy(const float4* __restrict__ src,
                                      float4* __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];  // 单指令加载 128-bit = 4 个 float
    }
}
```

### 10.2 C++ 风格类型转换

```cpp
// ⚠️ 避免 C 风格转换（坑多），使用 C++ 命名转换

// 1. static_cast：相关类型间的显式转换（编译期检查）
float f = static_cast<float>(42);          // int → float
int* p = static_cast<int*>(malloc(100));  // void* → int*

// 2. reinterpret_cast：不相关类型的位重解释（CUDA 中极常用）
float* d_float = /* ... */;
// 将 float* 转为 half*——相同显存，不同解释方式
half*  d_half  = reinterpret_cast<half*>(d_float);

// 3. const_cast：移除 const（谨慎使用）
const float* c_ptr = /* ... */;
float* mutable_ptr = const_cast<float*>(c_ptr);  // kernel 调用需要非 const

// 4. dynamic_cast：运行时多态转换（host only，device 不支持 RTTI）

// CUDA 典型模式：cudaMalloc 返回 void**，需 reinterpret_cast
float* d_A;
cudaMalloc(reinterpret_cast<void**>(&d_A), N * sizeof(float));
```

### 10.3 POD 与 Trivial 类型

```cpp
#include <type_traits>

// POD（Plain Old Data）：可在 host/device 间安全 memcpy 的类型
struct KernelParams {
    float* d_input;
    float* d_output;
    int    N;
    float  alpha;
};

static_assert(std::is_trivially_copyable_v<KernelParams>,
              "Kernel params must be trivially copyable");

// 传递给 kernel（必须 trivially copyable）
KernelParams params{d_in, d_out, N, 0.5f};
cudaMemcpy(d_params, &params, sizeof(params), cudaMemcpyHostToDevice);
```

### 10.4 共用体（union）—— Shared Memory 复用

```cpp
// CUDA 经典模式：同一块 shared memory，不同阶段解释为不同类型
__global__ void mixed_precision_kernel(half* input, float* output, int N) {
    __shared__ union {
        half  h_buf[1024];  // 前半部分：存储 half
        float f_buf[512];   // 后半部分：作为 float 累加
    } smem;

    // 阶段 1：half 加载
    smem.h_buf[threadIdx.x] = input[threadIdx.x + blockIdx.x * blockDim.x];

    __syncthreads();

    // 阶段 2：float 累加（复用同一块显存）
    smem.f_buf[threadIdx.x / 2] = __half2float(smem.h_buf[threadIdx.x]);

    __syncthreads();

    // warp reduce on float...
}
```

---

## 11. 继承、多态与虚函数

> **CUDA 相关度**：★★★☆☆（CUTLASS 大量使用继承构建层级结构，但很少用虚函数）

### 11.1 公有继承

```cpp
// 基类：定义通用 GEMM 接口
class GemmBase {
protected:
    int M_, N_, K_;
public:
    GemmBase(int M, int N, int K) : M_(M), N_(N), K_(K) {}
    virtual ~GemmBase() = default;  // 虚析构（host only）

    // 纯虚函数：派生类必须实现
    virtual void run(cudaStream_t stream = 0) = 0;

    // 非虚函数：通用实现
    int flops() const { return 2 * M_ * N_ * K_; }
};

// 派生类：具体实现
template <typename TA, typename TB, typename TC>
class GemmFP16 : public GemmBase {
public:
    using GemmBase::GemmBase;  // 继承构造函数

    void run(cudaStream_t stream = 0) override {
        // 调用具体的 FP16 Tensor Core GEMM
    }
};
```

### 11.2 运行期多态 vs 编译期多态

| 特性 | 运行期多态 (virtual) | 编译期多态 (CRTP/模板) |
|------|---------------------|----------------------|
| 机制 | 虚函数表 (vtable) | 模板实例化 |
| 开销 | vtable 查找 + 无法内联 | 零开销，完全内联 |
| Device 支持 | ❌ 不支持 | ✅ 完全支持 |
| CUDA 使用 | Host 侧接口 | Kernel 内 + CUTLASS 核心 |

```cpp
// CUTLASS 风格：全编译期多态，零 GPU 开销
template <typename Gemm_>
class MyGemm {
    using Gemm = Gemm_;
    typename Gemm::Params params_;

public:
    // 编译期绑定——无 vtable，device 可用
    void run() {
        typename Gemm::Kernel kernel;
        kernel(params_);
    }
};
```

### 11.3 多重继承

```cpp
// CUTLASS 使用多重继承组合能力
class TensorCore {};      // mixin：标记为 Tensor Core 算子
class Fp16Accum {};        // mixin：标记为 FP16 累加器

class MyGemm : public TensorCore, public Fp16Accum {
    // 同时具备两种能力
};

// 判断能力
template <typename T>
constexpr bool is_tensor_core = std::is_base_of_v<TensorCore, T>;
```

---

## 12. 智能指针与 Host 侧资源管理

> **CUDA 相关度**：★★★☆☆（仅 host 侧，管理 GPU 显存的 RAII 包装）

### 12.1 unique_ptr 与自定义删除器

```cpp
#include <memory>

// unique_ptr + 自定义删除器 = GPU 内存的 RAII 管理
struct CudaDeleter {
    void operator()(void* ptr) const {
        cudaFree(ptr);  // 自动调用 cudaFree
    }
};

template <typename T>
std::unique_ptr<T, CudaDeleter> make_device_buffer(size_t count) {
    T* ptr = nullptr;
    cudaMalloc(&ptr, count * sizeof(T));
    return std::unique_ptr<T, CudaDeleter>(ptr);
}

// 使用
void my_computation(int N) {
    auto d_A = make_device_buffer<float>(N);      // 自动分配
    auto d_B = make_device_buffer<float>(N);
    auto d_C = make_device_buffer<float>(N);

    launch_kernel(d_A.get(), d_B.get(), d_C.get(), N);

    // 无需手动 cudaFree：unique_ptr 出作用域时自动释放
}  // 无论正常返回还是异常退出，内存都会被释放

// cudaStream_t 也可以用相同模式
struct StreamDeleter {
    void operator()(cudaStream_t* stream) const {
        cudaStreamDestroy(*stream);
        delete stream;
    }
};

auto stream = std::unique_ptr<cudaStream_t, StreamDeleter>(
    new cudaStream_t,
    StreamDeleter{}
);
cudaStreamCreate(stream.get());
```

### 12.2 shared_ptr

```cpp
// GPU 内存共享所有权（多个对象可能引用同一块显存）
std::shared_ptr<float> d_A = make_device_buffer_shared<float>(N);

{
    auto d_B = d_A;  // 引用计数 = 2，不拷贝显存
    // ... 使用 d_B ...
}  // d_B 析构，引用计数 = 1，不释放显存

// d_A 析构，引用计数 = 0，释放显存
```

### 12.3 原始指针 vs 智能指针 使用场景

```cpp
// kernel 参数仍然使用原始指针（CUDA 要求）
template <typename T>
__global__ void kernel(T* data, int N) {
    // kernel 只能接受原始指针
}

// 安全转换：智能指针 → 原始指针
auto d_data = make_device_buffer<float>(N);
kernel<<<grid, block>>>(d_data.get(), N);  // .get() 获取原始指针
```

---

## 13. 编译、链接与构建系统

> **CUDA 相关度**：★★★★☆（分离编译、多文件项目、CMake 配置）

### 13.1 头文件与源文件分离

```cpp
// gemm_utils.h —— 声明
#pragma once

template <typename T>
void launch_gemm(const T* A, const T* B, T* C,
                 int M, int N, int K, cudaStream_t stream = 0);

// gemm_utils.cu —— 定义（模板通常在头文件，这里仅为演示分离编译概念）
#include "gemm_utils.h"

// 显式实例化：在 .cu 中生成具体版本
template void launch_gemm<float>(const float*, const float*, float*,
                                  int, int, int, cudaStream_t);
template void launch_gemm<half>(const half*, const half*, half*,
                                 int, int, int, cudaStream_t);
```

### 13.2 extern template

```cpp
// 头文件中声明：不要在此翻译单元实例化
// gemm_kernels.cuh
template <int BLOCK, int ITEMS>
__global__ void reduce_kernel(const float* in, float* out, int N);

// 避免重复实例化：告诉编译器"这个模板在别处已实例化"
extern template __global__ void reduce_kernel<256, 4>(
    const float* in, float* out, int N);

extern template __global__ void reduce_kernel<128, 8>(
    const float* in, float* out, int N);
```

### 13.3 CMake 基础配置

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 设置 CUDA 架构（非常重要——决定了编译器为哪些 SM 生成代码）
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
# SM 80 = A100,  86 = RTX 3090,  89 = RTX 4090,  90 = H100

# 构建可执行文件
add_executable(my_gemm
    src/main.cpp
    src/gemm_kernels.cu       # .cu 文件由 nvcc 编译
    src/utils.cpp              # .cpp 文件由 g++/clang++ 编译
)

# 链接 CUDA 库
target_link_libraries(my_gemm
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    cublas    # 如果使用 cuBLAS
)

# 设置 nvcc 编译选项
target_compile_options(my_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math          # 快速数学（降低精度）
        --expt-relaxed-constexpr # 放宽 constexpr 限制
        --expt-extended-lambda   # 支持 __device__ lambda
    >
)
```

### 13.4 分离编译（Separate Compilation）

```cmake
# 启用 CUDA 分离编译（允许 .cu 文件间相互调用 __device__ 函数）
set_property(TARGET my_gemm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
```

```cpp
// kernel_A.cu
__device__ float helper_func(float x) {
    return x * x + 1.0f;
}

// kernel_B.cu
// 声明来自其他编译单元的外部 device 函数
extern __device__ float helper_func(float x);

__global__ void main_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = helper_func(data[idx]);  // 跨 .cu 文件调用
    }
}
```

---

## 14. C++17/20 关键特性速览

> 推荐使用 **C++17** 作为 CUDA 开发的最低标准（nvcc 11.0+, SM 60+ 完全支持）

### 14.1 if constexpr（C++17）—— 已在前文详述

### 14.2 折叠表达式（C++17）

```cpp
// 编译期对参数包进行归约
template <typename... Args>
auto sum_all(Args... args) {
    return (... + args);  // 一元右折叠：((a1 + a2) + a3) + a4
}

template <int... Dims>
constexpr int total_elements_v = (1 * ... * Dims);  // 多维 tile 大小

static_assert(total_elements_v<128, 128, 4> == 65536);
```

### 14.3 std::optional（C++17）

```cpp
#include <optional>

// 安全地表示"可能没有值"——比返回 nullptr 更安全
std::optional<int> find_warp_index(int lane_id) {
    if (lane_id >= 0 && lane_id < 32) {
        return lane_id % 32;
    }
    return std::nullopt;  // 明确表示"不存在"
}

// 使用
auto idx = find_warp_index(threadIdx.x);
if (idx.has_value()) {
    int val = idx.value();
}
// 或直接用 value_or：
int val = find_warp_index(42).value_or(-1);
```

### 14.4 三路比较运算符（C++20）

```cpp
#include <compare>

// C++20 spaceship 运算符——自动生成全部比较运算符
struct TileSize {
    int m, n, k;
    auto operator<=>(const TileSize&) const = default;  // 自动生成 ==, !=, <, <=, >, >=

    // 自定义比较逻辑
    auto operator<=>(const TileSize& other) const {
        return m * n * k <=> other.m * other.n * other.k;  // 按面积比较
    }
};
```

### 14.5 Range-based 算法（C++20）

```cpp
#include <ranges>
#include <algorithm>

// 管道式数据处理（host side）
std::vector<float> data = {/* ... */};
auto result = data
    | std::views::filter([](float v) { return v > 0.0f; })
    | std::views::transform([](float v) { return std::sqrt(v); });
// result 是惰性视图，不分配新内存

// 类似于 Thrust 的 transform + filter 组合
```

### 14.6 designated initializers（C++20）

```cpp
// 类似 C 的结构体指定初始化器
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    cudaStream_t stream;
};

// C++20：按名称初始化，顺序需与声明一致
KernelConfig cfg = {
    .grid        = dim3(128),
    .block       = dim3(256),
    .shared_mem  = 49152,
    .stream      = 0
};
```

---

## 15. 综合练习与学习路径建议

### 15.1 知识地图

```
阶段 1：C++ 基础（2 周）
  ├── 变量、类型、控制流、函数
  ├── 命名空间、auto、范围 for
  ├── 类、struct、构造函数、析构函数
  └── enum class、初始化列表

阶段 2：核心机制（2 周）
  ├── 指针、引用、动态内存
  ├── RAII 与移动语义
  ├── 运算符重载
  └── 继承与多态

阶段 3：模板编程（3 周）← CUDA 的核心门槛
  ├── 函数模板、类模板
  ├── 模板特化
  ├── 类型萃取与 enable_if
  ├── 变参模板 + 折叠表达式
  └── CRTP 静态多态

阶段 4：编译期编程（1 周）
  ├── constexpr 变量与函数
  ├── constexpr if
  └── 编译期计算实战

阶段 5：工程化（1 周）
  ├── 智能指针
  ├── CMake + CUDA
  ├── 分离编译
  └── 内存布局与对齐
```

### 15.2 推荐练习

| 练习 | 涉及知识点 | 产出 |
|------|-----------|------|
| **1. 写一个 RAII 封装的 DeviceBuffer 类** | 类、构造/析构、移动语义、智能指针 | 可复用的显存管理组件 |
| **2. 实现一个类型无关的 ElementWise kernel** | 函数模板、`if constexpr`、`__device__` | 支持 float/half/int8 的通用 kernel |
| **3. 用 CRTP 实现不同激活函数** | CRTP、运算符重载 `operator()`、静态多态 | 了解 CUTLASS epilogue 模式 |
| **4. 写一个编译期 tile size 计算器** | `constexpr`、折叠表达式、NTTP | 自动计算 shared memory 需求的工具 |
| **5. 用变参模板实现多维 Tensor 下标** | 变参模板、折叠表达式、递归展开 | 多维索引辅助类 |
| **6. 阅读 CUTLASS 基础 GEMM 源码** | 全部 | 理解工业级模板元编程的层级结构 |

### 15.3 学习建议

1. **不要试图看完所有 C++ 再学 CUDA**——边学 CUDA 边补 C++ 更高效。碰到不认识的 C++ 语法时回到本文档查对应章节。

2. **模板是 CUDA 开发的硬门槛**。如果你在看 CUTLASS 源码时发现自己看不懂，90% 的原因是模板元编程那块没过关。花最多时间在模板上。

3. **写代码 > 看书**。每个知识点都要在 `.cu` 文件里写一遍。`constexpr` 是写出来的，不是读出来的。

4. **善用 `static_assert`**。在编译期验证你的模板参数，比运行时 debug 高效得多：
   ```cpp
   template <typename T, int BLOCK>
   void launch(T* data, int N) {
       static_assert(sizeof(T) <= 16, "Type too large for vectorized load");
       static_assert(BLOCK % 32 == 0, "Block size must be multiple of 32");
       // ...
   }
   ```

5. **现代 C++ 优先**。用 C++17，写 `if constexpr` 而非 `#ifdef`，写 `std::is_same_v` 而非 `std::is_same<>::value`，写 `using` 别名而非 `typedef`。

---

> **下一步**：完成本文档的学习后，建议直接切入
> - 《CUDA C++ Programming Guide》—— 重点读 Memory Hierarchy、Asynchronous Execution、Warp Matrix Functions
> - 《CUTLASS Documentation》—— 从 Quick Start 的 Basic GEMM 开始，逐层理解模板层级
> - TileLang/Triton —— 作为高层抽象的补充，对比理解 CUDA 底层机制
