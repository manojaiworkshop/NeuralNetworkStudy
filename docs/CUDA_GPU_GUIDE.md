# CUDA for Matrix Multiplication: CPU vs GPU

## Table of Contents
1. [CPU vs GPU Architecture](#architecture)
2. [Why GPU for Matrix Operations?](#why-gpu)
3. [CUDA Programming Model](#cuda-model)
4. [Matrix Multiplication: CPU vs GPU](#comparison)
5. [CUDA Implementation](#implementation)
6. [Setup and Installation](#setup)
7. [Performance Benchmarks](#benchmarks)

---

## 1. CPU vs GPU ARCHITECTURE {#architecture}

### CPU (Central Processing Unit)

```
CPU Architecture (Few cores, powerful each):
┌─────────────────────────────────────────────────┐
│ CPU Chip                                        │
├─────────────────────────────────────────────────┤
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐       │
│  │ Core │  │ Core │  │ Core │  │ Core │  ...  │
│  │  1   │  │  2   │  │  3   │  │  4   │       │
│  ├──────┤  ├──────┤  ├──────┤  ├──────┤       │
│  │ ALU  │  │ ALU  │  │ ALU  │  │ ALU  │       │
│  │ FPU  │  │ FPU  │  │ FPU  │  │ FPU  │       │
│  │Cache │  │Cache │  │Cache │  │Cache │       │
│  └──────┘  └──────┘  └──────┘  └──────┘       │
│                                                 │
│  Large Cache (L1, L2, L3)                      │
│  Complex Control Logic                         │
│  Branch Prediction                             │
│  Out-of-Order Execution                        │
└─────────────────────────────────────────────────┘

Characteristics:
- Few cores (4-16 typical)
- High clock speed (3-5 GHz)
- Large cache per core
- Complex instruction set
- Good for: Sequential tasks, complex logic
```

### GPU (Graphics Processing Unit)

```
GPU Architecture (Thousands of simple cores):
┌─────────────────────────────────────────────────────────┐
│ GPU Chip (NVIDIA)                                       │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐             │
│  │ SM (Streaming   │  │ SM (Streaming   │             │
│  │ Multiprocessor) │  │ Multiprocessor) │  ... (many) │
│  ├─────────────────┤  ├─────────────────┤             │
│  │ ┌─┐┌─┐┌─┐┌─┐   │  │ ┌─┐┌─┐┌─┐┌─┐   │             │
│  │ │C││C││C││C│   │  │ │C││C││C││C│   │  C = CUDA   │
│  │ │U││U││U││U│...│  │ │U││U││U││U│...│  Core       │
│  │ │D││D││D││D│   │  │ │D││D││D││D│   │             │
│  │ │A││A││A││A│   │  │ │A││A││A││A│   │             │
│  │ └─┘└─┘└─┘└─┘   │  │ └─┘└─┘└─┘└─┘   │             │
│  │ (32-128 cores)  │  │ (32-128 cores)  │             │
│  │ Shared Memory   │  │ Shared Memory   │             │
│  └─────────────────┘  └─────────────────┘             │
│                                                         │
│  Global Memory (VRAM): 8-48 GB                         │
│  Thousands of cores total                              │
└─────────────────────────────────────────────────────────┘

Characteristics:
- Thousands of cores (2000-10000+)
- Lower clock speed (1-2 GHz)
- Small cache per core
- Simple instruction set
- Good for: Parallel tasks, repetitive operations
```

### Comparison Table

| Feature | CPU | GPU |
|---------|-----|-----|
| **Cores** | 4-16 | 2,000-10,000+ |
| **Clock Speed** | 3-5 GHz | 1-2 GHz |
| **Memory** | System RAM (16-128 GB) | VRAM (8-48 GB) |
| **Cache** | Large (MB per core) | Small (KB per core) |
| **Latency** | Low (ns) | High (μs) |
| **Throughput** | Low | Very High |
| **Best For** | Sequential, Complex | Parallel, Simple |
| **Power** | 65-125W | 150-350W |

---

## 2. WHY GPU FOR MATRIX OPERATIONS? {#why-gpu}

### Matrix Multiplication is Perfectly Parallel

```
Matrix Multiplication: C = A × B

A (2x3):        B (3x2):        C (2x2):
[1 2 3]         [7 8]           [58  64]
[4 5 6]         [9 10]          [139 154]
                [11 12]

Each element of C can be computed INDEPENDENTLY:

C[0][0] = 1×7 + 2×9 + 3×11   ← Thread 1
C[0][1] = 1×8 + 2×10 + 3×12  ← Thread 2
C[1][0] = 4×7 + 5×9 + 6×11   ← Thread 3
C[1][1] = 4×8 + 5×10 + 6×12  ← Thread 4

All 4 calculations happen SIMULTANEOUSLY on GPU!
```

### CPU vs GPU Execution

#### CPU (Sequential):
```
Time: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→

Thread 1: ████████ (compute C[0][0])
Thread 2:         ████████ (compute C[0][1])
Thread 3:                 ████████ (compute C[1][0])
Thread 4:                         ████████ (compute C[1][1])

Total time: 32 units
```

#### GPU (Parallel):
```
Time: ━━━━━━━━━→

Thread 1: ████████ (compute C[0][0])
Thread 2: ████████ (compute C[0][1])
Thread 3: ████████ (compute C[1][0])
Thread 4: ████████ (compute C[1][1])
... (thousands more threads)

Total time: 8 units (4x faster!)
```

### Real-World Performance

```
Matrix Size: 1000×1000 multiplication

CPU (single core):  ~2 seconds
CPU (8 cores):      ~300 ms
GPU (CUDA):         ~5 ms      ← 400x faster!

Matrix Size: 10000×10000

CPU (single core):  ~30 minutes
CPU (8 cores):      ~4 minutes
GPU (CUDA):         ~200 ms    ← 9000x faster!
```

### Why Such Huge Speedup?

**1. Massive Parallelism:**
```
1000×1000 matrix = 1,000,000 elements to compute
GPU can launch 1,000,000 threads simultaneously!
CPU can only do 8-16 at a time
```

**2. Optimized for Math:**
```
Each GPU core has:
- Fast floating-point units
- Optimized for multiply-add (FMA)
- Hardware support for matrix ops
```

**3. High Memory Bandwidth:**
```
CPU: 50-100 GB/s
GPU: 500-1000 GB/s  ← 10x more bandwidth!
```

---

## 3. CUDA PROGRAMMING MODEL {#cuda-model}

### What is CUDA?

```
CUDA = Compute Unified Device Architecture
- Programming model for NVIDIA GPUs
- C/C++ extension
- Allows writing GPU code (kernels)
```

### CUDA Hierarchy

```
Grid (Entire GPU)
├── Block 0
│   ├── Thread (0,0)
│   ├── Thread (0,1)
│   ├── Thread (0,2)
│   └── ...
├── Block 1
│   ├── Thread (1,0)
│   ├── Thread (1,1)
│   └── ...
└── Block 2
    └── ...

Visual representation:
┌─────────────────────────────────────────────────────┐
│ Grid                                                │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│ │ Block (0,0) │ │ Block (0,1) │ │ Block (0,2) │   │
│ │ ┌─┐┌─┐┌─┐   │ │ ┌─┐┌─┐┌─┐   │ │ ┌─┐┌─┐┌─┐   │   │
│ │ │T││T││T│   │ │ │T││T││T│   │ │ │T││T││T│   │   │
│ │ └─┘└─┘└─┘   │ │ └─┘└─┘└─┘   │ │ └─┘└─┘└─┘   │   │
│ │ ┌─┐┌─┐┌─┐   │ │ ┌─┐┌─┐┌─┐   │ │ ┌─┐┌─┐┌─┐   │   │
│ │ │T││T││T│   │ │ │T││T││T│   │ │ │T││T││T│   │   │
│ │ └─┘└─┘└─┘   │ │ └─┘└─┘└─┘   │ │ └─┘└─┘└─┘   │   │
│ └─────────────┘ └─────────────┘ └─────────────┘   │
│ ┌─────────────┐ ┌─────────────┐                   │
│ │ Block (1,0) │ │ Block (1,1) │ ...               │
│ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────────────────────────┘

T = Thread (smallest unit of execution)
```

### CUDA Code Structure

```cpp
// Host (CPU) code
void cpu_function() {
    // Runs on CPU
}

// Device (GPU) code
__global__ void gpu_kernel() {
    // Runs on GPU
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// Main program
int main() {
    // 1. Allocate CPU memory
    float* h_data = new float[N];
    
    // 2. Allocate GPU memory
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // 3. Copy CPU → GPU
    cudaMemcpy(d_data, h_data, N * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 4. Launch kernel
    gpu_kernel<<<blocks, threads>>>(d_data);
    
    // 5. Copy GPU → CPU
    cudaMemcpy(h_data, d_data, N * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // 6. Free memory
    cudaFree(d_data);
    delete[] h_data;
}
```

### CUDA Keywords

| Keyword | Meaning | Runs on | Called from |
|---------|---------|---------|-------------|
| `__global__` | Kernel function | GPU | CPU |
| `__device__` | Device function | GPU | GPU |
| `__host__` | Host function | CPU | CPU |

### Thread Indexing

```cpp
__global__ void kernel() {
    // Block index
    int blockX = blockIdx.x;  // Which block (0 to gridDim.x-1)
    int blockY = blockIdx.y;
    
    // Thread index within block
    int threadX = threadIdx.x;  // Which thread (0 to blockDim.x-1)
    int threadY = threadIdx.y;
    
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
}
```

**Example:**
```
Block size: 16×16 threads
Thread (3, 2) in Block (1, 0):

Global X = blockIdx.x * blockDim.x + threadIdx.x
         = 1 * 16 + 3
         = 19

Global Y = blockIdx.y * blockDim.y + threadIdx.y
         = 0 * 16 + 2
         = 2

This thread computes C[2][19]
```

---

## 4. MATRIX MULTIPLICATION: CPU vs GPU {#comparison}

### CPU Implementation (Sequential)

```cpp
// Simple CPU matrix multiplication
void matmul_cpu(float* A, float* B, float* C, 
                int M, int N, int K) {
    // C(M×N) = A(M×K) × B(K×N)
    
    for (int i = 0; i < M; i++) {           // For each row of C
        for (int j = 0; j < N; j++) {       // For each column of C
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {   // Dot product
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

**Execution pattern:**
```
Time →
Thread 1: C[0][0] → C[0][1] → C[0][2] → ... → C[1][0] → ...
                                                           
One element at a time (sequential)
```

**Performance:**
- Time complexity: O(M × N × K)
- For 1000×1000: 1 billion operations
- Single core: ~2 seconds

### GPU Implementation (Parallel)

```cpp
// Simple GPU matrix multiplication kernel
__global__ void matmul_gpu(float* A, float* B, float* C, 
                           int M, int N, int K) {
    // Each thread computes ONE element of C
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Execution pattern:**
```
Time →
Thread 1: C[0][0]
Thread 2: C[0][1]
Thread 3: C[0][2]
Thread 4: C[0][3]
...
Thread 1M: C[999][999]

ALL elements computed simultaneously!
```

**Performance:**
- Each thread: O(K) operations
- 1000×1000 threads running in parallel
- GPU: ~5 milliseconds

### Memory Access Patterns

#### CPU:
```
Sequential access (cache-friendly):
A: [0] → [1] → [2] → [3] → ...
B: [0] → [N] → [2N] → [3N] → ...  (strided)
```

#### GPU:
```
Coalesced access (optimal):
Thread 0: A[0], B[0]
Thread 1: A[1], B[1]
Thread 2: A[2], B[2]
...
Threads 0-31 access consecutive memory
→ Single memory transaction!
```

### Optimized GPU (Shared Memory)

```cpp
__global__ void matmul_gpu_shared(float* A, float* B, float* C,
                                  int M, int N, int K) {
    // Shared memory for tile (fast, on-chip)
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = 
                A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = 
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();  // Wait for all threads to load
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Memory hierarchy:**
```
Speed:          Size:
┌────────────┐  ┌─────────┐
│ Registers  │  │ Few KB  │  ← Fastest
├────────────┤  ├─────────┤
│ Shared Mem │  │ 48 KB   │  ← Fast (on-chip)
├────────────┤  ├─────────┤
│ L1 Cache   │  │ 128 KB  │
├────────────┤  ├─────────┤
│ L2 Cache   │  │ Few MB  │
├────────────┤  ├─────────┤
│ Global Mem │  │ 8-48 GB │  ← Slow (off-chip)
└────────────┘  └─────────┘
```

### Performance Comparison

```
Matrix: 1024×1024 multiplication

┌─────────────────────┬──────────┬────────────┐
│ Method              │ Time     │ GFLOPS     │
├─────────────────────┼──────────┼────────────┤
│ CPU (1 core)        │ 2.5 s    │ 0.8        │
│ CPU (8 cores)       │ 350 ms   │ 6.1        │
│ GPU (naive)         │ 15 ms    │ 143        │
│ GPU (shared mem)    │ 5 ms     │ 429        │
│ cuBLAS (optimized)  │ 2 ms     │ 1073       │
└─────────────────────┴──────────┴────────────┘

GFLOPS = Giga Floating Point Operations Per Second
(higher is better)
```

---

## 5. CUDA IMPLEMENTATION {#implementation}

I'll create a complete CUDA implementation for your Matrix class in the next files:

### Files to Create:
1. `include/nn/matrix_cuda.h` - CUDA matrix class
2. `src/matrix_cuda.cu` - CUDA implementation
3. `example/matrix_cuda_example.cpp` - Usage example
4. `CMakeLists.txt` - Updated build file

---

## 6. SETUP AND INSTALLATION {#setup}

### Prerequisites

```bash
# Check if NVIDIA GPU is available
lspci | grep -i nvidia

# Check CUDA installation
nvcc --version

# Install CUDA toolkit (if not installed)
# Ubuntu/Debian:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-3

# Check installation
nvidia-smi  # Shows GPU info, driver version
```

### GPU Information

```bash
# Get detailed GPU info
nvidia-smi

# Output example:
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0 On  |                  N/A |
|-------------------------------+----------------------+----------------------+
```

---

## 7. PERFORMANCE BENCHMARKS {#benchmarks}

### Memory Transfer Overhead

```
Operation breakdown for 1000×1000 matrix:

CPU computation:     2000 ms
  ↓
GPU:
  CPU → GPU copy:      5 ms   ← Overhead
  GPU computation:     5 ms   ← Fast!
  GPU → CPU copy:      5 ms   ← Overhead
  Total:              15 ms   ← Still 133x faster!
```

### When to Use GPU vs CPU

**Use GPU when:**
✅ Large matrices (>256×256)
✅ Many operations (forward + backward pass)
✅ Batch processing (many matrices)
✅ Deep neural networks

**Use CPU when:**
❌ Small matrices (<64×64)
❌ Single operation
❌ Irregular computations
❌ Lots of branching/logic

### Break-even Point

```
Matrix Size vs Performance:

Size        CPU Time    GPU Time    Winner
────────────────────────────────────────────
16×16       0.01 ms     0.5 ms      CPU
64×64       0.2 ms      0.6 ms      CPU
128×128     1.5 ms      1.0 ms      GPU
256×256     10 ms       1.5 ms      GPU
512×512     80 ms       3 ms        GPU (26x)
1024×1024   600 ms      10 ms       GPU (60x)
2048×2048   5000 ms     40 ms       GPU (125x)
```

---

## Summary: Key Differences

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Philosophy** | Few fast cores | Many slow cores |
| **Parallelism** | 4-16 threads | 1000s of threads |
| **Best for** | Complex logic | Simple, repetitive math |
| **Latency** | Low | High |
| **Throughput** | Low | Very high |
| **Memory** | Large, slow access | Fast bandwidth |
| **Programming** | Easy (C++) | Medium (CUDA) |
| **Cost** | Included | Extra ($300-3000) |

**For Neural Networks:**
- Training: GPU (10-100x faster)
- Inference (small batches): CPU
- Inference (large batches): GPU
- Embedded devices: CPU/TPU

Next, I'll create the actual CUDA implementation files for your project!
