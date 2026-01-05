# 🎯 CUDA Quick Reference Guide

## Essential Commands

### Building the Project
```bash
# Clean build
cd "/media/crl/Extra Disk65/PYTHON_CODE/NNFROMSCRATCH"
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)

# Run examples
./matrix_example        # CPU version
./matrix_cuda_example   # GPU version
```

### GPU Monitoring
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# GPU info
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

---

## Code Usage Examples

### Basic Matrix Operations (GPU)

```cpp
#include "nn/matrix_cuda.h"

// Create matrices
MatrixCUDA A(1000, 1000);
MatrixCUDA B(1000, 1000);
A.randomize(0.0, 1.0);
B.randomize(0.0, 1.0);

// GPU operations (automatic memory management)
MatrixCUDA C = A.multiplyGPU(B);      // Matrix multiplication
MatrixCUDA D = A.addGPU(B);           // Addition
MatrixCUDA E = A.subtractGPU(B);      // Subtraction
MatrixCUDA F = A.hadamardGPU(B);      // Element-wise multiply
MatrixCUDA G = A.transposeGPU();      // Transpose

// Print results
C.print();
```

### Manual Memory Control

```cpp
MatrixCUDA A(1000, 1000);
A.randomize(0.0, 1.0);

// Transfer to GPU once
A.toGPU();

// Perform multiple operations (data stays on GPU)
MatrixCUDA B = A.multiplyGPU(A);
MatrixCUDA C = B.addGPU(A);
MatrixCUDA D = C.transposeGPU();

// Transfer back when done
D.toCPU();
```

### Neural Network Layer Example

```cpp
class NeuralLayer {
private:
    MatrixCUDA weights;
    MatrixCUDA biases;
    
public:
    NeuralLayer(int input_size, int output_size) {
        weights = MatrixCUDA(output_size, input_size);
        biases = MatrixCUDA(output_size, 1);
        
        // Initialize
        weights.randomize(-0.5, 0.5);
        biases.randomize(0.0, 0.0);
        
        // Keep on GPU
        weights.toGPU();
        biases.toGPU();
    }
    
    MatrixCUDA forward(const MatrixCUDA& input) {
        // Z = W × X + b
        MatrixCUDA z = weights.multiplyGPU(input);
        z = z.addGPU(biases);
        return z;
    }
};
```

---

## Performance Tips

### ✅ DO
- Use batch processing (process multiple samples at once)
- Keep data on GPU between operations
- Use large matrices (> 256×256) for GPU
- Profile your code to find bottlenecks
- Use shared memory for frequently accessed data

### ❌ DON'T
- Transfer data CPU ↔ GPU for every operation
- Use GPU for tiny matrices (< 128×128)
- Allocate/deallocate GPU memory repeatedly
- Forget to check for CUDA errors
- Mix CPU and GPU operations unnecessarily

---

## Debugging

### CUDA Error Checking

```cpp
// Add after CUDA calls
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}
```

### Common Issues

1. **Out of Memory**
   ```
   Solution: Reduce batch size or matrix dimensions
   Check: nvidia-smi for available memory
   ```

2. **Compilation Errors**
   ```
   Solution: Make sure CUDA 12.9 is selected
   Check: which nvcc && nvcc --version
   ```

3. **Slow Performance**
   ```
   Solution: Check matrix size (should be > 256)
   Check: Profile with nvprof or Nsight
   ```

---

## Memory Layout

### CPU (Stack)
```
Matrix mat(3, 3);
Size: 40 bytes
  ├─ rows: 8 bytes
  ├─ cols: 8 bytes  
  └─ data: 24 bytes (vector structure)
```

### GPU (Device Memory)
```
MatrixCUDA mat(1000, 1000);
CPU: 40 bytes (structure)
GPU: 8,000,000 bytes (actual data)
  ├─ Allocated: cudaMalloc()
  ├─ Transferred: cudaMemcpy()
  └─ Freed: cudaFree()
```

---

## Performance Metrics

### GFLOPS Calculation
```
Matrix Multiplication: C = A × B
  A: M × K
  B: K × N
  C: M × N

Operations: 2 × M × N × K (multiply + add)
GFLOPS = Operations / (Time in seconds × 10^9)

Example (1024×1024):
  Ops = 2 × 1024 × 1024 × 1024 = 2,147,483,648
  Time = 0.04466 seconds
  GFLOPS = 2,147,483,648 / 44,660,000 = 48.08 GFLOPS
```

### Speedup Calculation
```
Speedup = CPU Time / GPU Time

Example:
  CPU: 13.52 seconds
  GPU: 0.04466 seconds
  Speedup = 13.52 / 0.04466 = 302.8x
```

---

## Project Structure

```
NNFROMSCRATCH/
├── CMakeLists.txt              # Build configuration with CUDA
├── include/
│   └── nn/
│       ├── matrix.h            # CPU Matrix class
│       └── matrix_cuda.h       # GPU Matrix class
├── src/
│   ├── matrix.cpp              # CPU implementation
│   └── matrix_cuda.cu          # CUDA kernels
├── example/
│   ├── matrix_example.cpp      # CPU demo
│   └── matrix_cuda_example.cpp # GPU benchmark
├── docs/
│   ├── CMAKE_COMPLETE_GUIDE.md
│   ├── CODE_EXPLANATION_COMPLETE.md
│   ├── CUDA_GPU_GUIDE.md
│   ├── CUDA_RESULTS.md         # Benchmark results
│   └── CUDA_QUICK_REFERENCE.md # This file
└── build/                      # Build directory
    ├── matrix_example          # CPU executable
    └── matrix_cuda_example     # GPU executable
```

---

## CMakeLists.txt Key Settings

```cmake
# Enable CUDA
project(NeuralNetworkFromScratch VERSION 1.0 LANGUAGES CXX CUDA)

# Set CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)

# Set architecture (Quadro RTX 5000 = sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Compile CUDA library
add_library(matrix_cuda_lib STATIC src/matrix_cuda.cu)

# Link to executable
target_link_libraries(matrix_cuda_example 
    matrix_lib 
    matrix_cuda_lib
)
```

---

## Benchmarking Template

```cpp
#include <chrono>
using namespace std::chrono;

auto start = high_resolution_clock::now();

// Your GPU operation here
MatrixCUDA result = A.multiplyGPU(B);

auto end = high_resolution_clock::now();
auto duration = duration_cast<milliseconds>(end - start);
cout << "Time: " << duration.count() << " ms" << endl;
```

---

## GPU Specifications

**Your Hardware: NVIDIA Quadro RTX 5000 with Max-Q**

| Specification | Value |
|--------------|-------|
| CUDA Cores | ~3072 |
| Tensor Cores | 384 |
| RT Cores | 48 |
| Memory | 16 GB GDDR6 |
| Memory Bandwidth | 384 GB/s |
| Compute Capability | 7.5 |
| Architecture | Turing |
| Max Threads/Block | 1024 |
| Shared Memory/Block | 48 KB |
| Multiprocessors | 48 |

---

## Useful CUDA Functions

### Memory Management
```cpp
cudaMalloc(&d_ptr, size);           // Allocate GPU memory
cudaMemcpy(dst, src, size, kind);   // Copy data
cudaFree(d_ptr);                    // Free GPU memory
cudaMemset(d_ptr, value, size);     // Set memory
```

### Device Functions
```cpp
cudaGetDeviceCount(&count);         // Number of GPUs
cudaGetDeviceProperties(&prop, 0);  // GPU properties
cudaSetDevice(0);                   // Select GPU
```

### Synchronization
```cpp
cudaDeviceSynchronize();            // Wait for GPU
__syncthreads();                    // Block-level sync (in kernel)
```

### Error Checking
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

---

## Next Steps for Neural Networks

1. **Implement Activation Functions**
   ```cpp
   __global__ void relu_kernel(float* data, int size);
   __global__ void sigmoid_kernel(float* data, int size);
   ```

2. **Add Loss Functions**
   ```cpp
   __global__ void mse_kernel(float* pred, float* target, int size);
   __global__ void cross_entropy_kernel(float* pred, float* target, int size);
   ```

3. **Implement Backpropagation**
   ```cpp
   MatrixCUDA gradient = output.subtractGPU(target);
   MatrixCUDA delta = gradient.hadamardGPU(activation_derivative);
   ```

4. **Add Optimizers**
   ```cpp
   // SGD with momentum on GPU
   weights = weights.subtractGPU(gradient.scalarMultiply(learning_rate));
   ```

---

## Resources

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **cuBLAS Library**: https://docs.nvidia.com/cuda/cublas/
- **cuDNN Documentation**: https://docs.nvidia.com/deeplearning/cudnn/
- **NVIDIA Developer Blog**: https://developer.nvidia.com/blog/

---

*Last Updated: Based on successful compilation and 287x speedup results*
