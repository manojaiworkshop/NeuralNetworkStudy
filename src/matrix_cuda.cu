#include "nn/matrix_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// CUDA KERNELS (Run on GPU)
// ============================================================================

/**
 * Matrix multiplication kernel (naive implementation)
 * Each thread computes one element of the result matrix
 */
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        // Dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Matrix multiplication kernel with shared memory (optimized)
 * Uses tiling to reduce global memory access
 */
#define TILE_SIZE 16

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // Shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to make sure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Element-wise addition kernel
 */
__global__ void add_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * Element-wise subtraction kernel
 */
__global__ void subtract_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

/**
 * Hadamard product (element-wise multiplication) kernel
 */
__global__ void hadamard_kernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

/**
 * Transpose kernel
 */
__global__ void transpose_kernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

/**
 * Apply function kernel (e.g., ReLU, Sigmoid)
 */
__device__ float relu_device(float x) {
    return fmaxf(0.0f, x);
}

__device__ float sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanh_device(float x) {
    return tanhf(x);
}

__global__ void apply_kernel(const float* input, float* output, int size,
                            float (*func)(float)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = func(input[idx]);
    }
}

// ============================================================================
// MatrixCUDA CLASS IMPLEMENTATION
// ============================================================================

// Constructor: default
MatrixCUDA::MatrixCUDA() : Matrix(), d_data(nullptr), dataOnGPU(false) {}

// Constructor: with dimensions
MatrixCUDA::MatrixCUDA(size_t rows, size_t cols) 
    : Matrix(rows, cols), d_data(nullptr), dataOnGPU(false) {}

// Constructor: with dimensions and value
MatrixCUDA::MatrixCUDA(size_t rows, size_t cols, double value)
    : Matrix(rows, cols, value), d_data(nullptr), dataOnGPU(false) {}

// Constructor: from base Matrix
MatrixCUDA::MatrixCUDA(const Matrix& other)
    : Matrix(other), d_data(nullptr), dataOnGPU(false) {}

// Destructor
MatrixCUDA::~MatrixCUDA() {
    freeGPU();
}

// Allocate GPU memory
void MatrixCUDA::allocateGPU() {
    if (d_data == nullptr) {
        size_t size = getRows() * getCols() * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_data, size));
    }
}

// Free GPU memory
void MatrixCUDA::freeGPU() {
    if (d_data != nullptr) {
        CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
        dataOnGPU = false;
    }
}

// Copy data from CPU to GPU
void MatrixCUDA::copyToGPU() {
    allocateGPU();
    
    // Convert double to float and flatten matrix
    size_t size = getRows() * getCols();
    float* h_temp = new float[size];
    
    for (size_t i = 0; i < getRows(); i++) {
        for (size_t j = 0; j < getCols(); j++) {
            h_temp[i * getCols() + j] = static_cast<float>(get(i, j));
        }
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_temp, size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    delete[] h_temp;
    dataOnGPU = true;
}

// Copy data from GPU to CPU
void MatrixCUDA::copyFromGPU() {
    if (!dataOnGPU || d_data == nullptr) {
        std::cerr << "No data on GPU to copy!" << std::endl;
        return;
    }
    
    size_t size = getRows() * getCols();
    float* h_temp = new float[size];
    
    // Copy from GPU
    CUDA_CHECK(cudaMemcpy(h_temp, d_data, size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Convert float to double and unflatten
    for (size_t i = 0; i < getRows(); i++) {
        for (size_t j = 0; j < getCols(); j++) {
            set(i, j, static_cast<double>(h_temp[i * getCols() + j]));
        }
    }
    
    delete[] h_temp;
}

// Transfer data to GPU
void MatrixCUDA::toGPU() {
    if (!dataOnGPU) {
        copyToGPU();
    }
}

// Transfer data to CPU
void MatrixCUDA::toCPU() {
    if (dataOnGPU) {
        copyFromGPU();
    }
}

// GPU Matrix Multiplication
MatrixCUDA MatrixCUDA::multiplyGPU(const MatrixCUDA& other) const {
    if (getCols() != other.getRows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    // Ensure data is on GPU
    const_cast<MatrixCUDA*>(this)->toGPU();
    const_cast<MatrixCUDA&>(other).toGPU();
    
    // Create result matrix
    MatrixCUDA result(getRows(), other.getCols());
    result.allocateGPU();
    
    // Set up grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((other.getCols() + TILE_SIZE - 1) / TILE_SIZE,
                  (getRows() + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel (using optimized shared memory version)
    matmul_shared_kernel<<<gridSize, blockSize>>>(
        d_data, other.d_data, result.d_data,
        getRows(), other.getCols(), getCols()
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// GPU Matrix Addition
MatrixCUDA MatrixCUDA::addGPU(const MatrixCUDA& other) const {
    if (!sameShape(other)) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    const_cast<MatrixCUDA*>(this)->toGPU();
    const_cast<MatrixCUDA&>(other).toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    add_kernel<<<gridSize, blockSize>>>(d_data, other.d_data, result.d_data, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// GPU Matrix Subtraction
MatrixCUDA MatrixCUDA::subtractGPU(const MatrixCUDA& other) const {
    if (!sameShape(other)) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    const_cast<MatrixCUDA*>(this)->toGPU();
    const_cast<MatrixCUDA&>(other).toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    subtract_kernel<<<gridSize, blockSize>>>(d_data, other.d_data, result.d_data, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// GPU Hadamard Product
MatrixCUDA MatrixCUDA::hadamardGPU(const MatrixCUDA& other) const {
    if (!sameShape(other)) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    const_cast<MatrixCUDA*>(this)->toGPU();
    const_cast<MatrixCUDA&>(other).toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    hadamard_kernel<<<gridSize, blockSize>>>(d_data, other.d_data, result.d_data, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// GPU Transpose
MatrixCUDA MatrixCUDA::transposeGPU() const {
    const_cast<MatrixCUDA*>(this)->toGPU();
    
    MatrixCUDA result(getCols(), getRows());
    result.allocateGPU();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((getCols() + 15) / 16, (getRows() + 15) / 16);
    
    transpose_kernel<<<gridSize, blockSize>>>(
        d_data, result.d_data, getRows(), getCols()
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// Print GPU info
void MatrixCUDA::printGPUInfo() const {
    std::cout << "Matrix on GPU: " << (dataOnGPU ? "Yes" : "No") << std::endl;
    if (dataOnGPU) {
        std::cout << "GPU memory used: " 
                  << (getRows() * getCols() * sizeof(float)) / (1024.0 * 1024.0) 
                  << " MB" << std::endl;
    }
}

// Print device information
void MatrixCUDA::printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "\n=== CUDA Device Information ===" << std::endl;
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores: ~" << prop.multiProcessorCount * 64 << " (estimated)" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " 
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
                  << " GB/s" << std::endl;
    }
    std::cout << "==============================\n" << std::endl;
}
