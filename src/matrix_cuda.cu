#include "nn/matrix_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <climits>

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

// Copy constructor
MatrixCUDA::MatrixCUDA(const MatrixCUDA& other)
    : Matrix(other), d_data(nullptr), dataOnGPU(false) {
    // If other has GPU data, copy it
    if (other.dataOnGPU && other.d_data != nullptr) {
        allocateGPU();
        size_t size = getRows() * getCols() * sizeof(float);
        cudaError_t err = cudaMemcpy(d_data, other.d_data, size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed in copy constructor: ") + cudaGetErrorString(err));
        }
        dataOnGPU = true;
    }
}

// Copy assignment operator
MatrixCUDA& MatrixCUDA::operator=(const MatrixCUDA& other) {
    if (this != &other) {
        // Free existing GPU memory
        freeGPU();
        
        // Copy CPU data (base class assignment)
        Matrix::operator=(other);
        
        // Copy GPU data if it exists
        if (other.dataOnGPU && other.d_data != nullptr) {
            allocateGPU();
            size_t size = getRows() * getCols() * sizeof(float);
            CUDA_CHECK(cudaMemcpy(d_data, other.d_data, size, cudaMemcpyDeviceToDevice));
            dataOnGPU = true;
        }
    }
    return *this;
}

// Destructor
MatrixCUDA::~MatrixCUDA() {
    freeGPU();
}

// Allocate GPU memory
void MatrixCUDA::allocateGPU() {
    if (d_data == nullptr) {
        size_t size = getRows() * getCols() * sizeof(float);
        
        // Clear any previous errors
        cudaGetLastError();
        
        cudaError_t err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for " << getRows() << "x" << getCols() 
                      << " (" << size << " bytes): " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
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
    if (getRows() == 0 || getCols() == 0) {
        throw std::runtime_error("Cannot copy empty matrix to GPU");
    }
    
    // Clear any previous errors before allocation
    cudaGetLastError();
    
    allocateGPU();
    
    if (d_data == nullptr) {
        throw std::runtime_error("Failed to allocate GPU memory in copyToGPU");
    }
    
    // Convert double to float and flatten matrix
    size_t size = getRows() * getCols();
    float* h_temp = new float[size];
    
    for (size_t i = 0; i < getRows(); i++) {
        for (size_t j = 0; j < getCols(); j++) {
            h_temp[i * getCols() + j] = static_cast<float>(get(i, j));
        }
    }
    
    // Clear any previous errors before copy
    cudaGetLastError();
    
    // Copy to GPU
    cudaError_t err = cudaMemcpy(d_data, h_temp, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        delete[] h_temp;
        std::cerr << "cudaMemcpy H2D failed for " << getRows() << "x" << getCols() 
                  << " (" << size * sizeof(float) << " bytes): " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
    }
    
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
    
    // Check for zero-sized matrices
    if (getRows() == 0 || getCols() == 0 || other.getCols() == 0) {
        throw std::runtime_error("Cannot multiply zero-sized matrices on GPU");
    }
    
    // Check for overflow when casting to int
    if (getRows() > INT_MAX || getCols() > INT_MAX || other.getCols() > INT_MAX) {
        throw std::runtime_error("Matrix dimensions too large for CUDA kernel (> INT_MAX)");
    }
    
    // Create mutable copies if needed
    MatrixCUDA A_copy = *this;
    MatrixCUDA B_copy = other;
    
    // Ensure data is on GPU
    A_copy.toGPU();
    
    // Check for errors after A copy
    cudaError_t error_A = cudaGetLastError();
    if (error_A != cudaSuccess) {
        std::cerr << "CUDA error after copying A to GPU: " << cudaGetErrorString(error_A) << std::endl;
        throw std::runtime_error(std::string("Failed to copy A to GPU: ") + cudaGetErrorString(error_A));
    }
    
    B_copy.toGPU();
    
    // Check for errors after B copy
    cudaError_t error_B = cudaGetLastError();
    if (error_B != cudaSuccess) {
        std::cerr << "CUDA error after copying B to GPU: " << cudaGetErrorString(error_B) << std::endl;
        throw std::runtime_error(std::string("Failed to copy B to GPU: ") + cudaGetErrorString(error_B));
    }
    
    // Extra validation
    if (!A_copy.dataOnGPU || !B_copy.dataOnGPU) {
        throw std::runtime_error("Failed to transfer matrices to GPU before multiplication");
    }
    
    // Validate GPU pointers
    if (A_copy.d_data == nullptr || B_copy.d_data == nullptr) {
        throw std::runtime_error("GPU data pointer is null in multiplyGPU");
    }
    
    // Create result matrix
    MatrixCUDA result(getRows(), other.getCols());
    
    // Check for errors after creating result
    cudaError_t error_result_create = cudaGetLastError();
    if (error_result_create != cudaSuccess) {
        std::cerr << "CUDA error after creating result matrix: " << cudaGetErrorString(error_result_create) << std::endl;
        throw std::runtime_error(std::string("Failed to create result matrix: ") + cudaGetErrorString(error_result_create));
    }
    
    result.allocateGPU();
    
    // Check for errors after allocating result
    cudaError_t error_result_alloc = cudaGetLastError();
    if (error_result_alloc != cudaSuccess) {
        std::cerr << "CUDA error after allocating result on GPU: " << cudaGetErrorString(error_result_alloc) << std::endl;
        std::cerr << "  Trying to allocate: " << result.getRows() << "x" << result.getCols() 
                  << " = " << (result.getRows() * result.getCols() * sizeof(float)) << " bytes" << std::endl;
        throw std::runtime_error(std::string("Failed to allocate result on GPU: ") + cudaGetErrorString(error_result_alloc));
    }
    
    if (result.d_data == nullptr) {
        throw std::runtime_error("Failed to allocate GPU memory for result");
    }
    
    // Check for any errors from allocations
    cudaError_t alloc_error = cudaGetLastError();
    if (alloc_error != cudaSuccess) {
        std::cerr << "CUDA error after allocations: " << cudaGetErrorString(alloc_error) << std::endl;
        throw std::runtime_error(std::string("CUDA error before kernel: ") + cudaGetErrorString(alloc_error));
    }
    
    // Set up grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((other.getCols() + 15) / 16,
                  (getRows() + 15) / 16);
    
    // Validate dimensions for CUDA
    if (gridSize.x == 0 || gridSize.y == 0) {
        throw std::runtime_error("Invalid grid dimensions in multiplyGPU");
    }
    
    // Validate grid/block configuration
    if (blockSize.x * blockSize.y > 1024) {
        throw std::runtime_error("Block size exceeds maximum threads per block");
    }
    
    // Debug output for kernel launch parameters
    #ifdef DEBUG_MATMUL
    std::cout << "Launching matmul_kernel:" << std::endl;
    std::cout << "  A: " << getRows() << "x" << getCols() << " @ " << A_copy.d_data << std::endl;
    std::cout << "  B: " << other.getRows() << "x" << other.getCols() << " @ " << B_copy.d_data << std::endl;
    std::cout << "  C: " << result.getRows() << "x" << result.getCols() << " @ " << result.d_data << std::endl;
    std::cout << "  Grid: (" << gridSize.x << ", " << gridSize.y << "), Block: (" 
              << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    std::cout << "  M=" << getRows() << ", N=" << other.getCols() << ", K=" << getCols() << std::endl;
    #endif
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(
        A_copy.d_data, B_copy.d_data, result.d_data,
        static_cast<int>(getRows()), 
        static_cast<int>(other.getCols()), 
        static_cast<int>(getCols())
    );
    
    // Check for errors
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        std::cerr << "CUDA matmul error: A(" << getRows() << "x" << getCols() 
                  << ") * B(" << other.getRows() << "x" << other.getCols() << ")"
                  << " - " << cudaGetErrorString(kernel_error) << std::endl;
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(kernel_error));
    }
    
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
    
    MatrixCUDA A_copy = *this;
    MatrixCUDA B_copy = other;
    A_copy.toGPU();
    B_copy.toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    add_kernel<<<gridSize, blockSize>>>(A_copy.d_data, B_copy.d_data, result.d_data, size);
    
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
    
    MatrixCUDA A_copy = *this;
    MatrixCUDA B_copy = other;
    A_copy.toGPU();
    B_copy.toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    subtract_kernel<<<gridSize, blockSize>>>(A_copy.d_data, B_copy.d_data, result.d_data, size);
    
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
    
    MatrixCUDA A_copy = *this;
    MatrixCUDA B_copy = other;
    A_copy.toGPU();
    B_copy.toGPU();
    
    MatrixCUDA result(getRows(), getCols());
    result.allocateGPU();
    
    int size = getRows() * getCols();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    hadamard_kernel<<<gridSize, blockSize>>>(A_copy.d_data, B_copy.d_data, result.d_data, size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    result.dataOnGPU = true;
    result.copyFromGPU();
    
    return result;
}

// GPU Transpose
MatrixCUDA MatrixCUDA::transposeGPU() const {
    MatrixCUDA A_copy = *this;
    A_copy.toGPU();
    
    MatrixCUDA result(getCols(), getRows());
    result.allocateGPU();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((getCols() + 15) / 16, (getRows() + 15) / 16);
    
    transpose_kernel<<<gridSize, blockSize>>>(
        A_copy.d_data, result.d_data, getRows(), getCols()
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
