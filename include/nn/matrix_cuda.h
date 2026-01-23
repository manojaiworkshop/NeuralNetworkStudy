#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include "matrix.h"
#include <cuda_runtime.h>

/**
 * @brief CUDA-accelerated Matrix class
 * 
 * This class extends the base Matrix class with GPU acceleration
 * for matrix operations using CUDA.
 */
class MatrixCUDA : public Matrix {
private:
    float* d_data;  // Device (GPU) pointer
    bool dataOnGPU; // Flag to track if data is on GPU
    
    // Helper functions
    void allocateGPU();
    void freeGPU();
    void copyToGPU();
    void copyFromGPU();
    
public:
    // Constructors
    MatrixCUDA();
    MatrixCUDA(size_t rows, size_t cols);
    MatrixCUDA(size_t rows, size_t cols, double value);
    MatrixCUDA(const Matrix& other);
    MatrixCUDA(const MatrixCUDA& other);  // Copy constructor
    
    // Assignment operator
    MatrixCUDA& operator=(const MatrixCUDA& other);
    
    // Destructor
    ~MatrixCUDA();
    
    // CUDA-accelerated operations
    MatrixCUDA multiplyGPU(const MatrixCUDA& other) const;
    MatrixCUDA addGPU(const MatrixCUDA& other) const;
    MatrixCUDA subtractGPU(const MatrixCUDA& other) const;
    MatrixCUDA hadamardGPU(const MatrixCUDA& other) const;
    MatrixCUDA transposeGPU() const;
    MatrixCUDA applyGPU(float (*func)(float)) const;  // GPU function pointer
    
    // Memory management
    void toGPU();    // Transfer data to GPU
    void toCPU();    // Transfer data to CPU
    void forceToGPU() { dataOnGPU = false; toGPU(); }  // Force transfer even if already on GPU
    bool isOnGPU() const { return dataOnGPU; }
    
    // Device pointer access (for custom CUDA kernels)
    float* getDevicePointer() { return d_data; }
    const float* getDevicePointer() const { return d_data; }
    
    // Utility
    void printGPUInfo() const;
    static void printDeviceInfo();
};

// CUDA kernel declarations (implemented in .cu file)
// These will be called by the MatrixCUDA methods

#endif // MATRIX_CUDA_H
