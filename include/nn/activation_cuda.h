#ifndef ACTIVATION_CUDA_H
#define ACTIVATION_CUDA_H

#include "matrix_cuda.h"
#include "activation.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>

/**
 * @brief CUDA-accelerated activation functions using CUDA
 * 
 * These classes extend CPU activation functions with CUDA implementations
 * for significantly faster computation on large matrices.
 */

// ==================== BASE CUDA ACTIVATION ====================

/**
 * @brief Base class for CUDA-accelerated activation functions
 */
class ActivationCUDA {
public:
    virtual ~ActivationCUDA() = default;
    
    /**
     * @brief Forward pass through activation function on CUDA
     * @param input Input matrix (on CPU or CUDA)
     * @return Activated output on CUDA
     */
    virtual MatrixCUDA forward(const MatrixCUDA& input) const = 0;
    
    /**
     * @brief Backward pass (compute gradient) on CUDA
     * @param input Original input to forward pass
     * @param output_gradient Gradient from next layer
     * @return Gradient with respect to input
     */
    virtual MatrixCUDA backward(const MatrixCUDA& input, 
                                const MatrixCUDA& output_gradient) const = 0;
    
    /**
     * @brief Get name of activation function
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Clone the activation function
     */
    virtual std::unique_ptr<ActivationCUDA> clone() const = 0;
};

// ==================== SIGMOID CUDA ====================

/**
 * @brief CUDA-accelerated Sigmoid activation
 * σ(x) = 1 / (1 + e^(-x))
 */
class SigmoidCUDA : public ActivationCUDA {
public:
    MatrixCUDA forward(const MatrixCUDA& input) const override;
    MatrixCUDA backward(const MatrixCUDA& input, 
                       const MatrixCUDA& output_gradient) const override;
    std::string getName() const override { return "SigmoidCUDA"; }
    std::unique_ptr<ActivationCUDA> clone() const override;
};

// ==================== RELU CUDA ====================

/**
 * @brief CUDA-accelerated ReLU activation
 * ReLU(x) = max(0, x)
 */
class ReLUCUDA : public ActivationCUDA {
public:
    MatrixCUDA forward(const MatrixCUDA& input) const override;
    MatrixCUDA backward(const MatrixCUDA& input, 
                       const MatrixCUDA& output_gradient) const override;
    std::string getName() const override { return "ReLUCUDA"; }
    std::unique_ptr<ActivationCUDA> clone() const override;
};

// ==================== TANH CUDA ====================

/**
 * @brief CUDA-accelerated Tanh activation
 * tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
class TanhCUDA : public ActivationCUDA {
public:
    MatrixCUDA forward(const MatrixCUDA& input) const override;
    MatrixCUDA backward(const MatrixCUDA& input, 
                       const MatrixCUDA& output_gradient) const override;
    std::string getName() const override { return "TanhCUDA"; }
    std::unique_ptr<ActivationCUDA> clone() const override;
};

// ==================== LEAKY RELU CUDA ====================

/**
 * @brief CUDA-accelerated Leaky ReLU activation
 * LeakyReLU(x) = x if x > 0, else alpha * x
 */
class LeakyReLUCUDA : public ActivationCUDA {
private:
    float alpha;
    
public:
    explicit LeakyReLUCUDA(float alpha = 0.01f) : alpha(alpha) {}
    
    MatrixCUDA forward(const MatrixCUDA& input) const override;
    MatrixCUDA backward(const MatrixCUDA& input, 
                       const MatrixCUDA& output_gradient) const override;
    std::string getName() const override { return "LeakyReLUCUDA"; }
    std::unique_ptr<ActivationCUDA> clone() const override;
    
    float getAlpha() const { return alpha; }
};

// ==================== ELU CUDA ====================

/**
 * @brief CUDA-accelerated ELU (Exponential Linear Unit) activation
 * ELU(x) = x if x > 0, else alpha * (e^x - 1)
 */
class ELUCUDA : public ActivationCUDA {
private:
    float alpha;
    
public:
    explicit ELUCUDA(float alpha = 1.0f) : alpha(alpha) {}
    
    MatrixCUDA forward(const MatrixCUDA& input) const override;
    MatrixCUDA backward(const MatrixCUDA& input, 
                       const MatrixCUDA& output_gradient) const override;
    std::string getName() const override { return "ELUCUDA"; }
    std::unique_ptr<ActivationCUDA> clone() const override;
    
    float getAlpha() const { return alpha; }
};

// ==================== CUDA KERNEL DECLARATIONS ====================

// Device functions for activation computations
__device__ float sigmoid_device(float x);
__device__ float sigmoid_derivative_device(float x);
__device__ float relu_device(float x);
__device__ float relu_derivative_device(float x);
__device__ float tanh_derivative_device(float x);
__device__ float leaky_relu_device(float x, float alpha);
__device__ float leaky_relu_derivative_device(float x, float alpha);
__device__ float elu_device(float x, float alpha);
__device__ float elu_derivative_device(float x, float alpha);

// Kernel declarations for forward passes
__global__ void sigmoid_forward_kernel(const float* input, float* output, int size);
__global__ void relu_forward_kernel(const float* input, float* output, int size);
__global__ void tanh_forward_kernel(const float* input, float* output, int size);
__global__ void leaky_relu_forward_kernel(const float* input, float* output, int size, float alpha);
__global__ void elu_forward_kernel(const float* input, float* output, int size, float alpha);

// Kernel declarations for backward passes
__global__ void sigmoid_backward_kernel(const float* input, const float* grad_output, 
                                       float* grad_input, int size);
__global__ void relu_backward_kernel(const float* input, const float* grad_output, 
                                    float* grad_input, int size);
__global__ void tanh_backward_kernel(const float* input, const float* grad_output, 
                                    float* grad_input, int size);
__global__ void leaky_relu_backward_kernel(const float* input, const float* grad_output, 
                                          float* grad_input, int size, float alpha);
__global__ void elu_backward_kernel(const float* input, const float* grad_output, 
                                   float* grad_input, int size, float alpha);

#endif // ACTIVATION_CUDA_H
