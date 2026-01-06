#include "nn/activation_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// ==================== DEVICE FUNCTIONS ====================

/**
 * @brief Sigmoid function on device
 */
__device__ float activation_sigmoid_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * @brief Sigmoid derivative on device
 * σ'(x) = σ(x) * (1 - σ(x))
 */
__device__ float sigmoid_derivative_device(float x) {
    float sig = activation_sigmoid_device(x);
    return sig * (1.0f - sig);
}

/**
 * @brief ReLU function on device
 */
__device__ float activation_ractivation_elu_device(float x) {
    return fmaxf(0.0f, x);
}

/**
 * @brief ReLU derivative on device
 */
__device__ float relu_derivative_device(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

/**
 * @brief Tanh derivative on device
 * tanh'(x) = 1 - tanh²(x)
 */
__device__ float activation_tanh_derivative_device(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

/**
 * @brief Leaky ReLU function on device
 */
__device__ float leaky_activation_ractivation_elu_device(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

/**
 * @brief Leaky ReLU derivative on device
 */
__device__ float leaky_relu_derivative_device(float x, float alpha) {
    return (x > 0.0f) ? 1.0f : alpha;
}

/**
 * @brief ELU function on device
 */
__device__ float activation_elu_device(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
}

/**
 * @brief ELU derivative on device
 */
__device__ float elu_derivative_device(float x, float alpha) {
    return (x > 0.0f) ? 1.0f : alpha * expf(x);
}

// ==================== FORWARD KERNELS ====================

/**
 * @brief Sigmoid forward pass kernel
 */
__global__ void sigmoid_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = activation_sigmoid_device(input[idx]);
    }
}

/**
 * @brief ReLU forward pass kernel
 */
__global__ void relu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = activation_ractivation_elu_device(input[idx]);
    }
}

/**
 * @brief Tanh forward pass kernel
 */
__global__ void tanh_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

/**
 * @brief Leaky ReLU forward pass kernel
 */
__global__ void leaky_relu_forward_kernel(const float* input, float* output, 
                                         int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = leaky_activation_ractivation_elu_device(input[idx], alpha);
    }
}

/**
 * @brief ELU forward pass kernel
 */
__global__ void elu_forward_kernel(const float* input, float* output, 
                                   int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = activation_elu_device(input[idx], alpha);
    }
}

// ==================== BACKWARD KERNELS ====================

/**
 * @brief Sigmoid backward pass kernel
 */
__global__ void sigmoid_backward_kernel(const float* input, const float* grad_output, 
                                       float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * sigmoid_derivative_device(input[idx]);
    }
}

/**
 * @brief ReLU backward pass kernel
 */
__global__ void relu_backward_kernel(const float* input, const float* grad_output, 
                                    float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * relu_derivative_device(input[idx]);
    }
}

/**
 * @brief Tanh backward pass kernel
 */
__global__ void tanh_backward_kernel(const float* input, const float* grad_output, 
                                    float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * activation_tanh_derivative_device(input[idx]);
    }
}

/**
 * @brief Leaky ReLU backward pass kernel
 */
__global__ void leaky_relu_backward_kernel(const float* input, const float* grad_output, 
                                          float* grad_input, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * leaky_relu_derivative_device(input[idx], alpha);
    }
}

/**
 * @brief ELU backward pass kernel
 */
__global__ void elu_backward_kernel(const float* input, const float* grad_output, 
                                   float* grad_input, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * elu_derivative_device(input[idx], alpha);
    }
}

// ==================== HELPER FUNCTIONS ====================

/**
 * @brief Launch configuration helper
 */
void getLaunchConfig(int size, int& blocks, int& threads) {
    threads = 256;  // Common block size
    blocks = (size + threads - 1) / threads;
}

// ==================== SIGMOID CUDA IMPLEMENTATION ====================

MatrixCUDA SigmoidCUDA::forward(const MatrixCUDA& input) const {
    int size = input.getRows() * input.getCols();
    int blocks, threads;
    getLaunchConfig(size, blocks, threads);
    
    // Create output matrix
    MatrixCUDA result(input.getRows(), input.getCols());
    result.toGPU();
    
    // Ensure input is on GPU
    MatrixCUDA input_gpu = input;
    if (!input_gpu.isOnGPU()) {
        input_gpu.toGPU();
    }
    
    // Launch kernel directly on GPU memory
    sigmoid_forward_kernel<<<blocks, threads>>>(
        input_gpu.getDevicePointer(),
        result.getDevicePointer(),
        size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in SigmoidCUDA::forward: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    return result;
}

MatrixCUDA SigmoidCUDA::backward(const MatrixCUDA& input, 
                               const MatrixCUDA& output_gradient) const {
    int size = input.getRows() * input.getCols();
    int blocks, threads;
    getLaunchConfig(size, blocks, threads);
    
    // Create output matrix
    MatrixCUDA grad_input(input.getRows(), input.getCols());
    grad_input.toGPU();
    
    // Ensure inputs are on GPU
    MatrixCUDA input_gpu = input;
    MatrixCUDA grad_out_gpu = output_gradient;
    if (!input_gpu.isOnGPU()) input_gpu.toGPU();
    if (!grad_out_gpu.isOnGPU()) grad_out_gpu.toGPU();
    
    // Launch backward kernel
    sigmoid_backward_kernel<<<blocks, threads>>>(
        input_gpu.getDevicePointer(),
        grad_out_gpu.getDevicePointer(),
        grad_input.getDevicePointer(),
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in SigmoidCUDA::backward: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    return grad_input;
}

std::unique_ptr<ActivationCUDA> SigmoidCUDA::clone() const {
    return std::make_unique<SigmoidCUDA>();
}

// ==================== RELU CUDA IMPLEMENTATION ====================

MatrixCUDA ReLUCUDA::forward(const MatrixCUDA& input) const {
    int size = input.getRows() * input.getCols();
    int blocks, threads;
    getLaunchConfig(size, blocks, threads);
    
    MatrixCUDA output(input.getRows(), input.getCols());
    output.toGPU();
    
    MatrixCUDA input_gpu = input;
    if (!input_gpu.isOnGPU()) input_gpu.toGPU();
    
    relu_forward_kernel<<<blocks, threads>>>(
        input_gpu.getDevicePointer(),
        output.getDevicePointer(),
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ReLUCUDA::forward: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    return output;
}

MatrixCUDA ReLUCUDA::backward(const MatrixCUDA& input, 
                            const MatrixCUDA& output_gradient) const {
    int size = input.getRows() * input.getCols();
    int blocks, threads;
    getLaunchConfig(size, blocks, threads);
    
    MatrixCUDA grad_input(input.getRows(), input.getCols());
    grad_input.toGPU();
    
    MatrixCUDA input_gpu = input;
    MatrixCUDA grad_out_gpu = output_gradient;
    if (!input_gpu.isOnGPU()) input_gpu.toGPU();
    if (!grad_out_gpu.isOnGPU()) grad_out_gpu.toGPU();
    
    relu_backward_kernel<<<blocks, threads>>>(
        input_gpu.getDevicePointer(),
        grad_out_gpu.getDevicePointer(),
        grad_input.getDevicePointer(),
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ReLUCUDA::backward: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    return grad_input;
}

std::unique_ptr<ActivationCUDA> ReLUCUDA::clone() const {
    return std::make_unique<ReLUCUDA>();
}

// ==================== TANH CUDA IMPLEMENTATION ====================

MatrixCUDA TanhCUDA::forward(const MatrixCUDA& input) const {
    Matrix cpu_input = static_cast<Matrix>(input);
    Matrix cpu_output = cpu_input.apply([](double x) {
        return std::tanh(x);
    });
    return MatrixCUDA(cpu_output);
}

MatrixCUDA TanhCUDA::backward(const MatrixCUDA& input, 
                            const MatrixCUDA& output_gradient) const {
    MatrixCUDA activated = forward(input);
    Matrix cpu_activated = static_cast<Matrix>(activated);
    Matrix cpu_grad = static_cast<Matrix>(output_gradient);
    
    Matrix derivative = cpu_activated.apply([](double x) {
        return 1.0 - x * x;
    });
    
    Matrix grad_input = derivative.hadamard(cpu_grad);
    
    return MatrixCUDA(grad_input);
}

std::unique_ptr<ActivationCUDA> TanhCUDA::clone() const {
    return std::make_unique<TanhCUDA>();
}

// ==================== LEAKY RELU CUDA IMPLEMENTATION ====================

MatrixCUDA LeakyReLUCUDA::forward(const MatrixCUDA& input) const {
    Matrix cpu_input = static_cast<Matrix>(input);
    float local_alpha = alpha;
    Matrix cpu_output = cpu_input.apply([local_alpha](double x) {
        return (x > 0.0) ? x : local_alpha * x;
    });
    return MatrixCUDA(cpu_output);
}

MatrixCUDA LeakyReLUCUDA::backward(const MatrixCUDA& input, 
                                 const MatrixCUDA& output_gradient) const {
    Matrix cpu_input = static_cast<Matrix>(input);
    Matrix cpu_grad = static_cast<Matrix>(output_gradient);
    float local_alpha = alpha;
    
    Matrix derivative = cpu_input.apply([local_alpha](double x) {
        return (x > 0.0) ? 1.0 : local_alpha;
    });
    
    Matrix grad_input = derivative.hadamard(cpu_grad);
    
    return MatrixCUDA(grad_input);
}

std::unique_ptr<ActivationCUDA> LeakyReLUCUDA::clone() const {
    return std::make_unique<LeakyReLUCUDA>(alpha);
}

// ==================== ELU CUDA IMPLEMENTATION ====================

MatrixCUDA ELUCUDA::forward(const MatrixCUDA& input) const {
    Matrix cpu_input = static_cast<Matrix>(input);
    float local_alpha = alpha;
    Matrix cpu_output = cpu_input.apply([local_alpha](double x) {
        return (x > 0.0) ? x : local_alpha * (std::exp(x) - 1.0);
    });
    return MatrixCUDA(cpu_output);
}

MatrixCUDA ELUCUDA::backward(const MatrixCUDA& input, 
                           const MatrixCUDA& output_gradient) const {
    Matrix cpu_input = static_cast<Matrix>(input);
    Matrix cpu_grad = static_cast<Matrix>(output_gradient);
    float local_alpha = alpha;
    
    Matrix derivative = cpu_input.apply([local_alpha](double x) {
        return (x > 0.0) ? 1.0 : local_alpha * std::exp(x);
    });
    
    Matrix grad_input = derivative.hadamard(cpu_grad);
    
    return MatrixCUDA(grad_input);
}

std::unique_ptr<ActivationCUDA> ELUCUDA::clone() const {
    return std::make_unique<ELUCUDA>(alpha);
}
