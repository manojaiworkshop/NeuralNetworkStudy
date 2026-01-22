#include "nn/attention_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA Kernels for Attention

__global__ void scaled_dot_product_kernel(const float* Q, const float* K, float* scores,
                                         int batch_seq, int d_k, int seq_len, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_seq && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) {
            sum += Q[row * d_k + i] * K[col * d_k + i];
        }
        scores[row * seq_len + col] = sum * scale;
    }
}

__global__ void softmax_kernel(float* scores, int batch_seq, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_seq) {
        // Find max for numerical stability
        float max_val = scores[row * seq_len];
        for (int i = 1; i < seq_len; i++) {
            max_val = fmaxf(max_val, scores[row * seq_len + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float val = expf(scores[row * seq_len + i] - max_val);
            scores[row * seq_len + i] = val;
            sum_exp += val;
        }
        
        // Normalize
        for (int i = 0; i < seq_len; i++) {
            scores[row * seq_len + i] /= sum_exp;
        }
    }
}

__global__ void attention_output_kernel(const float* attn_weights, const float* V, 
                                       float* output, int batch_seq, int seq_len, int d_v) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < batch_seq && col < d_v) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += attn_weights[row * seq_len + i] * V[i * d_v + col];
        }
        output[row * d_v + col] = sum;
    }
}

// ScaledDotProductAttentionCUDA Implementation

ScaledDotProductAttentionCUDA::ScaledDotProductAttentionCUDA(size_t d_k)
    : d_k(d_k), scale_factor(1.0 / std::sqrt(static_cast<double>(d_k))),
      d_scores(nullptr), d_attention_weights(nullptr), gpu_allocated(false) {}

ScaledDotProductAttentionCUDA::~ScaledDotProductAttentionCUDA() {
    freeGPU();
}

void ScaledDotProductAttentionCUDA::allocateGPU(size_t max_seq_len) {
    if (gpu_allocated) return;
    
    size_t scores_size = max_seq_len * max_seq_len * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_attention_weights, scores_size));
    
    gpu_allocated = true;
}

void ScaledDotProductAttentionCUDA::freeGPU() {
    if (d_scores) cudaFree(d_scores);
    if (d_attention_weights) cudaFree(d_attention_weights);
    gpu_allocated = false;
}

MatrixCUDA ScaledDotProductAttentionCUDA::forward(const MatrixCUDA& Q, 
                                                  const MatrixCUDA& K,
                                                  const MatrixCUDA& V) {
    size_t batch_seq = Q.getRows();
    size_t seq_len = K.getRows();
    size_t d_v = V.getCols();
    
    // Ensure inputs are on GPU
    MatrixCUDA Q_gpu = Q;
    MatrixCUDA K_gpu = K;
    MatrixCUDA V_gpu = V;
    Q_gpu.toGPU();
    K_gpu.toGPU();
    V_gpu.toGPU();
    
    // Allocate GPU memory if needed
    allocateGPU(std::max(batch_seq, seq_len));
    
    // Compute attention scores: QK^T / sqrt(d_k)
    dim3 blockDim(16, 16);
    dim3 gridDim((batch_seq + 15) / 16, (seq_len + 15) / 16);
    
    scaled_dot_product_kernel<<<gridDim, blockDim>>>(
        Q_gpu.getDevicePointer(), K_gpu.getDevicePointer(), d_scores,
        batch_seq, d_k, seq_len, scale_factor
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Apply softmax
    dim3 softmax_grid((batch_seq + 255) / 256);
    dim3 softmax_block(256);
    
    softmax_kernel<<<softmax_grid, softmax_block>>>(d_scores, batch_seq, seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy attention weights for caching
    cached_attention_weights = MatrixCUDA(batch_seq, seq_len);
    cached_attention_weights.toGPU();
    CUDA_CHECK(cudaMemcpy(cached_attention_weights.getDevicePointer(), d_scores,
                         batch_seq * seq_len * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Compute output: attention_weights × V
    MatrixCUDA output(batch_seq, d_v);
    output.toGPU();
    
    dim3 output_grid((batch_seq + 15) / 16, (d_v + 15) / 16);
    attention_output_kernel<<<output_grid, blockDim>>>(
        d_scores, V_gpu.getDevicePointer(), output.getDevicePointer(),
        batch_seq, seq_len, d_v
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Cache for backward
    cached_Q = Q_gpu;
    cached_K = K_gpu;
    cached_V = V_gpu;
    
    return output;
}

void ScaledDotProductAttentionCUDA::backward(const MatrixCUDA& grad_output,
                                            MatrixCUDA& dQ, MatrixCUDA& dK, 
                                            MatrixCUDA& dV) {
    // Simplified backward (full implementation would require more kernels)
    // This is a placeholder for gradient computation
    dV = cached_attention_weights.transposeGPU().multiplyGPU(grad_output);
    
    MatrixCUDA grad_attn = grad_output.multiplyGPU(cached_V.transposeGPU());
    
    // Backward through softmax and scaling (simplified)
    MatrixCUDA grad_scores = grad_attn;
    
    // dQ and dK computation (simplified)
    dQ = grad_scores.multiplyGPU(cached_K);
    dK = grad_scores.transposeGPU().multiplyGPU(cached_Q);
}

// MultiHeadAttentionCUDA Implementation

MultiHeadAttentionCUDA::MultiHeadAttentionCUDA(size_t d_model, size_t num_heads)
    : d_model(d_model), num_heads(num_heads),
      d_k(d_model / num_heads), d_v(d_model / num_heads) {
    
    // Initialize projection matrices
    W_Q.resize(num_heads);
    W_K.resize(num_heads);
    W_V.resize(num_heads);
    dW_Q.resize(num_heads);
    dW_K.resize(num_heads);
    dW_V.resize(num_heads);
    
    for (size_t h = 0; h < num_heads; h++) {
        W_Q[h] = MatrixCUDA(d_model, d_k);
        W_K[h] = MatrixCUDA(d_model, d_k);
        W_V[h] = MatrixCUDA(d_model, d_v);
        
        dW_Q[h] = MatrixCUDA(d_model, d_k, 0.0);
        dW_K[h] = MatrixCUDA(d_model, d_k, 0.0);
        dW_V[h] = MatrixCUDA(d_model, d_v, 0.0);
        
        attention_heads.push_back(
            std::make_unique<ScaledDotProductAttentionCUDA>(d_k)
        );
    }
    
    W_O = MatrixCUDA(d_model, d_model);
    dW_O = MatrixCUDA(d_model, d_model, 0.0);
    
    initializeWeights();
}

void MultiHeadAttentionCUDA::initializeWeights() {
    // Xavier initialization
    double std = std::sqrt(2.0 / (d_model + d_k));
    
    for (size_t h = 0; h < num_heads; h++) {
        // Initialize on CPU then transfer to GPU
        Matrix temp_Q(d_model, d_k);
        Matrix temp_K(d_model, d_k);
        Matrix temp_V(d_model, d_v);
        
        temp_Q.randomNormal(0.0, std);
        temp_K.randomNormal(0.0, std);
        temp_V.randomNormal(0.0, std);
        
        W_Q[h] = MatrixCUDA(temp_Q);
        W_K[h] = MatrixCUDA(temp_K);
        W_V[h] = MatrixCUDA(temp_V);
        
        W_Q[h].toGPU();
        W_K[h].toGPU();
        W_V[h].toGPU();
    }
    
    Matrix temp_O(d_model, d_model);
    temp_O.randomNormal(0.0, std);
    W_O = MatrixCUDA(temp_O);
    W_O.toGPU();
}

MatrixCUDA MultiHeadAttentionCUDA::forward(const MatrixCUDA& Q, 
                                          const MatrixCUDA& K,
                                          const MatrixCUDA& V) {
    cached_Q = Q;
    cached_K = K;
    cached_V = V;
    
    std::vector<MatrixCUDA> head_outputs;
    
    // Process each attention head in parallel
    for (size_t h = 0; h < num_heads; h++) {
        // Project Q, K, V
        MatrixCUDA Q_proj = Q.multiplyGPU(W_Q[h]);
        MatrixCUDA K_proj = K.multiplyGPU(W_K[h]);
        MatrixCUDA V_proj = V.multiplyGPU(W_V[h]);
        
        // Apply attention
        MatrixCUDA head_output = attention_heads[h]->forward(Q_proj, K_proj, V_proj);
        head_outputs.push_back(head_output);
    }
    
    // Concatenate heads (simplified - should be done on GPU)
    cached_heads = head_outputs;
    
    // For now, concatenate on CPU
    size_t batch_seq = head_outputs[0].getRows();
    MatrixCUDA concat(batch_seq, d_model);
    
    for (size_t h = 0; h < num_heads; h++) {
        MatrixCUDA& head = head_outputs[h];
        head.toCPU();
        
        for (size_t i = 0; i < batch_seq; i++) {
            for (size_t j = 0; j < d_v; j++) {
                concat.set(i, h * d_v + j, head.get(i, j));
            }
        }
    }
    
    concat.toGPU();
    cached_concat = concat;
    
    // Output projection
    MatrixCUDA output = concat.multiplyGPU(W_O);
    
    return output;
}

void MultiHeadAttentionCUDA::backward(const MatrixCUDA& grad_output,
                                     MatrixCUDA& dQ, MatrixCUDA& dK, 
                                     MatrixCUDA& dV) {
    // Backward through output projection
    MatrixCUDA grad_concat = grad_output.multiplyGPU(W_O.transposeGPU());
    dW_O = dW_O.addGPU(cached_concat.transposeGPU().multiplyGPU(grad_output));
    
    // Initialize gradients
    dQ = MatrixCUDA(cached_Q.getRows(), cached_Q.getCols(), 0.0);
    dK = MatrixCUDA(cached_K.getRows(), cached_K.getCols(), 0.0);
    dV = MatrixCUDA(cached_V.getRows(), cached_V.getCols(), 0.0);
    dQ.toGPU();
    dK.toGPU();
    dV.toGPU();
    
    // Backward through each head
    for (size_t h = 0; h < num_heads; h++) {
        // Extract gradient for this head
        MatrixCUDA grad_head(grad_concat.getRows(), d_v);
        grad_concat.toCPU();
        
        for (size_t i = 0; i < grad_concat.getRows(); i++) {
            for (size_t j = 0; j < d_v; j++) {
                grad_head.set(i, j, grad_concat.get(i, h * d_v + j));
            }
        }
        grad_head.toGPU();
        
        // Backward through attention
        MatrixCUDA dQ_h, dK_h, dV_h;
        attention_heads[h]->backward(grad_head, dQ_h, dK_h, dV_h);
        
        // Backward through projections
        MatrixCUDA dQ_proj = dQ_h.multiplyGPU(W_Q[h].transposeGPU());
        MatrixCUDA dK_proj = dK_h.multiplyGPU(W_K[h].transposeGPU());
        MatrixCUDA dV_proj = dV_h.multiplyGPU(W_V[h].transposeGPU());
        
        // Accumulate gradients
        dQ = dQ.addGPU(dQ_proj);
        dK = dK.addGPU(dK_proj);
        dV = dV.addGPU(dV_proj);
        
        // Weight gradients
        dW_Q[h] = dW_Q[h].addGPU(cached_Q.transposeGPU().multiplyGPU(dQ_h));
        dW_K[h] = dW_K[h].addGPU(cached_K.transposeGPU().multiplyGPU(dK_h));
        dW_V[h] = dW_V[h].addGPU(cached_V.transposeGPU().multiplyGPU(dV_h));
    }
}

void MultiHeadAttentionCUDA::updateParameters(double learning_rate) {
    for (size_t h = 0; h < num_heads; h++) {
        // W = W - lr * dW (on GPU)
        MatrixCUDA grad_scaled_Q = dW_Q[h];
        MatrixCUDA grad_scaled_K = dW_K[h];
        MatrixCUDA grad_scaled_V = dW_V[h];
        
        grad_scaled_Q.toCPU();
        grad_scaled_K.toCPU();
        grad_scaled_V.toCPU();
        
        for (size_t i = 0; i < d_model; i++) {
            for (size_t j = 0; j < d_k; j++) {
                grad_scaled_Q.set(i, j, grad_scaled_Q.get(i, j) * learning_rate);
                grad_scaled_K.set(i, j, grad_scaled_K.get(i, j) * learning_rate);
            }
            for (size_t j = 0; j < d_v; j++) {
                grad_scaled_V.set(i, j, grad_scaled_V.get(i, j) * learning_rate);
            }
        }
        
        grad_scaled_Q.toGPU();
        grad_scaled_K.toGPU();
        grad_scaled_V.toGPU();
        
        W_Q[h] = W_Q[h].subtractGPU(grad_scaled_Q);
        W_K[h] = W_K[h].subtractGPU(grad_scaled_K);
        W_V[h] = W_V[h].subtractGPU(grad_scaled_V);
        
        // Reset gradients
        dW_Q[h] = MatrixCUDA(d_model, d_k, 0.0);
        dW_K[h] = MatrixCUDA(d_model, d_k, 0.0);
        dW_V[h] = MatrixCUDA(d_model, d_v, 0.0);
    }
    
    // Update W_O
    MatrixCUDA grad_scaled_O = dW_O;
    grad_scaled_O.toCPU();
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_model; j++) {
            grad_scaled_O.set(i, j, grad_scaled_O.get(i, j) * learning_rate);
        }
    }
    grad_scaled_O.toGPU();
    W_O = W_O.subtractGPU(grad_scaled_O);
    dW_O = MatrixCUDA(d_model, d_model, 0.0);
}

std::vector<MatrixCUDA> MultiHeadAttentionCUDA::getAllAttentionWeights() const {
    std::vector<MatrixCUDA> weights;
    for (const auto& head : attention_heads) {
        weights.push_back(head->getAttentionWeights());
    }
    return weights;
}

// FeedForwardCUDA Implementation

FeedForwardCUDA::FeedForwardCUDA(size_t d_model, size_t d_ff)
    : d_model(d_model), d_ff(d_ff) {
    
    W1 = MatrixCUDA(d_model, d_ff);
    b1 = MatrixCUDA(1, d_ff, 0.0);
    W2 = MatrixCUDA(d_ff, d_model);
    b2 = MatrixCUDA(1, d_model, 0.0);
    
    dW1 = MatrixCUDA(d_model, d_ff, 0.0);
    db1 = MatrixCUDA(1, d_ff, 0.0);
    dW2 = MatrixCUDA(d_ff, d_model, 0.0);
    db2 = MatrixCUDA(1, d_model, 0.0);
    
    activation = std::make_unique<ReLUCUDA>();
    
    initializeWeights();
}

void FeedForwardCUDA::initializeWeights() {
    // He initialization
    Matrix temp_W1(d_model, d_ff);
    Matrix temp_W2(d_ff, d_model);
    
    double std1 = std::sqrt(2.0 / d_model);
    double std2 = std::sqrt(2.0 / d_ff);
    
    temp_W1.randomNormal(0.0, std1);
    temp_W2.randomNormal(0.0, std2);
    
    W1 = MatrixCUDA(temp_W1);
    W2 = MatrixCUDA(temp_W2);
    
    W1.toGPU();
    W2.toGPU();
    b1.toGPU();
    b2.toGPU();
}

MatrixCUDA FeedForwardCUDA::forward(const MatrixCUDA& input) {
    cached_input = input;
    
    // First linear layer
    cached_hidden_pre = input.multiplyGPU(W1);
    
    // Add bias (simplified - should use kernel)
    cached_hidden_pre.toCPU();
    for (size_t i = 0; i < cached_hidden_pre.getRows(); i++) {
        for (size_t j = 0; j < d_ff; j++) {
            cached_hidden_pre.set(i, j, cached_hidden_pre.get(i, j) + b1.get(0, j));
        }
    }
    cached_hidden_pre.toGPU();
    
    // ReLU activation
    cached_hidden = activation->forward(cached_hidden_pre);
    
    // Second linear layer
    MatrixCUDA output = cached_hidden.multiplyGPU(W2);
    
    // Add bias
    output.toCPU();
    for (size_t i = 0; i < output.getRows(); i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b2.get(0, j));
        }
    }
    output.toGPU();
    
    return output;
}

MatrixCUDA FeedForwardCUDA::backward(const MatrixCUDA& grad_output) {
    // Gradient w.r.t W2 and b2
    dW2 = dW2.addGPU(cached_hidden.transposeGPU().multiplyGPU(grad_output));
    
    // Bias gradient (sum over batch)
    MatrixCUDA grad_out_cpu = grad_output;
    grad_out_cpu.toCPU();
    for (size_t j = 0; j < d_model; j++) {
        for (size_t i = 0; i < grad_out_cpu.getRows(); i++) {
            db2.set(0, j, db2.get(0, j) + grad_out_cpu.get(i, j));
        }
    }
    
    // Gradient w.r.t hidden
    MatrixCUDA grad_hidden = grad_output.multiplyGPU(W2.transposeGPU());
    
    // Backward through ReLU
    MatrixCUDA grad_hidden_pre = activation->backward(cached_hidden_pre, grad_hidden);
    
    // Gradient w.r.t W1 and b1
    dW1 = dW1.addGPU(cached_input.transposeGPU().multiplyGPU(grad_hidden_pre));
    
    grad_hidden_pre.toCPU();
    for (size_t j = 0; j < d_ff; j++) {
        for (size_t i = 0; i < grad_hidden_pre.getRows(); i++) {
            db1.set(0, j, db1.get(0, j) + grad_hidden_pre.get(i, j));
        }
    }
    
    // Gradient w.r.t input
    MatrixCUDA grad_input = grad_hidden_pre.multiplyGPU(W1.transposeGPU());
    
    return grad_input;
}

void FeedForwardCUDA::updateParameters(double learning_rate) {
    // Update on GPU
    MatrixCUDA lr_dW1 = dW1;
    MatrixCUDA lr_dW2 = dW2;
    
    lr_dW1.toCPU();
    lr_dW2.toCPU();
    
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            lr_dW1.set(i, j, lr_dW1.get(i, j) * learning_rate);
        }
    }
    
    for (size_t i = 0; i < d_ff; i++) {
        for (size_t j = 0; j < d_model; j++) {
            lr_dW2.set(i, j, lr_dW2.get(i, j) * learning_rate);
        }
    }
    
    lr_dW1.toGPU();
    lr_dW2.toGPU();
    
    W1 = W1.subtractGPU(lr_dW1);
    W2 = W2.subtractGPU(lr_dW2);
    
    // Biases
    for (size_t j = 0; j < d_ff; j++) {
        b1.set(0, j, b1.get(0, j) - learning_rate * db1.get(0, j));
    }
    for (size_t j = 0; j < d_model; j++) {
        b2.set(0, j, b2.get(0, j) - learning_rate * db2.get(0, j));
    }
    
    // Reset gradients
    dW1 = MatrixCUDA(d_model, d_ff, 0.0);
    db1 = MatrixCUDA(1, d_ff, 0.0);
    dW2 = MatrixCUDA(d_ff, d_model, 0.0);
    db2 = MatrixCUDA(1, d_model, 0.0);
}

// LayerNormCUDA Implementation (simplified)

LayerNormCUDA::LayerNormCUDA(size_t normalized_shape, double epsilon)
    : normalized_shape(normalized_shape), epsilon(epsilon) {
    
    gamma = MatrixCUDA(1, normalized_shape, 1.0);
    beta = MatrixCUDA(1, normalized_shape, 0.0);
    gamma_grad = MatrixCUDA(1, normalized_shape, 0.0);
    beta_grad = MatrixCUDA(1, normalized_shape, 0.0);
    
    gamma.toGPU();
    beta.toGPU();
}

MatrixCUDA LayerNormCUDA::forward(const MatrixCUDA& input) {
    cached_input = input;
    
    size_t batch_seq = input.getRows();
    size_t features = input.getCols();
    
    cached_mean = MatrixCUDA(batch_seq, 1, 0.0);
    cached_std = MatrixCUDA(batch_seq, 1, 0.0);
    
    MatrixCUDA input_cpu = input;
    input_cpu.toCPU();
    
    // Compute mean and std
    for (size_t i = 0; i < batch_seq; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < features; j++) {
            sum += input_cpu.get(i, j);
        }
        double mean = sum / features;
        cached_mean.set(i, 0, mean);
        
        double sum_sq = 0.0;
        for (size_t j = 0; j < features; j++) {
            double diff = input_cpu.get(i, j) - mean;
            sum_sq += diff * diff;
        }
        double std = std::sqrt(sum_sq / features + epsilon);
        cached_std.set(i, 0, std);
    }
    
    // Normalize
    cached_normalized = MatrixCUDA(batch_seq, features);
    for (size_t i = 0; i < batch_seq; i++) {
        double mean = cached_mean.get(i, 0);
        double std = cached_std.get(i, 0);
        for (size_t j = 0; j < features; j++) {
            double normalized = (input_cpu.get(i, j) - mean) / std;
            cached_normalized.set(i, j, normalized);
        }
    }
    
    // Scale and shift
    MatrixCUDA output(batch_seq, features);
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < features; j++) {
            output.set(i, j, gamma.get(0, j) * cached_normalized.get(i, j) + beta.get(0, j));
        }
    }
    
    output.toGPU();
    return output;
}

MatrixCUDA LayerNormCUDA::backward(const MatrixCUDA& grad_output) {
    // Simplified backward pass
    size_t batch_seq = grad_output.getRows();
    size_t features = grad_output.getCols();
    
    MatrixCUDA grad_out_cpu = grad_output;
    grad_out_cpu.toCPU();
    
    // Gradient w.r.t gamma and beta
    for (size_t j = 0; j < features; j++) {
        for (size_t i = 0; i < batch_seq; i++) {
            gamma_grad.set(0, j, gamma_grad.get(0, j) + 
                          grad_out_cpu.get(i, j) * cached_normalized.get(i, j));
            beta_grad.set(0, j, beta_grad.get(0, j) + grad_out_cpu.get(i, j));
        }
    }
    
    // Gradient w.r.t input (simplified)
    MatrixCUDA grad_input(batch_seq, features);
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < features; j++) {
            grad_input.set(i, j, grad_out_cpu.get(i, j) * gamma.get(0, j) / cached_std.get(i, 0));
        }
    }
    
    grad_input.toGPU();
    return grad_input;
}

void LayerNormCUDA::updateParameters(double learning_rate) {
    for (size_t j = 0; j < normalized_shape; j++) {
        gamma.set(0, j, gamma.get(0, j) - learning_rate * gamma_grad.get(0, j));
        beta.set(0, j, beta.get(0, j) - learning_rate * beta_grad.get(0, j));
    }
    gamma_grad = MatrixCUDA(1, normalized_shape, 0.0);
    beta_grad = MatrixCUDA(1, normalized_shape, 0.0);
}

// EncoderLayerCUDA and other higher-level components follow similar patterns...
// For brevity, including simplified versions

EncoderLayerCUDA::EncoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff)
    : d_model(d_model) {
    
    self_attention = std::make_unique<MultiHeadAttentionCUDA>(d_model, num_heads);
    feed_forward = std::make_unique<FeedForwardCUDA>(d_model, d_ff);
    norm1 = std::make_unique<LayerNormCUDA>(d_model);
    norm2 = std::make_unique<LayerNormCUDA>(d_model);
}

MatrixCUDA EncoderLayerCUDA::forward(const MatrixCUDA& input) {
    cached_input = input;
    
    // Self-attention
    cached_attn_output = self_attention->forward(input, input, input);
    
    // Add & Norm
    MatrixCUDA attn_residual = input.addGPU(cached_attn_output);
    MatrixCUDA attn_normalized = norm1->forward(attn_residual);
    
    cached_ffn_input = attn_normalized;
    
    // Feed-forward
    MatrixCUDA ffn_output = feed_forward->forward(attn_normalized);
    
    // Add & Norm
    MatrixCUDA ffn_residual = attn_normalized.addGPU(ffn_output);
    MatrixCUDA output = norm2->forward(ffn_residual);
    
    return output;
}

MatrixCUDA EncoderLayerCUDA::backward(const MatrixCUDA& grad_output) {
    // Backward through norm2
    MatrixCUDA grad_ffn_residual = norm2->backward(grad_output);
    
    // Backward through FFN
    MatrixCUDA grad_ffn = feed_forward->backward(grad_ffn_residual);
    
    // Add gradient from residual
    MatrixCUDA grad_attn_norm = grad_ffn_residual.addGPU(grad_ffn);
    
    // Backward through norm1
    MatrixCUDA grad_attn_residual = norm1->backward(grad_attn_norm);
    
    // Backward through attention
    MatrixCUDA dQ, dK, dV;
    self_attention->backward(grad_attn_residual, dQ, dK, dV);
    
    // Add gradient from residual
    MatrixCUDA grad_input = grad_attn_residual.addGPU(dQ);
    
    return grad_input;
}

void EncoderLayerCUDA::updateParameters(double learning_rate) {
    self_attention->updateParameters(learning_rate);
    feed_forward->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
}

std::vector<MatrixCUDA> EncoderLayerCUDA::getAttentionWeights() const {
    return self_attention->getAllAttentionWeights();
}

// TransformerEncoderCUDA Implementation

TransformerEncoderCUDA::TransformerEncoderCUDA(size_t num_layers, size_t d_model,
                                             size_t num_heads, size_t d_ff)
    : num_layers(num_layers), d_model(d_model) {
    
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<EncoderLayerCUDA>(d_model, num_heads, d_ff));
    }
    
    final_norm = std::make_unique<LayerNormCUDA>(d_model);
}

MatrixCUDA TransformerEncoderCUDA::forward(const MatrixCUDA& input) {
    MatrixCUDA x = input;
    
    for (size_t i = 0; i < num_layers; i++) {
        x = layers[i]->forward(x);
    }
    
    x = final_norm->forward(x);
    
    return x;
}

MatrixCUDA TransformerEncoderCUDA::backward(const MatrixCUDA& grad_output) {
    MatrixCUDA grad = final_norm->backward(grad_output);
    
    for (int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
    }
    
    return grad;
}

void TransformerEncoderCUDA::updateParameters(double learning_rate) {
    for (auto& layer : layers) {
        layer->updateParameters(learning_rate);
    }
    final_norm->updateParameters(learning_rate);
}

std::vector<std::vector<MatrixCUDA>> TransformerEncoderCUDA::getAllAttentionWeights() const {
    std::vector<std::vector<MatrixCUDA>> all_weights;
    for (const auto& layer : layers) {
        all_weights.push_back(layer->getAttentionWeights());
    }
    return all_weights;
}

// TokenEmbeddingCUDA and PositionalEncodingCUDA (simplified)

TokenEmbeddingCUDA::TokenEmbeddingCUDA(size_t vocab_size, size_t embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    
    embeddings = MatrixCUDA(vocab_size, embedding_dim);
    gradients = MatrixCUDA(vocab_size, embedding_dim, 0.0);
    
    initializeEmbeddings();
}

void TokenEmbeddingCUDA::initializeEmbeddings() {
    Matrix temp(vocab_size, embedding_dim);
    temp.randomNormal(0.0, 0.02);
    embeddings = MatrixCUDA(temp);
    embeddings.toGPU();
}

MatrixCUDA TokenEmbeddingCUDA::forward(const std::vector<std::vector<int>>& token_ids) {
    size_t batch_size = token_ids.size();
    size_t seq_len = token_ids[0].size();
    
    MatrixCUDA output(batch_size * seq_len, embedding_dim);
    embeddings.toCPU();
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = token_ids[b][t];
            size_t out_idx = b * seq_len + t;
            
            for (size_t d = 0; d < embedding_dim; d++) {
                output.set(out_idx, d, embeddings.get(token_id, d));
            }
        }
    }
    
    output.toGPU();
    return output;
}

void TokenEmbeddingCUDA::backward(const MatrixCUDA& grad_output,
                                 const std::vector<std::vector<int>>& token_ids) {
    // Accumulate gradients
    MatrixCUDA grad_out_cpu = grad_output;
    grad_out_cpu.toCPU();
    
    size_t batch_size = token_ids.size();
    size_t seq_len = token_ids[0].size();
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = token_ids[b][t];
            size_t out_idx = b * seq_len + t;
            
            for (size_t d = 0; d < embedding_dim; d++) {
                gradients.set(token_id, d, 
                            gradients.get(token_id, d) + grad_out_cpu.get(out_idx, d));
            }
        }
    }
}

void TokenEmbeddingCUDA::updateParameters(double learning_rate) {
    embeddings.toCPU();
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < embedding_dim; j++) {
            embeddings.set(i, j, embeddings.get(i, j) - 
                          learning_rate * gradients.get(i, j));
        }
    }
    embeddings.toGPU();
    gradients = MatrixCUDA(vocab_size, embedding_dim, 0.0);
}

PositionalEncodingCUDA::PositionalEncodingCUDA(size_t max_seq_len, size_t d_model)
    : max_seq_len(max_seq_len), d_model(d_model) {
    
    encoding = MatrixCUDA(max_seq_len, d_model);
    
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / d_model);
            
            if (i % 2 == 0) {
                encoding.set(pos, i, std::sin(angle));
            } else {
                encoding.set(pos, i, std::cos(angle));
            }
        }
    }
    
    encoding.toGPU();
}

MatrixCUDA PositionalEncodingCUDA::forward(const MatrixCUDA& input) {
    MatrixCUDA output = input;
    output.toCPU();
    encoding.toCPU();
    
    size_t batch_seq = input.getRows();
    
    for (size_t i = 0; i < batch_seq; i++) {
        size_t pos = i % max_seq_len;
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + encoding.get(pos, j));
        }
    }
    
    output.toGPU();
    return output;
}
