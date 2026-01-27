#include "nn/bert_encoder_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper to convert MatrixCUDA to host array
float* matrixToHost(const MatrixCUDA& m) {
    size_t size = m.getRows() * m.getCols();
    float* host = new float[size];
    for (size_t i = 0; i < m.getRows(); i++) {
        for (size_t j = 0; j < m.getCols(); j++) {
            host[i * m.getCols() + j] = static_cast<float>(m.get(i, j));
        }
    }
    return host;
}

// Helper to set MatrixCUDA from host array
void hostToMatrix(MatrixCUDA& m, const float* host) {
    for (size_t i = 0; i < m.getRows(); i++) {
        for (size_t j = 0; j < m.getCols(); j++) {
            m.set(i, j, static_cast<double>(host[i * m.getCols() + j]));
        }
    }
}

// ============================================================================
// CUDA KERNELS
// ============================================================================

__global__ void layerNormForwardKernel(const float* input, float* output, 
                                       float* mean, float* variance,
                                       const float* gamma, const float* beta,
                                       int seq_len, int d_model, float eps) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    
    // Calculate mean
    float sum = 0.0f;
    for (int j = 0; j < d_model; j++) {
        sum += input[row * d_model + j];
    }
    float m = sum / d_model;
    mean[row] = m;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int j = 0; j < d_model; j++) {
        float diff = input[row * d_model + j] - m;
        var_sum += diff * diff;
    }
    float var = var_sum / d_model;
    variance[row] = var;
    
    // Normalize and apply gamma, beta
    float std = sqrtf(var + eps);
    for (int j = 0; j < d_model; j++) {
        float normalized = (input[row * d_model + j] - m) / std;
        output[row * d_model + j] = gamma[j] * normalized + beta[j];
    }
}

__global__ void addBiasKernel(float* output, const float* bias, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[row * cols + col] += bias[col];
    }
}

__global__ void reluForwardKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void reluBackwardKernel(const float* input, const float* grad_output, 
                                   float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

__global__ void softmaxKernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    // Find max for numerical stability
    float max_val = input[row * cols];
    for (int j = 1; j < cols; j++) {
        max_val = fmaxf(max_val, input[row * cols + j]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float exp_val = expf(input[row * cols + j] - max_val);
        output[row * cols + j] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (int j = 0; j < cols; j++) {
        output[row * cols + j] /= sum;
    }
}

__global__ void embeddingLookupKernel(const float* embeddings, const int* token_ids,
                                     const float* pos_encodings, float* output,
                                     int seq_len, int d_model, int vocab_size) {
    int pos = blockIdx.x;
    int dim = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pos < seq_len && dim < d_model) {
        int token_id = token_ids[pos];
        if (token_id >= 0 && token_id < vocab_size) {
            output[pos * d_model + dim] = embeddings[token_id * d_model + dim] + 
                                         pos_encodings[pos * d_model + dim];
        }
    }
}

// ============================================================================
// LAYER NORMALIZATION IMPLEMENTATION
// ============================================================================

LayerNormCUDA::LayerNormCUDA(size_t d_model, float epsilon)
    : d_model(d_model), eps(epsilon)
{
    gamma = MatrixCUDA(1, d_model);
    beta = MatrixCUDA(1, d_model);
    gamma.fill(1.0f);
    beta.fill(0.0f);
}

MatrixCUDA LayerNormCUDA::forward(const MatrixCUDA& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    MatrixCUDA output(seq_len, d_model);
    mean_cache = MatrixCUDA(seq_len, 1);
    variance_cache = MatrixCUDA(seq_len, 1);
    
    // Simple layer norm on CPU (can be optimized later)
    for (size_t i = 0; i < seq_len; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < d_model; j++) {
            sum += input.get(i, j);
        }
        double m = sum / d_model;
        mean_cache.set(i, 0, m);
        
        double var_sum = 0.0;
        for (size_t j = 0; j < d_model; j++) {
            double diff = input.get(i, j) - m;
            var_sum += diff * diff;
        }
        double var = var_sum / d_model;
        variance_cache.set(i, 0, var);
        
        double std = std::sqrt(var + eps);
        for (size_t j = 0; j < d_model; j++) {
            double normalized = (input.get(i, j) - m) / std;
            output.set(i, j, gamma.get(0, j) * normalized + beta.get(0, j));
        }
    }
    
    return output;
}

MatrixCUDA LayerNormCUDA::backward(const MatrixCUDA& grad_output) {
    size_t seq_len = input_cache.getRows();
    MatrixCUDA grad_input(seq_len, d_model);
    MatrixCUDA grad_gamma(1, d_model);
    MatrixCUDA grad_beta(1, d_model);
    
    grad_gamma.fill(0.0);
    grad_beta.fill(0.0);
    
    for (size_t i = 0; i < seq_len; i++) {
        double mean = mean_cache.get(i, 0);
        double variance = variance_cache.get(i, 0);
        double std = std::sqrt(variance + eps);
        
        // Accumulate gradients for gamma and beta
        for (size_t j = 0; j < d_model; j++) {
            double x = input_cache.get(i, j);
            double normalized = (x - mean) / std;
            
            grad_gamma.set(0, j, grad_gamma.get(0, j) + grad_output.get(i, j) * normalized);
            grad_beta.set(0, j, grad_beta.get(0, j) + grad_output.get(i, j));
        }
        
        // Compute gradient w.r.t. normalized values
        double grad_var = 0.0;
        double grad_mean = 0.0;
        
        for (size_t j = 0; j < d_model; j++) {
            double x = input_cache.get(i, j);
            double g = gamma.get(0, j);
            double dy = grad_output.get(i, j);
            
            double dx_norm = dy * g;
            grad_var += dx_norm * (x - mean) * (-0.5) * std::pow(std, -3);
            grad_mean += dx_norm * (-1.0 / std);
        }
        
        // Add contribution from variance to mean gradient
        grad_mean += grad_var * (-2.0 / d_model) * 0.0;  // sum of (x - mean) is 0
        
        // Compute gradient w.r.t. input
        for (size_t j = 0; j < d_model; j++) {
            double x = input_cache.get(i, j);
            double g = gamma.get(0, j);
            double dy = grad_output.get(i, j);
            
            double dx_norm = dy * g;
            double dx = dx_norm / std + grad_var * 2.0 * (x - mean) / d_model + grad_mean / d_model;
            grad_input.set(i, j, dx);
        }
    }
    
    // Update gamma and beta (simplified - accumulate gradients)
    for (size_t j = 0; j < d_model; j++) {
        gamma.set(0, j, gamma.get(0, j) - 0.0001f * grad_gamma.get(0, j));  // Small LR for stability
        beta.set(0, j, beta.get(0, j) - 0.0001f * grad_beta.get(0, j));
    }
    
    return grad_input;
}

void LayerNormCUDA::updateParameters(float learning_rate) {
    // Parameters are updated in backward pass for stability
}

// ============================================================================
// MULTI-HEAD ATTENTION IMPLEMENTATION
// ============================================================================

MultiHeadAttentionCUDA::MultiHeadAttentionCUDA(size_t d_model, size_t num_heads)
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads)
{
    for (size_t h = 0; h < num_heads; h++) {
        MatrixCUDA W_Q_h(d_model, d_k);
        MatrixCUDA W_K_h(d_model, d_k);
        MatrixCUDA W_V_h(d_model, d_k);
        
        W_Q_h.randomNormal(0.0f, 0.02f);
        W_K_h.randomNormal(0.0f, 0.02f);
        W_V_h.randomNormal(0.0f, 0.02f);
        
        W_Q.push_back(W_Q_h);
        W_K.push_back(W_K_h);
        W_V.push_back(W_V_h);
        
        grad_W_Q.push_back(MatrixCUDA(d_model, d_k));
        grad_W_K.push_back(MatrixCUDA(d_model, d_k));
        grad_W_V.push_back(MatrixCUDA(d_model, d_k));
    }
    
    W_O = MatrixCUDA(d_model, d_model);
    W_O.randomNormal(0.0f, 0.02f);
    b_O = MatrixCUDA(1, d_model);
    b_O.fill(0.0f);
    
    grad_W_O = MatrixCUDA(d_model, d_model);
    grad_b_O = MatrixCUDA(1, d_model);
}

MatrixCUDA MultiHeadAttentionCUDA::forward(const MatrixCUDA& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    Q_cache.clear();
    K_cache.clear();
    V_cache.clear();
    attention_weights_cache.clear();
    
    std::vector<MatrixCUDA> head_outputs;
    
    for (size_t h = 0; h < num_heads; h++) {
        // Q, K, V projections - use CPU multiplyGPU
        MatrixCUDA Q = input.multiplyGPU(W_Q[h]);
        MatrixCUDA K = input.multiplyGPU(W_K[h]);
        MatrixCUDA V = input.multiplyGPU(W_V[h]);
        
        Q_cache.push_back(Q);
        K_cache.push_back(K);
        V_cache.push_back(V);
        
        // Scaled dot-product attention
        MatrixCUDA K_T = K.transposeGPU();
        MatrixCUDA scores = Q.multiplyGPU(K_T);
        
        // Scale
        float scale = 1.0f / sqrtf(static_cast<float>(d_k));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                scores.set(i, j, scores.get(i, j) * scale);
            }
        }
        
        // Softmax
        MatrixCUDA attention_weights(seq_len, seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            double max_score = scores.get(i, 0);
            for (size_t j = 1; j < seq_len; j++) {
                max_score = std::max(max_score, scores.get(i, j));
            }
            
            double sum_exp = 0.0;
            for (size_t j = 0; j < seq_len; j++) {
                sum_exp += std::exp(scores.get(i, j) - max_score);
            }
            
            for (size_t j = 0; j < seq_len; j++) {
                double weight = std::exp(scores.get(i, j) - max_score) / sum_exp;
                attention_weights.set(i, j, weight);
            }
        }
        
        attention_weights_cache.push_back(attention_weights);
        
        // Apply attention
        MatrixCUDA head_output = attention_weights.multiplyGPU(V);
        head_outputs.push_back(head_output);
    }
    
    // Concatenate heads
    MatrixCUDA concatenated(seq_len, d_model);
    for (size_t i = 0; i < seq_len; i++) {
        size_t col_offset = 0;
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t j = 0; j < d_k; j++) {
                concatenated.set(i, col_offset + j, head_outputs[h].get(i, j));
            }
            col_offset += d_k;
        }
    }
    
    // Output projection
    MatrixCUDA output = concatenated.multiplyGPU(W_O);
    
    // Add bias
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b_O.get(0, j));
        }
    }
    
    return output;
}

MatrixCUDA MultiHeadAttentionCUDA::backward(const MatrixCUDA& grad_output) {
    size_t seq_len = grad_output.getRows();
    
    // Backward through output projection
    // dL/dW_O = concatenated^T * grad_output
    MatrixCUDA concatenated(seq_len, d_model);
    for (size_t i = 0; i < seq_len; i++) {
        size_t col_offset = 0;
        for (size_t h = 0; h < num_heads; h++) {
            MatrixCUDA head_output = attention_weights_cache[h].multiplyGPU(V_cache[h]);
            for (size_t j = 0; j < d_k; j++) {
                concatenated.set(i, col_offset + j, head_output.get(i, j));
            }
            col_offset += d_k;
        }
    }
    
    MatrixCUDA concatenated_T = concatenated.transposeGPU();
    grad_W_O = concatenated_T.multiplyGPU(grad_output);
    
    // dL/db_O = sum(grad_output) across sequence
    grad_b_O = MatrixCUDA(1, d_model);
    for (size_t j = 0; j < d_model; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < seq_len; i++) {
            sum += grad_output.get(i, j);
        }
        grad_b_O.set(0, j, sum);
    }
    
    // dL/dconcatenated = grad_output * W_O^T
    MatrixCUDA W_O_T = W_O.transposeGPU();
    MatrixCUDA grad_concatenated = grad_output.multiplyGPU(W_O_T);
    
    // Split gradients for each head
    MatrixCUDA grad_input(seq_len, d_model);
    grad_input.fill(0.0);
    
    for (size_t h = 0; h < num_heads; h++) {
        // Extract gradient for this head
        MatrixCUDA grad_head_output(seq_len, d_k);
        size_t col_offset = h * d_k;
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_k; j++) {
                grad_head_output.set(i, j, grad_concatenated.get(i, col_offset + j));
            }
        }
        
        // Backward through attention application: head_output = attention_weights * V
        // dL/dattention_weights = grad_head_output * V^T
        MatrixCUDA V_T = V_cache[h].transposeGPU();
        MatrixCUDA grad_attention_weights = grad_head_output.multiplyGPU(V_T);
        
        // dL/dV = attention_weights^T * grad_head_output
        MatrixCUDA attention_T = attention_weights_cache[h].transposeGPU();
        MatrixCUDA grad_V = attention_T.multiplyGPU(grad_head_output);
        
        // Backward through softmax - simplified (use gradient as is for scores)
        MatrixCUDA grad_scores = grad_attention_weights;
        
        // Backward through scaling
        float scale = 1.0f / sqrtf(static_cast<float>(d_k));
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                grad_scores.set(i, j, grad_scores.get(i, j) * scale);
            }
        }
        
        // Backward through scores = Q * K^T
        // dL/dQ = grad_scores * K
        MatrixCUDA grad_Q = grad_scores.multiplyGPU(K_cache[h]);
        
        // dL/dK = grad_scores^T * Q
        MatrixCUDA grad_scores_T = grad_scores.transposeGPU();
        MatrixCUDA grad_K = grad_scores_T.multiplyGPU(Q_cache[h]);
        
        // Gradients for weight matrices
        // dL/dW_Q = input^T * grad_Q
        MatrixCUDA input_T = input_cache.transposeGPU();
        grad_W_Q[h] = input_T.multiplyGPU(grad_Q);
        
        // dL/dW_K = input^T * grad_K
        grad_W_K[h] = input_T.multiplyGPU(grad_K);
        
        // dL/dW_V = input^T * grad_V
        grad_W_V[h] = input_T.multiplyGPU(grad_V);
        
        // Accumulate gradients w.r.t. input
        // dL/dinput += grad_Q * W_Q^T + grad_K * W_K^T + grad_V * W_V^T
        MatrixCUDA W_Q_T = W_Q[h].transposeGPU();
        MatrixCUDA W_K_T = W_K[h].transposeGPU();
        MatrixCUDA W_V_T = W_V[h].transposeGPU();
        
        MatrixCUDA grad_input_Q = grad_Q.multiplyGPU(W_Q_T);
        MatrixCUDA grad_input_K = grad_K.multiplyGPU(W_K_T);
        MatrixCUDA grad_input_V = grad_V.multiplyGPU(W_V_T);
        
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < d_model; j++) {
                double val = grad_input.get(i, j);
                val += grad_input_Q.get(i, j) + grad_input_K.get(i, j) + grad_input_V.get(i, j);
                grad_input.set(i, j, val);
            }
        }
    }
    
    return grad_input;
}

void MultiHeadAttentionCUDA::updateParameters(float learning_rate) {
    // Update W_Q, W_K, W_V for each head
    for (size_t h = 0; h < num_heads; h++) {
        // Update W_Q[h]
        for (size_t i = 0; i < W_Q[h].getRows(); i++) {
            for (size_t j = 0; j < W_Q[h].getCols(); j++) {
                double w = W_Q[h].get(i, j);
                double dw = grad_W_Q[h].get(i, j);
                W_Q[h].set(i, j, w - learning_rate * dw);
            }
        }
        
        // Update W_K[h]
        for (size_t i = 0; i < W_K[h].getRows(); i++) {
            for (size_t j = 0; j < W_K[h].getCols(); j++) {
                double w = W_K[h].get(i, j);
                double dw = grad_W_K[h].get(i, j);
                W_K[h].set(i, j, w - learning_rate * dw);
            }
        }
        
        // Update W_V[h]
        for (size_t i = 0; i < W_V[h].getRows(); i++) {
            for (size_t j = 0; j < W_V[h].getCols(); j++) {
                double w = W_V[h].get(i, j);
                double dw = grad_W_V[h].get(i, j);
                W_V[h].set(i, j, w - learning_rate * dw);
            }
        }
    }
    
    // Update W_O
    for (size_t i = 0; i < W_O.getRows(); i++) {
        for (size_t j = 0; j < W_O.getCols(); j++) {
            double w = W_O.get(i, j);
            double dw = grad_W_O.get(i, j);
            W_O.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update b_O
    for (size_t j = 0; j < b_O.getCols(); j++) {
        double b = b_O.get(0, j);
        double db = grad_b_O.get(0, j);
        b_O.set(0, j, b - learning_rate * db);
    }
}

// ============================================================================
// FEED-FORWARD NETWORK IMPLEMENTATION
// ============================================================================

FeedForwardCUDA::FeedForwardCUDA(size_t d_model, size_t d_ff)
    : d_model(d_model), d_ff(d_ff)
{
    W1 = MatrixCUDA(d_model, d_ff);
    W1.randomNormal(0.0f, 0.02f);
    b1 = MatrixCUDA(1, d_ff);
    b1.fill(0.0f);
    
    W2 = MatrixCUDA(d_ff, d_model);
    W2.randomNormal(0.0f, 0.02f);
    b2 = MatrixCUDA(1, d_model);
    b2.fill(0.0f);
    
    grad_W1 = MatrixCUDA(d_model, d_ff);
    grad_b1 = MatrixCUDA(1, d_ff);
    grad_W2 = MatrixCUDA(d_ff, d_model);
    grad_b2 = MatrixCUDA(1, d_model);
}

MatrixCUDA FeedForwardCUDA::forward(const MatrixCUDA& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    // First layer
    MatrixCUDA hidden = input.multiplyGPU(W1);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            hidden.set(i, j, hidden.get(i, j) + b1.get(0, j));
        }
    }
    
    // ReLU
    hidden_cache = MatrixCUDA(seq_len, d_ff);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            double val = hidden.get(i, j);
            hidden_cache.set(i, j, val > 0 ? val : 0.0);
        }
    }
    
    // Second layer
    MatrixCUDA output = hidden_cache.multiplyGPU(W2);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b2.get(0, j));
        }
    }
    
    return output;
}

MatrixCUDA FeedForwardCUDA::backward(const MatrixCUDA& grad_output) {
    // grad_output shape: (seq_len, d_model)
    // hidden_cache shape: (seq_len, d_ff) - after ReLU
    // input_cache shape: (seq_len, d_model)
    
    size_t seq_len = grad_output.getRows();
    
    // Backward through second layer
    // dL/dW2 = hidden^T * grad_output
    MatrixCUDA hidden_T = hidden_cache.transposeGPU();
    grad_W2 = hidden_T.multiplyGPU(grad_output);
    
    // dL/db2 = sum(grad_output) across batch
    grad_b2 = MatrixCUDA(1, d_model);
    for (size_t j = 0; j < d_model; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < seq_len; i++) {
            sum += grad_output.get(i, j);
        }
        grad_b2.set(0, j, sum);
    }
    
    // dL/dhidden = grad_output * W2^T
    MatrixCUDA W2_T = W2.transposeGPU();
    MatrixCUDA grad_hidden = grad_output.multiplyGPU(W2_T);
    
    // Backward through ReLU
    MatrixCUDA grad_hidden_pre_relu(seq_len, d_ff);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            // ReLU derivative: 1 if x > 0, else 0
            double val = hidden_cache.get(i, j);
            grad_hidden_pre_relu.set(i, j, val > 0 ? grad_hidden.get(i, j) : 0.0);
        }
    }
    
    // Backward through first layer
    // dL/dW1 = input^T * grad_hidden_pre_relu
    MatrixCUDA input_T = input_cache.transposeGPU();
    grad_W1 = input_T.multiplyGPU(grad_hidden_pre_relu);
    
    // dL/db1 = sum(grad_hidden_pre_relu) across batch
    grad_b1 = MatrixCUDA(1, d_ff);
    for (size_t j = 0; j < d_ff; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < seq_len; i++) {
            sum += grad_hidden_pre_relu.get(i, j);
        }
        grad_b1.set(0, j, sum);
    }
    
    // dL/dinput = grad_hidden_pre_relu * W1^T
    MatrixCUDA W1_T = W1.transposeGPU();
    MatrixCUDA grad_input = grad_hidden_pre_relu.multiplyGPU(W1_T);
    
    return grad_input;
}

void FeedForwardCUDA::updateParameters(float learning_rate) {
    // Update W1: W1 = W1 - learning_rate * grad_W1
    for (size_t i = 0; i < W1.getRows(); i++) {
        for (size_t j = 0; j < W1.getCols(); j++) {
            double w = W1.get(i, j);
            double dw = grad_W1.get(i, j);
            W1.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update b1: b1 = b1 - learning_rate * grad_b1
    for (size_t j = 0; j < b1.getCols(); j++) {
        double b = b1.get(0, j);
        double db = grad_b1.get(0, j);
        b1.set(0, j, b - learning_rate * db);
    }
    
    // Update W2: W2 = W2 - learning_rate * grad_W2
    for (size_t i = 0; i < W2.getRows(); i++) {
        for (size_t j = 0; j < W2.getCols(); j++) {
            double w = W2.get(i, j);
            double dw = grad_W2.get(i, j);
            W2.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update b2: b2 = b2 - learning_rate * grad_b2
    for (size_t j = 0; j < b2.getCols(); j++) {
        double b = b2.get(0, j);
        double db = grad_b2.get(0, j);
        b2.set(0, j, b - learning_rate * db);
    }
}

// ============================================================================
// TRANSFORMER ENCODER LAYER IMPLEMENTATION
// ============================================================================

TransformerEncoderLayerCUDA::TransformerEncoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff)
{
    attention = std::make_unique<MultiHeadAttentionCUDA>(d_model, num_heads);
    ffn = std::make_unique<FeedForwardCUDA>(d_model, d_ff);
    norm1 = std::make_unique<LayerNormCUDA>(d_model);
    norm2 = std::make_unique<LayerNormCUDA>(d_model);
}

MatrixCUDA TransformerEncoderLayerCUDA::forward(const MatrixCUDA& input) {
    residual1_cache = input;
    MatrixCUDA attn_out = attention->forward(input);
    MatrixCUDA add1 = attn_out + residual1_cache;
    MatrixCUDA norm1_out = norm1->forward(add1);
    
    residual2_cache = norm1_out;
    MatrixCUDA ffn_out = ffn->forward(norm1_out);
    MatrixCUDA add2 = ffn_out + residual2_cache;
    MatrixCUDA output = norm2->forward(add2);
    
    return output;
}

MatrixCUDA TransformerEncoderLayerCUDA::backward(const MatrixCUDA& grad_output) {
    // Backward through norm2
    MatrixCUDA grad_add2 = norm2->backward(grad_output);
    
    // Backward through residual connection 2
    MatrixCUDA grad_ffn_out = grad_add2;
    MatrixCUDA grad_norm1_out = grad_add2;
    
    // Backward through FFN
    MatrixCUDA grad_ffn_input = ffn->backward(grad_ffn_out);
    
    // Add gradient from residual
    for (size_t i = 0; i < grad_norm1_out.getRows(); i++) {
        for (size_t j = 0; j < grad_norm1_out.getCols(); j++) {
            grad_norm1_out.set(i, j, grad_norm1_out.get(i, j) + grad_ffn_input.get(i, j));
        }
    }
    
    // Backward through norm1
    MatrixCUDA grad_add1 = norm1->backward(grad_norm1_out);
    
    // Backward through residual connection 1
    MatrixCUDA grad_attn_out = grad_add1;
    MatrixCUDA grad_input = grad_add1;
    
    // Backward through attention
    MatrixCUDA grad_attn_input = attention->backward(grad_attn_out);
    
    // Add gradient from residual
    for (size_t i = 0; i < grad_input.getRows(); i++) {
        for (size_t j = 0; j < grad_input.getCols(); j++) {
            grad_input.set(i, j, grad_input.get(i, j) + grad_attn_input.get(i, j));
        }
    }
    
    return grad_input;
}

void TransformerEncoderLayerCUDA::updateParameters(float learning_rate) {
    attention->updateParameters(learning_rate);
    ffn->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
}

// ============================================================================
// BERT ENCODER IMPLEMENTATION
// ============================================================================

BERTEncoderCUDA::BERTEncoderCUDA(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers)
    : d_model(d_model), num_layers(num_layers)
{
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<TransformerEncoderLayerCUDA>(d_model, num_heads, d_ff));
    }
}

MatrixCUDA BERTEncoderCUDA::forward(const MatrixCUDA& input) {
    layer_outputs.clear();
    layer_outputs.push_back(input);
    
    MatrixCUDA current = input;
    for (size_t i = 0; i < num_layers; i++) {
        current = layers[i]->forward(current);
        layer_outputs.push_back(current);
    }
    
    return current;
}

MatrixCUDA BERTEncoderCUDA::backward(const MatrixCUDA& grad_output) {
    // Backward through layers in reverse order
    MatrixCUDA current_grad = grad_output;
    
    for (int i = num_layers - 1; i >= 0; i--) {
        current_grad = layers[i]->backward(current_grad);
    }
    
    return current_grad;
}

void BERTEncoderCUDA::updateParameters(float learning_rate) {
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->updateParameters(learning_rate);
    }
}

// ============================================================================
// BERT EMBEDDING IMPLEMENTATION
// ============================================================================

BERTEmbeddingCUDA::BERTEmbeddingCUDA(size_t vocab_size, size_t d_model, size_t max_seq_length)
    : vocab_size(vocab_size), d_model(d_model), max_seq_length(max_seq_length)
{
    token_embeddings = MatrixCUDA(vocab_size, d_model);
    token_embeddings.randomNormal(0.0f, 0.02f);
    
    positional_encodings = MatrixCUDA(max_seq_length, d_model);
    initializePositionalEncoding();
    
    grad_token_embeddings = MatrixCUDA(vocab_size, d_model);
    grad_token_embeddings.fill(0.0f);
}

void BERTEmbeddingCUDA::initializePositionalEncoding() {
    for (size_t pos = 0; pos < max_seq_length; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            float angle = pos / powf(10000.0f, (2.0f * i) / d_model);
            if (i % 2 == 0) {
                positional_encodings.set(pos, i, sinf(angle));
            } else {
                positional_encodings.set(pos, i, cosf(angle));
            }
        }
    }
}

MatrixCUDA BERTEmbeddingCUDA::forward(const std::vector<int>& token_ids) {
    token_ids_cache = token_ids;
    size_t seq_len = token_ids.size();
    
    MatrixCUDA output(seq_len, d_model);
    
    for (size_t i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        if (token_id >= 0 && token_id < static_cast<int>(vocab_size)) {
            for (size_t j = 0; j < d_model; j++) {
                double token_emb = token_embeddings.get(token_id, j);
                double pos_enc = positional_encodings.get(i, j);
                output.set(i, j, token_emb + pos_enc);
            }
        }
    }
    
    return output;
}

void BERTEmbeddingCUDA::backward(const MatrixCUDA& grad_output) {
    // Accumulate gradients for token embeddings
    grad_token_embeddings.fill(0.0);
    
    size_t seq_len = token_ids_cache.size();
    for (size_t i = 0; i < seq_len; i++) {
        int token_id = token_ids_cache[i];
        if (token_id >= 0 && token_id < static_cast<int>(vocab_size)) {
            for (size_t j = 0; j < d_model; j++) {
                double grad = grad_token_embeddings.get(token_id, j);
                grad += grad_output.get(i, j);
                grad_token_embeddings.set(token_id, j, grad);
            }
        }
    }
    // Note: Positional encodings are not learned, so no gradient needed
}

void BERTEmbeddingCUDA::updateParameters(float learning_rate) {
    // Update token embeddings: embeddings = embeddings - learning_rate * gradients
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < d_model; j++) {
            double emb = token_embeddings.get(i, j);
            double grad = grad_token_embeddings.get(i, j);
            token_embeddings.set(i, j, emb - learning_rate * grad);
        }
    }
}

// ============================================================================
// OUTPUT HEADS IMPLEMENTATION
// ============================================================================

IntentClassifierCUDA::IntentClassifierCUDA(size_t d_model, size_t num_intents)
    : d_model(d_model), num_intents(num_intents)
{
    W = MatrixCUDA(d_model, num_intents);
    W.randomNormal(0.0f, 0.02f);
    b = MatrixCUDA(1, num_intents);
    b.fill(0.0f);
    
    grad_W = MatrixCUDA(d_model, num_intents);
    grad_b = MatrixCUDA(1, num_intents);
}

MatrixCUDA IntentClassifierCUDA::forward(const MatrixCUDA& cls_representation) {
    input_cache = cls_representation;
    MatrixCUDA logits = cls_representation.multiplyGPU(W);
    
    for (size_t j = 0; j < num_intents; j++) {
        logits.set(0, j, logits.get(0, j) + b.get(0, j));
    }
    
    return logits;
}

MatrixCUDA IntentClassifierCUDA::backward(const MatrixCUDA& grad_output) {
    // Gradient w.r.t. weights: dL/dW = input^T * grad_output
    MatrixCUDA input_T = input_cache.transposeGPU();
    grad_W = input_T.multiplyGPU(grad_output);
    
    // Gradient w.r.t. bias: dL/db = sum(grad_output) across batch
    grad_b = grad_output;
    
    // Gradient w.r.t. input: dL/dinput = grad_output * W^T
    MatrixCUDA W_T = W.transposeGPU();
    MatrixCUDA grad_input = grad_output.multiplyGPU(W_T);
    
    return grad_input;
}

void IntentClassifierCUDA::updateParameters(float learning_rate) {
    // Update weights: W = W - learning_rate * grad_W
    for (size_t i = 0; i < W.getRows(); i++) {
        for (size_t j = 0; j < W.getCols(); j++) {
            double w = W.get(i, j);
            double dw = grad_W.get(i, j);
            W.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update biases: b = b - learning_rate * grad_b
    for (size_t j = 0; j < b.getCols(); j++) {
        double bias = b.get(0, j);
        double db = grad_b.get(0, j);
        b.set(0, j, bias - learning_rate * db);
    }
}

SlotTaggerCUDA::SlotTaggerCUDA(size_t d_model, size_t num_slots)
    : d_model(d_model), num_slots(num_slots)
{
    W = MatrixCUDA(d_model, num_slots);
    W.randomNormal(0.0, 0.02);
    b = MatrixCUDA(1, num_slots);
    b.fill(0.0);
    
    grad_W = MatrixCUDA(d_model, num_slots);
    grad_b = MatrixCUDA(1, num_slots);
}

MatrixCUDA SlotTaggerCUDA::forward(const MatrixCUDA& sequence_representations) {
    input_cache = sequence_representations;
    size_t seq_len = sequence_representations.getRows();
    
    MatrixCUDA logits = sequence_representations.multiplyGPU(W);
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_slots; j++) {
            logits.set(i, j, logits.get(i, j) + b.get(0, j));
        }
    }
    
    return logits;
}

MatrixCUDA SlotTaggerCUDA::backward(const MatrixCUDA& grad_output) {
    // grad_output shape: (seq_len, num_slots)
    // input_cache shape: (seq_len, d_model)
    // W shape: (d_model, num_slots)
    
    // Gradient w.r.t. weights: dL/dW = input^T * grad_output
    MatrixCUDA input_T = input_cache.transposeGPU();
    grad_W = input_T.multiplyGPU(grad_output);
    
    // Gradient w.r.t. bias: sum across sequence dimension
    grad_b = MatrixCUDA(1, num_slots);
    for (size_t j = 0; j < num_slots; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_output.getRows(); i++) {
            sum += grad_output.get(i, j);
        }
        grad_b.set(0, j, sum);
    }
    
    // Gradient w.r.t. input: dL/dinput = grad_output * W^T
    MatrixCUDA W_T = W.transposeGPU();
    MatrixCUDA grad_input = grad_output.multiplyGPU(W_T);
    
    return grad_input;
}

void SlotTaggerCUDA::updateParameters(float learning_rate) {
    // Update weights: W = W - learning_rate * grad_W
    for (size_t i = 0; i < W.getRows(); i++) {
        for (size_t j = 0; j < W.getCols(); j++) {
            double w = W.get(i, j);
            double dw = grad_W.get(i, j);
            W.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update biases: b = b - learning_rate * grad_b
    for (size_t j = 0; j < b.getCols(); j++) {
        double bias = b.get(0, j);
        double db = grad_b.get(0, j);
        b.set(0, j, bias - learning_rate * db);
    }
}

EntityDetectorCUDA::EntityDetectorCUDA(size_t d_model, size_t num_entities)
    : d_model(d_model), num_entities(num_entities)
{
    W = MatrixCUDA(d_model, num_entities);
    W.randomNormal(0.0, 0.02);
    b = MatrixCUDA(1, num_entities);
    b.fill(0.0);
    
    grad_W = MatrixCUDA(d_model, num_entities);
    grad_b = MatrixCUDA(1, num_entities);
}

MatrixCUDA EntityDetectorCUDA::forward(const MatrixCUDA& sequence_representations) {
    input_cache = sequence_representations;
    size_t seq_len = sequence_representations.getRows();
    
    MatrixCUDA logits = sequence_representations.multiplyGPU(W);
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_entities; j++) {
            logits.set(i, j, logits.get(i, j) + b.get(0, j));
        }
    }
    
    return logits;
}

MatrixCUDA EntityDetectorCUDA::backward(const MatrixCUDA& grad_output) {
    // grad_output shape: (seq_len, num_entities)
    // input_cache shape: (seq_len, d_model)
    // W shape: (d_model, num_entities)
    
    // Gradient w.r.t. weights: dL/dW = input^T * grad_output
    MatrixCUDA input_T = input_cache.transposeGPU();
    grad_W = input_T.multiplyGPU(grad_output);
    
    // Gradient w.r.t. bias: sum across sequence dimension
    grad_b = MatrixCUDA(1, num_entities);
    for (size_t j = 0; j < num_entities; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_output.getRows(); i++) {
            sum += grad_output.get(i, j);
        }
        grad_b.set(0, j, sum);
    }
    
    // Gradient w.r.t. input: dL/dinput = grad_output * W^T
    MatrixCUDA W_T = W.transposeGPU();
    MatrixCUDA grad_input = grad_output.multiplyGPU(W_T);
    
    return grad_input;
}

void EntityDetectorCUDA::updateParameters(float learning_rate) {
    // Update weights: W = W - learning_rate * grad_W
    for (size_t i = 0; i < W.getRows(); i++) {
        for (size_t j = 0; j < W.getCols(); j++) {
            double w = W.get(i, j);
            double dw = grad_W.get(i, j);
            W.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update biases: b = b - learning_rate * grad_b
    for (size_t j = 0; j < b.getCols(); j++) {
        double bias = b.get(0, j);
        double db = grad_b.get(0, j);
        b.set(0, j, bias - learning_rate * db);
    }
}

// ============================================================================
// COMPLETE BERT-NLU MODEL IMPLEMENTATION
// ============================================================================

BERTForNLUCUDA::BERTForNLUCUDA(size_t vocab_size, size_t d_model, size_t num_heads,
                               size_t d_ff, size_t num_layers, size_t max_seq_length,
                               size_t num_intents, size_t num_slots, size_t num_entities)
    : d_model(d_model), vocab_size(vocab_size),
      num_intents(num_intents), num_slots(num_slots), num_entities(num_entities)
{
    embedding = std::make_unique<BERTEmbeddingCUDA>(vocab_size, d_model, max_seq_length);
    encoder = std::make_unique<BERTEncoderCUDA>(d_model, num_heads, d_ff, num_layers);
    intent_head = std::make_unique<IntentClassifierCUDA>(d_model, num_intents);
    slot_head = std::make_unique<SlotTaggerCUDA>(d_model, num_slots);
    entity_head = std::make_unique<EntityDetectorCUDA>(d_model, num_entities);
}

std::tuple<MatrixCUDA, MatrixCUDA, MatrixCUDA> BERTForNLUCUDA::forward(const std::vector<int>& token_ids) {
    MatrixCUDA embedded = embedding->forward(token_ids);
    encoder_output_cache = encoder->forward(embedded);
    
    // Extract [CLS] token
    MatrixCUDA cls_token(1, d_model);
    for (size_t j = 0; j < d_model; j++) {
        cls_token.set(0, j, encoder_output_cache.get(0, j));
    }
    
    MatrixCUDA intent_logits = intent_head->forward(cls_token);
    MatrixCUDA slot_logits = slot_head->forward(encoder_output_cache);
    MatrixCUDA entity_logits = entity_head->forward(encoder_output_cache);
    
    return std::make_tuple(intent_logits, slot_logits, entity_logits);
}

void BERTForNLUCUDA::backward(const MatrixCUDA& grad_intent, const MatrixCUDA& grad_slots,
                              const MatrixCUDA& grad_entities) {
    // Backward through task heads
    MatrixCUDA grad_from_intent = intent_head->backward(grad_intent);
    MatrixCUDA grad_from_slots = slot_head->backward(grad_slots);
    MatrixCUDA grad_from_entities = entity_head->backward(grad_entities);
    
    // Combine gradients from all heads
    // Intent head only affects [CLS] token (first position)
    // Slot and entity heads affect all positions
    MatrixCUDA grad_encoder_output(encoder_output_cache.getRows(), encoder_output_cache.getCols());
    grad_encoder_output.fill(0.0);
    
    // Add gradient from intent head to [CLS] token
    for (size_t j = 0; j < d_model; j++) {
        grad_encoder_output.set(0, j, grad_from_intent.get(0, j));
    }
    
    // Add gradients from slot and entity heads
    for (size_t i = 0; i < grad_encoder_output.getRows(); i++) {
        for (size_t j = 0; j < grad_encoder_output.getCols(); j++) {
            double grad = grad_encoder_output.get(i, j);
            grad += grad_from_slots.get(i, j) + grad_from_entities.get(i, j);
            grad_encoder_output.set(i, j, grad);
        }
    }
    
    // Backward through encoder
    MatrixCUDA grad_embedded = encoder->backward(grad_encoder_output);
    
    // Backward through embedding
    embedding->backward(grad_embedded);
}

void BERTForNLUCUDA::updateParameters(float learning_rate) {
    embedding->updateParameters(learning_rate);
    encoder->updateParameters(learning_rate);
    intent_head->updateParameters(learning_rate);
    slot_head->updateParameters(learning_rate);
    entity_head->updateParameters(learning_rate);
}

std::tuple<int, std::vector<int>, std::vector<int>> BERTForNLUCUDA::predict(const std::vector<int>& token_ids) {
    auto [intent_logits, slot_logits, entity_logits] = forward(token_ids);
    
    // Intent argmax
    int predicted_intent = 0;
    double max_intent = intent_logits.get(0, 0);
    for (size_t i = 1; i < num_intents; i++) {
        if (intent_logits.get(0, i) > max_intent) {
            max_intent = intent_logits.get(0, i);
            predicted_intent = i;
        }
    }
    
    // Slots argmax
    std::vector<int> predicted_slots;
    size_t seq_len = slot_logits.getRows();
    for (size_t i = 0; i < seq_len; i++) {
        int slot_id = 0;
        double max_slot = slot_logits.get(i, 0);
        for (size_t j = 1; j < num_slots; j++) {
            if (slot_logits.get(i, j) > max_slot) {
                max_slot = slot_logits.get(i, j);
                slot_id = j;
            }
        }
        predicted_slots.push_back(slot_id);
    }
    
    // Entities argmax
    std::vector<int> predicted_entities;
    for (size_t i = 0; i < seq_len; i++) {
        int entity_id = 0;
        double max_entity = entity_logits.get(i, 0);
        for (size_t j = 1; j < num_entities; j++) {
            if (entity_logits.get(i, j) > max_entity) {
                max_entity = entity_logits.get(i, j);
                entity_id = j;
            }
        }
        predicted_entities.push_back(entity_id);
    }
    
    return std::make_tuple(predicted_intent, predicted_slots, predicted_entities);
}
