#ifndef ATTENTION_CUDA_H
#define ATTENTION_CUDA_H

#include "matrix_cuda.h"
#include "activation_cuda.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

/**
 * @brief CUDA-accelerated Scaled Dot-Product Attention
 * 
 * GPU-optimized attention: Attention(Q,K,V) = softmax(QK^T / √d_k) × V
 * 
 * PERFORMANCE:
 * - 10-50x faster than CPU for long sequences
 * - Parallel softmax across batch and heads
 * - Fused attention kernels
 */
class ScaledDotProductAttentionCUDA {
private:
    size_t d_k;
    double scale_factor;
    
    // Device memory for attention computations
    float* d_scores;
    float* d_attention_weights;
    
    // Cached for backward pass
    MatrixCUDA cached_Q, cached_K, cached_V;
    MatrixCUDA cached_attention_weights;
    
    bool gpu_allocated;
    
    void allocateGPU(size_t max_seq_len);
    void freeGPU();
    
public:
    ScaledDotProductAttentionCUDA(size_t d_k);
    ~ScaledDotProductAttentionCUDA();
    
    /**
     * @brief Forward pass on GPU
     */
    MatrixCUDA forward(const MatrixCUDA& Q, const MatrixCUDA& K, 
                       const MatrixCUDA& V);
    
    /**
     * @brief Backward pass
     */
    void backward(const MatrixCUDA& grad_output,
                  MatrixCUDA& dQ, MatrixCUDA& dK, MatrixCUDA& dV);
    
    const MatrixCUDA& getAttentionWeights() const { 
        return cached_attention_weights; 
    }
};

/**
 * @brief CUDA-accelerated Multi-Head Attention
 * 
 * Parallel attention heads with GPU acceleration
 * Each head processes independently → perfect for GPU parallelism
 */
class MultiHeadAttentionCUDA {
private:
    size_t d_model;
    size_t num_heads;
    size_t d_k;
    size_t d_v;
    
    // Projection matrices on GPU
    std::vector<MatrixCUDA> W_Q;
    std::vector<MatrixCUDA> W_K;
    std::vector<MatrixCUDA> W_V;
    MatrixCUDA W_O;
    
    // Gradients
    std::vector<MatrixCUDA> dW_Q, dW_K, dW_V;
    MatrixCUDA dW_O;
    
    // Attention heads
    std::vector<std::unique_ptr<ScaledDotProductAttentionCUDA>> attention_heads;
    
    // Cached for backward
    std::vector<MatrixCUDA> cached_heads;
    MatrixCUDA cached_Q, cached_K, cached_V;
    MatrixCUDA cached_concat;
    
public:
    MultiHeadAttentionCUDA(size_t d_model, size_t num_heads);
    
    void initializeWeights();
    
    /**
     * @brief Forward pass - all heads in parallel on GPU
     */
    MatrixCUDA forward(const MatrixCUDA& Q, const MatrixCUDA& K, 
                       const MatrixCUDA& V);
    
    /**
     * @brief Backward pass
     */
    void backward(const MatrixCUDA& grad_output,
                  MatrixCUDA& dQ, MatrixCUDA& dK, MatrixCUDA& dV);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get attention weights from all heads
     */
    std::vector<MatrixCUDA> getAllAttentionWeights() const;
    
    int getParameterCount() const {
        return 4 * d_model * d_model;
    }
};

/**
 * @brief CUDA-accelerated Position-wise Feed-Forward Network
 */
class FeedForwardCUDA {
private:
    size_t d_model;
    size_t d_ff;
    
    // Weights on GPU
    MatrixCUDA W1, b1, W2, b2;
    
    // Gradients
    MatrixCUDA dW1, db1, dW2, db2;
    
    // Activation
    std::unique_ptr<ActivationCUDA> activation;
    
    // Cached
    MatrixCUDA cached_input;
    MatrixCUDA cached_hidden;
    MatrixCUDA cached_hidden_pre;
    
public:
    FeedForwardCUDA(size_t d_model, size_t d_ff);
    
    void initializeWeights();
    
    /**
     * @brief Forward: ReLU(xW1 + b1)W2 + b2 on GPU
     */
    MatrixCUDA forward(const MatrixCUDA& input);
    
    /**
     * @brief Backward pass
     */
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
};

/**
 * @brief CUDA-accelerated Layer Normalization
 */
class LayerNormCUDA {
private:
    size_t normalized_shape;
    double epsilon;
    
    // Learnable parameters
    MatrixCUDA gamma;
    MatrixCUDA beta;
    
    // Gradients
    MatrixCUDA gamma_grad;
    MatrixCUDA beta_grad;
    
    // Cached
    MatrixCUDA cached_input;
    MatrixCUDA cached_mean;
    MatrixCUDA cached_std;
    MatrixCUDA cached_normalized;
    
public:
    LayerNormCUDA(size_t normalized_shape, double epsilon = 1e-6);
    
    /**
     * @brief Forward: LayerNorm(x) = γ(x-μ)/σ + β on GPU
     */
    MatrixCUDA forward(const MatrixCUDA& input);
    
    /**
     * @brief Backward pass
     */
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
};

/**
 * @brief CUDA-accelerated Transformer Encoder Layer
 */
class EncoderLayerCUDA {
private:
    size_t d_model;
    
    std::unique_ptr<MultiHeadAttentionCUDA> self_attention;
    std::unique_ptr<FeedForwardCUDA> feed_forward;
    std::unique_ptr<LayerNormCUDA> norm1;
    std::unique_ptr<LayerNormCUDA> norm2;
    
    // Cached
    MatrixCUDA cached_input;
    MatrixCUDA cached_attn_output;
    MatrixCUDA cached_ffn_input;
    
public:
    EncoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff);
    
    /**
     * @brief Forward: Self-Attention → Add&Norm → FFN → Add&Norm
     */
    MatrixCUDA forward(const MatrixCUDA& input);
    
    /**
     * @brief Backward pass
     */
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    std::vector<MatrixCUDA> getAttentionWeights() const;
};

/**
 * @brief CUDA-accelerated Transformer Encoder Stack
 */
class TransformerEncoderCUDA {
private:
    size_t num_layers;
    size_t d_model;
    
    std::vector<std::unique_ptr<EncoderLayerCUDA>> layers;
    std::unique_ptr<LayerNormCUDA> final_norm;
    
public:
    TransformerEncoderCUDA(size_t num_layers, size_t d_model, 
                           size_t num_heads, size_t d_ff);
    
    /**
     * @brief Forward through all encoder layers
     */
    MatrixCUDA forward(const MatrixCUDA& input);
    
    /**
     * @brief Backward through all layers
     */
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    std::vector<std::vector<MatrixCUDA>> getAllAttentionWeights() const;
};

/**
 * @brief CUDA-accelerated Token Embedding
 */
class TokenEmbeddingCUDA {
private:
    size_t vocab_size;
    size_t embedding_dim;
    
    MatrixCUDA embeddings;  // On GPU
    MatrixCUDA gradients;
    
public:
    TokenEmbeddingCUDA(size_t vocab_size, size_t embedding_dim);
    
    void initializeEmbeddings();
    
    /**
     * @brief Forward: lookup embeddings on GPU
     */
    MatrixCUDA forward(const std::vector<std::vector<int>>& token_ids);
    
    /**
     * @brief Backward: accumulate gradients
     */
    void backward(const MatrixCUDA& grad_output, 
                  const std::vector<std::vector<int>>& token_ids);
    
    /**
     * @brief Update embeddings
     */
    void updateParameters(double learning_rate);
    
    const MatrixCUDA& getEmbeddings() const { return embeddings; }
};

/**
 * @brief CUDA-accelerated Positional Encoding
 */
class PositionalEncodingCUDA {
private:
    size_t max_seq_len;
    size_t d_model;
    MatrixCUDA encoding;  // Pre-computed on GPU
    
public:
    PositionalEncodingCUDA(size_t max_seq_len, size_t d_model);
    
    /**
     * @brief Add positional encoding on GPU
     */
    MatrixCUDA forward(const MatrixCUDA& input);
};

#endif // ATTENTION_CUDA_H
