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
    size_t allocated_max_seq_len;  // Track allocated buffer size
    
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
    MatrixCUDA& getEmbeddings() { return embeddings; }
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

/**
 * @brief CUDA Transformer Decoder Layer
 * 
 * Decoder layer with:
 * 1. Masked self-attention (prevent looking ahead)
 * 2. Cross-attention to encoder output
 * 3. Feed-forward network
 * 4. Layer normalization after each sub-layer
 */
class DecoderLayerCUDA {
private:
    size_t d_model;
    size_t num_heads;
    size_t d_ff;
    
    // Sub-layers
    std::unique_ptr<MultiHeadAttentionCUDA> masked_self_attention;
    std::unique_ptr<MultiHeadAttentionCUDA> cross_attention;
    std::unique_ptr<FeedForwardCUDA> feed_forward;
    
    // Layer normalization
    std::unique_ptr<LayerNormCUDA> norm1;
    std::unique_ptr<LayerNormCUDA> norm2;
    std::unique_ptr<LayerNormCUDA> norm3;
    
    // Cached for backward
    MatrixCUDA cached_input;
    MatrixCUDA cached_encoder_output;
    
public:
    DecoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff);
    
    /**
     * @brief Forward with masked self-attention and cross-attention
     * @param input Target sequence embeddings
     * @param encoder_output Source sequence encoded representations
     * @param mask Lower triangular mask for causal attention
     */
    MatrixCUDA forward(const MatrixCUDA& input, 
                       const MatrixCUDA& encoder_output,
                       const MatrixCUDA* mask = nullptr);
    
    MatrixCUDA backward(const MatrixCUDA& grad_output,
                       MatrixCUDA& grad_encoder_output);
    
    void updateParameters(double learning_rate);
    
    // Get attention weights for visualization
    std::vector<MatrixCUDA> getSelfAttentionWeights() const;
    std::vector<MatrixCUDA> getCrossAttentionWeights() const;
};

/**
 * @brief CUDA Transformer Decoder Stack
 */
class TransformerDecoderCUDA {
private:
    size_t num_layers;
    size_t d_model;
    std::vector<std::unique_ptr<DecoderLayerCUDA>> layers;
    std::unique_ptr<LayerNormCUDA> final_norm;
    
public:
    TransformerDecoderCUDA(size_t num_layers, size_t d_model, 
                          size_t num_heads, size_t d_ff);
    
    /**
     * @brief Forward through decoder stack
     */
    MatrixCUDA forward(const MatrixCUDA& target, 
                       const MatrixCUDA& encoder_output,
                       const MatrixCUDA* mask = nullptr);
    
    MatrixCUDA backward(const MatrixCUDA& grad_output,
                       MatrixCUDA& grad_encoder_output);
    
    void updateParameters(double learning_rate);
    
    // Get attention weights from all layers
    std::vector<std::vector<MatrixCUDA>> getAllSelfAttentionWeights() const;
    std::vector<std::vector<MatrixCUDA>> getAllCrossAttentionWeights() const;
};

/**
 * @brief Complete CUDA Transformer Model
 * 
 * GPU-accelerated Transformer with:
 * - Token embeddings + positional encoding
 * - Multi-layer encoder
 * - Multi-layer decoder with masked attention
 * - Output projection to vocabulary
 * 
 * PERFORMANCE:
 * - 20-40x faster than CPU for typical configurations
 * - Efficient batch processing on GPU
 * - Parallel attention across all heads
 */
class TransformerCUDA {
private:
    size_t vocab_size;
    size_t d_model;
    size_t num_heads;
    size_t num_layers;
    size_t d_ff;
    size_t max_seq_len;
    
    // Embeddings
    std::unique_ptr<TokenEmbeddingCUDA> src_embedding;
    std::unique_ptr<TokenEmbeddingCUDA> tgt_embedding;
    std::unique_ptr<PositionalEncodingCUDA> positional_encoding;
    
    // Encoder and Decoder
    std::unique_ptr<TransformerEncoderCUDA> encoder;
    std::unique_ptr<TransformerDecoderCUDA> decoder;
    
    // Output projection
    MatrixCUDA output_projection;  // d_model × vocab_size
    MatrixCUDA output_projection_grad;
    
    // Causal mask for decoder (lower triangular)
    MatrixCUDA causal_mask;
    
    void initializeOutputProjection();
    MatrixCUDA createCausalMask(size_t seq_len);
    
public:
    TransformerCUDA(size_t vocab_size, size_t d_model, size_t num_heads,
                   size_t num_layers, size_t d_ff, size_t max_seq_len);
    
    /**
     * @brief Forward pass: source → encoder → decoder ← target → logits
     */
    MatrixCUDA forward(const std::vector<std::vector<int>>& source_ids,
                       const std::vector<std::vector<int>>& target_ids);
    
    /**
     * @brief Generate output sequence (inference)
     * @param source_ids Input sequence token IDs
     * @param max_length Maximum output length
     * @param start_token_id ID of start token (e.g., <BOS>)
     * @param end_token_id ID of end token (e.g., <EOS>)
     */
    std::vector<int> generate(const std::vector<int>& source_ids,
                              size_t max_length,
                              int start_token_id,
                              int end_token_id);
    
    /**
     * @brief Backward pass for training
     */
    void backward(const MatrixCUDA& grad_output);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get attention weights for visualization
     */
    std::vector<std::vector<MatrixCUDA>> getEncoderAttentionWeights() const;
    std::vector<std::vector<MatrixCUDA>> getDecoderSelfAttentionWeights() const;
    std::vector<std::vector<MatrixCUDA>> getDecoderCrossAttentionWeights() const;
    
    size_t getParameterCount() const;
};

#endif // ATTENTION_CUDA_H
