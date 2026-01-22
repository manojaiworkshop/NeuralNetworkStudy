#ifndef ATTENTION_H
#define ATTENTION_H

#include "../matrix.h"
#include <vector>
#include <memory>
#include <cmath>

/**
 * @brief Scaled Dot-Product Attention
 * 
 * Attention(Q, K, V) = softmax(QK^T / √d_k) V
 * 
 * Where:
 * - Q: Query matrix
 * - K: Key matrix
 * - V: Value matrix
 * - d_k: Dimension of keys
 */
class ScaledDotProductAttention {
private:
    double dropout_rate;
    double scale_factor;  // 1 / √d_k
    
    // Cached for backward pass
    Matrix cached_attention_weights;
    Matrix cached_Q, cached_K, cached_V;
    Matrix dropout_mask;
    
public:
    /**
     * @brief Constructor
     * @param d_k Dimension of keys
     * @param dropout Dropout rate for attention weights
     */
    ScaledDotProductAttention(size_t d_k, double dropout = 0.1);
    
    /**
     * @brief Forward pass
     * @param Q Query matrix (batch × seq_len_q × d_k)
     * @param K Key matrix (batch × seq_len_k × d_k)
     * @param V Value matrix (batch × seq_len_v × d_v)
     * @param mask Optional attention mask (1=attend, 0=mask)
     * @param training Whether in training mode (for dropout)
     * @return Output (batch × seq_len_q × d_v)
     */
    Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V,
                   const Matrix* mask = nullptr, bool training = true);
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @param dQ, dK, dV Output gradients for Q, K, V
     */
    void backward(const Matrix& grad_output,
                  Matrix& dQ, Matrix& dK, Matrix& dV);
    
    /**
     * @brief Get attention weights (for visualization)
     */
    const Matrix& getAttentionWeights() const { return cached_attention_weights; }
};

/**
 * @brief Multi-Head Attention
 * 
 * MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
 * where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
 * 
 * Allows model to attend to different representation subspaces
 */
class MultiHeadAttention {
private:
    size_t d_model;      // Model dimension
    size_t num_heads;    // Number of attention heads
    size_t d_k;          // d_model / num_heads (key/query dim)
    size_t d_v;          // d_model / num_heads (value dim)
    double dropout_rate;
    
    // Projection matrices
    std::vector<Matrix> W_Q;  // Query projections (num_heads × d_model × d_k)
    std::vector<Matrix> W_K;  // Key projections
    std::vector<Matrix> W_V;  // Value projections
    Matrix W_O;               // Output projection (d_model × d_model)
    
    // Gradients
    std::vector<Matrix> dW_Q, dW_K, dW_V;
    Matrix dW_O;
    
    // Attention mechanisms for each head
    std::vector<std::unique_ptr<ScaledDotProductAttention>> attention_heads;
    
    // Cached for backward pass
    std::vector<Matrix> cached_heads;
    Matrix cached_Q, cached_K, cached_V;
    Matrix cached_concat;
    
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param dropout Dropout rate
     */
    MultiHeadAttention(size_t d_model, size_t num_heads, double dropout = 0.1);
    
    /**
     * @brief Initialize projection matrices
     */
    void initializeWeights();
    
    /**
     * @brief Forward pass
     * @param Q Query matrix (batch × seq_len_q × d_model)
     * @param K Key matrix (batch × seq_len_k × d_model)
     * @param V Value matrix (batch × seq_len_v × d_model)
     * @param mask Optional attention mask
     * @param training Whether in training mode
     * @return Output (batch × seq_len_q × d_model)
     */
    Matrix forward(const Matrix& Q, const Matrix& K, const Matrix& V,
                   const Matrix* mask = nullptr, bool training = true);
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @param dQ, dK, dV Output gradients
     */
    void backward(const Matrix& grad_output,
                  Matrix& dQ, Matrix& dK, Matrix& dV);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get attention weights from all heads (for visualization)
     */
    std::vector<Matrix> getAllAttentionWeights() const;
    
    int getParameterCount() const {
        return 4 * d_model * d_model;  // Q, K, V, O projections
    }
    
    /**
     * @brief Save weights to binary stream
     */
    void saveWeights(std::ofstream& out) const;
    
    /**
     * @brief Load weights from binary stream
     */
    void loadWeights(std::ifstream& in);
};

/**
 * @brief Create causal mask for decoder (prevent attending to future)
 * 
 * Mask[i][j] = 1 if i >= j (can attend)
 *            = 0 if i < j  (mask out future)
 */
Matrix createCausalMask(size_t seq_len);

/**
 * @brief Create padding mask
 * 
 * Mask out padding tokens (typically token_id = 0)
 */
Matrix createPaddingMask(const std::vector<std::vector<int>>& token_ids, 
                         int pad_token_id = 0);

#endif // ATTENTION_H
