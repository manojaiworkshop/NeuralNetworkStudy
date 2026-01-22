#ifndef ATTENTION_H
#define ATTENTION_H

#include "matrix.h"
#include "activation.h"
#include <memory>
#include <vector>
#include <string>

/**
 * @file attention.h
 * @brief Attention Mechanisms for Neural Networks
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT IS ATTENTION?
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Without Attention:
 * ─────────────────
 * Encoder: [x₁, x₂, x₃, x₄] → h (single context vector)
 *                              ↓
 * Decoder:                     Tries to decode everything from h
 * 
 * Problem: Single vector h must encode ENTIRE input sequence!
 * - Information bottleneck
 * - Loses details for long sequences
 * - Equal importance to all inputs
 * 
 * 
 * With Attention:
 * ──────────────
 * Encoder: [x₁, x₂, x₃, x₄] → [h₁, h₂, h₃, h₄]
 *                              ↓   ↓   ↓   ↓
 *                         α₁  α₂  α₃  α₄ (attention weights)
 *                              ↓
 *                         context = Σ αᵢ·hᵢ
 *                              ↓
 * Decoder:                Uses different context for each step!
 * 
 * Benefits:
 * ✓ No information bottleneck
 * ✓ Focus on relevant parts
 * ✓ Handles longer sequences
 * ✓ Interpretable (can visualize attention weights)
 * 
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * ATTENTION MECHANISM TYPES
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * 1. DOT-PRODUCT ATTENTION (Luong):
 *    score(h, s) = h·s
 *    α = softmax(score)
 *    context = Σ αᵢ·hᵢ
 * 
 * 2. ADDITIVE ATTENTION (Bahdanau):
 *    score(h, s) = v·tanh(W₁·h + W₂·s)
 *    α = softmax(score)
 *    context = Σ αᵢ·hᵢ
 * 
 * 3. SCALED DOT-PRODUCT (Transformer):
 *    score(Q, K) = (Q·Kᵀ) / √d
 *    α = softmax(score)
 *    context = α·V
 */

/**
 * @brief Base class for attention mechanisms
 */
class Attention {
public:
    virtual ~Attention() = default;
    
    /**
     * @brief Compute attention weights and context
     * 
     * @param query Current decoder state (batch_size × query_dim)
     * @param keys Encoder hidden states (seq_len × batch_size × key_dim)
     * @param values Encoder hidden states (seq_len × batch_size × value_dim)
     * @return Pair of (context_vector, attention_weights)
     * 
     * context_vector: (batch_size × value_dim)
     * attention_weights: (batch_size × seq_len)
     */
    virtual std::pair<Matrix, Matrix> forward(
        const Matrix& query,
        const std::vector<Matrix>& keys,
        const std::vector<Matrix>& values) = 0;
    
    /**
     * @brief Backward pass
     */
    virtual std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> backward(
        const Matrix& grad_context,
        const Matrix& grad_attention_weights) = 0;
    
    /**
     * @brief Update parameters
     */
    virtual void updateParameters(double learning_rate) = 0;
    
    /**
     * @brief Reset gradients
     */
    virtual void resetGradients() = 0;
    
    /**
     * @brief Get attention name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get parameter count
     */
    virtual int getParameterCount() const = 0;
};

/**
 * @brief Dot-Product Attention (Luong et al.)
 * 
 * Simplest and fastest attention mechanism:
 * 
 * score(h, s) = hᵀ · s
 * α = softmax(score)
 * context = Σ αᵢ · hᵢ
 * 
 * Advantages:
 * ✓ No parameters to learn
 * ✓ Very fast
 * ✓ Works well when query and key dimensions match
 * 
 * Disadvantages:
 * ✗ Requires query_dim = key_dim
 * ✗ No learned alignment
 */
class DotProductAttention : public Attention {
private:
    Matrix cached_query;
    std::vector<Matrix> cached_keys;
    std::vector<Matrix> cached_values;
    Matrix cached_scores;
    Matrix cached_attention_weights;
    
    // Softmax for attention weights
    std::unique_ptr<Activation> softmax;
    
public:
    DotProductAttention();
    
    std::pair<Matrix, Matrix> forward(
        const Matrix& query,
        const std::vector<Matrix>& keys,
        const std::vector<Matrix>& values) override;
    
    std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> backward(
        const Matrix& grad_context,
        const Matrix& grad_attention_weights) override;
    
    void updateParameters(double learning_rate) override {}
    void resetGradients() override {}
    
    std::string getName() const override { return "DotProductAttention"; }
    int getParameterCount() const override { return 0; }
};

/**
 * @brief Additive Attention (Bahdanau et al.)
 * 
 * More flexible attention with learnable alignment:
 * 
 * score(h, s) = vᵀ · tanh(W₁·h + W₂·s)
 * α = softmax(score)
 * context = Σ αᵢ · hᵢ
 * 
 * Advantages:
 * ✓ Learns alignment function
 * ✓ Works with any query/key dimensions
 * ✓ Often better performance
 * 
 * Disadvantages:
 * ✗ More parameters
 * ✗ Slower than dot-product
 */
class AdditiveAttention : public Attention {
private:
    size_t query_dim;
    size_t key_dim;
    size_t hidden_dim;
    
    // Learnable parameters
    Matrix W_query;   // (hidden_dim × query_dim)
    Matrix W_key;     // (hidden_dim × key_dim)
    Matrix v;         // (1 × hidden_dim)
    
    // Gradients
    Matrix dW_query;
    Matrix dW_key;
    Matrix dv;
    
    // Cached for backward
    Matrix cached_query;
    std::vector<Matrix> cached_keys;
    std::vector<Matrix> cached_values;
    Matrix cached_scores;
    Matrix cached_attention_weights;
    std::vector<Matrix> cached_tanh_output;
    
    std::unique_ptr<Activation> tanh_activation;
    
public:
    /**
     * @brief Constructor
     * @param query_dim Dimension of query vector
     * @param key_dim Dimension of key vectors
     * @param hidden_dim Dimension of attention hidden layer
     */
    AdditiveAttention(size_t query_dim, size_t key_dim, size_t hidden_dim);
    
    void initializeWeights(const std::string& strategy = "xavier");
    
    std::pair<Matrix, Matrix> forward(
        const Matrix& query,
        const std::vector<Matrix>& keys,
        const std::vector<Matrix>& values) override;
    
    std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> backward(
        const Matrix& grad_context,
        const Matrix& grad_attention_weights) override;
    
    void updateParameters(double learning_rate) override;
    void resetGradients() override;
    
    std::string getName() const override { return "AdditiveAttention"; }
    int getParameterCount() const override {
        return query_dim * hidden_dim + key_dim * hidden_dim + hidden_dim;
    }
    
    // Getters
    Matrix getWeightQuery() const { return W_query; }
    Matrix getWeightKey() const { return W_key; }
    Matrix getV() const { return v; }
};

/**
 * @brief Scaled Dot-Product Attention (Vaswani et al.)
 * 
 * Used in Transformers, scales by √d:
 * 
 * score(Q, K) = (Q·Kᵀ) / √d
 * α = softmax(score)
 * context = α·V
 * 
 * The scaling prevents dot products from growing too large
 * (which would push softmax into regions with small gradients)
 */
class ScaledDotProductAttention : public Attention {
private:
    double scale_factor;
    
    Matrix cached_query;
    std::vector<Matrix> cached_keys;
    std::vector<Matrix> cached_values;
    Matrix cached_scores;
    Matrix cached_attention_weights;
    
public:
    ScaledDotProductAttention(size_t key_dim);
    
    std::pair<Matrix, Matrix> forward(
        const Matrix& query,
        const std::vector<Matrix>& keys,
        const std::vector<Matrix>& values) override;
    
    std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> backward(
        const Matrix& grad_context,
        const Matrix& grad_attention_weights) override;
    
    void updateParameters(double learning_rate) override {}
    void resetGradients() override {}
    
    std::string getName() const override { return "ScaledDotProductAttention"; }
    int getParameterCount() const override { return 0; }
};

/**
 * @brief Multi-Head Attention (Transformer)
 * 
 * Runs multiple attention mechanisms in parallel:
 * 
 * head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)
 * MultiHead(Q,K,V) = Concat(head₁, ..., head_h)·W^O
 * 
 * Benefits:
 * ✓ Attend to different aspects simultaneously
 * ✓ More expressive
 * ✓ Core of Transformer architecture
 */
class MultiHeadAttention : public Attention {
private:
    size_t num_heads;
    size_t d_model;
    size_t d_k;
    size_t d_v;
    
    // Projection matrices for each head
    std::vector<Matrix> W_Q;  // Query projections
    std::vector<Matrix> W_K;  // Key projections
    std::vector<Matrix> W_V;  // Value projections
    Matrix W_O;               // Output projection
    
    // Attention mechanisms (one per head)
    std::vector<std::unique_ptr<ScaledDotProductAttention>> attention_heads;
    
    // Gradients
    std::vector<Matrix> dW_Q;
    std::vector<Matrix> dW_K;
    std::vector<Matrix> dW_V;
    Matrix dW_O;
    
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);
    
    void initializeWeights(const std::string& strategy = "xavier");
    
    std::pair<Matrix, Matrix> forward(
        const Matrix& query,
        const std::vector<Matrix>& keys,
        const std::vector<Matrix>& values) override;
    
    std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> backward(
        const Matrix& grad_context,
        const Matrix& grad_attention_weights) override;
    
    void updateParameters(double learning_rate) override;
    void resetGradients() override;
    
    std::string getName() const override { return "MultiHeadAttention"; }
    int getParameterCount() const override;
};

#endif // ATTENTION_H
