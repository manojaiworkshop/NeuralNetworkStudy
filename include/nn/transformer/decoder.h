#ifndef DECODER_H
#define DECODER_H

#include "../matrix.h"
#include "attention.h"
#include "feedforward.h"
#include "layer_norm.h"
#include <memory>
#include <vector>

/**
 * @brief Single Transformer Decoder Layer
 * 
 * Architecture:
 * x → Masked Multi-Head Self-Attention → Add & Norm →
 * Multi-Head Cross-Attention (with encoder output) → Add & Norm →
 * Feed-Forward → Add & Norm → output
 * 
 * With residual connections around each sub-layer
 */
class DecoderLayer {
private:
    size_t d_model;
    
    // Sub-layers
    std::unique_ptr<MultiHeadAttention> self_attention;   // Masked
    std::unique_ptr<MultiHeadAttention> cross_attention;  // Attend to encoder
    std::unique_ptr<PositionWiseFeedForward> feed_forward;
    std::unique_ptr<LayerNormalization> norm1;
    std::unique_ptr<LayerNormalization> norm2;
    std::unique_ptr<LayerNormalization> norm3;
    
    double dropout_rate;
    
    // Cached for backward pass
    Matrix cached_input;
    Matrix cached_self_attn_output;
    Matrix cached_cross_attn_output;
    Matrix cached_ffn_input;
    Matrix cached_encoder_output;
    Matrix dropout_mask1, dropout_mask2, dropout_mask3;
    
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension
     * @param dropout Dropout rate
     */
    DecoderLayer(size_t d_model, size_t num_heads, size_t d_ff,
                 double dropout = 0.1);
    
    /**
     * @brief Forward pass
     * @param input Decoder input (batch × tgt_seq_len × d_model)
     * @param encoder_output Encoder output (batch × src_seq_len × d_model)
     * @param src_mask Padding mask for source (encoder output)
     * @param tgt_mask Causal mask for target (prevent future peeking)
     * @param training Whether in training mode
     * @return Output (batch × tgt_seq_len × d_model)
     */
    Matrix forward(const Matrix& input, const Matrix& encoder_output,
                   const Matrix* src_mask = nullptr,
                   const Matrix* tgt_mask = nullptr,
                   bool training = true);
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer or loss
     * @param grad_encoder Output gradient for encoder output
     * @return Gradient with respect to decoder input
     */
    Matrix backward(const Matrix& grad_output, Matrix& grad_encoder);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get self-attention weights (for visualization)
     */
    std::vector<Matrix> getSelfAttentionWeights() const;
    
    /**
     * @brief Get cross-attention weights (for visualization)
     */
    std::vector<Matrix> getCrossAttentionWeights() const;
};

/**
 * @brief Stack of Transformer Decoder Layers
 * 
 * Multiple decoder layers stacked on top of each other
 */
class TransformerDecoder {
private:
    size_t num_layers;
    size_t d_model;
    
    // Stack of decoder layers
    std::vector<std::unique_ptr<DecoderLayer>> layers;
    
    // Final layer normalization (optional, post-norm style)
    std::unique_ptr<LayerNormalization> final_norm;
    
public:
    /**
     * @brief Constructor
     * @param num_layers Number of decoder layers
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension
     * @param dropout Dropout rate
     */
    TransformerDecoder(size_t num_layers, size_t d_model, size_t num_heads,
                      size_t d_ff, double dropout = 0.1);
    
    /**
     * @brief Forward pass through all layers
     * @param input Decoder input embeddings (batch × tgt_seq_len × d_model)
     * @param encoder_output Encoder output (batch × src_seq_len × d_model)
     * @param src_mask Padding mask for source
     * @param tgt_mask Causal mask for target
     * @param training Whether in training mode
     * @return Decoded output (batch × tgt_seq_len × d_model)
     */
    Matrix forward(const Matrix& input, const Matrix& encoder_output,
                   const Matrix* src_mask = nullptr,
                   const Matrix* tgt_mask = nullptr,
                   bool training = true);
    
    /**
     * @brief Backward pass through all layers
     * @param grad_output Gradient from output layer
     * @param grad_encoder Accumulated gradient for encoder output
     * @return Gradient with respect to decoder input embeddings
     */
    Matrix backward(const Matrix& grad_output, Matrix& grad_encoder);
    
    /**
     * @brief Update parameters in all layers
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get self-attention weights from all layers
     */
    std::vector<std::vector<Matrix>> getAllSelfAttentionWeights() const;
    
    /**
     * @brief Get cross-attention weights from all layers
     */
    std::vector<std::vector<Matrix>> getAllCrossAttentionWeights() const;
    
    size_t getNumLayers() const { return num_layers; }
    size_t getDModel() const { return d_model; }
};

#endif // DECODER_H
