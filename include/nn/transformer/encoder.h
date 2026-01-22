#ifndef ENCODER_H
#define ENCODER_H

#include "../matrix.h"
#include "attention.h"
#include "feedforward.h"
#include "layer_norm.h"
#include <memory>
#include <vector>

/**
 * @brief Single Transformer Encoder Layer
 * 
 * Architecture:
 * x → Multi-Head Self-Attention → Add & Norm → 
 * Feed-Forward → Add & Norm → output
 * 
 * With residual connections around each sub-layer
 */
class EncoderLayer {
private:
    size_t d_model;
    
    // Sub-layers
    std::unique_ptr<MultiHeadAttention> self_attention;
    std::unique_ptr<PositionWiseFeedForward> feed_forward;
    std::unique_ptr<LayerNormalization> norm1;
    std::unique_ptr<LayerNormalization> norm2;
    
    double dropout_rate;
    
    // Cached for backward pass
    Matrix cached_input;
    Matrix cached_attn_output;
    Matrix cached_ffn_input;
    Matrix dropout_mask1, dropout_mask2;
    
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension
     * @param dropout Dropout rate
     */
    EncoderLayer(size_t d_model, size_t num_heads, size_t d_ff, 
                 double dropout = 0.1);
    
    /**
     * @brief Forward pass
     * @param input Input matrix (batch × seq_len × d_model)
     * @param mask Optional padding mask
     * @param training Whether in training mode
     * @return Output (batch × seq_len × d_model)
     */
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr,
                   bool training = true);
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    Matrix backward(const Matrix& grad_output);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get attention weights (for visualization)
     */
    std::vector<Matrix> getAttentionWeights() const;
    
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
 * @brief Stack of Transformer Encoder Layers
 * 
 * Multiple encoder layers stacked on top of each other
 */
class TransformerEncoder {
private:
    size_t num_layers;
    size_t d_model;
    
    // Stack of encoder layers
    std::vector<std::unique_ptr<EncoderLayer>> layers;
    
    // Final layer normalization (optional, post-norm style)
    std::unique_ptr<LayerNormalization> final_norm;
    
public:
    /**
     * @brief Constructor
     * @param num_layers Number of encoder layers
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward dimension
     * @param dropout Dropout rate
     */
    TransformerEncoder(size_t num_layers, size_t d_model, size_t num_heads,
                      size_t d_ff, double dropout = 0.1);
    
    /**
     * @brief Forward pass through all layers
     * @param input Input embeddings (batch × seq_len × d_model)
     * @param mask Optional padding mask
     * @param training Whether in training mode
     * @return Encoded output (batch × seq_len × d_model)
     */
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr,
                   bool training = true);
    
    /**
     * @brief Backward pass through all layers
     * @param grad_output Gradient from decoder or loss
     * @return Gradient with respect to input embeddings
     */
    Matrix backward(const Matrix& grad_output);
    
    /**
     * @brief Update parameters in all layers
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get attention weights from all layers
     */
    std::vector<std::vector<Matrix>> getAllAttentionWeights() const;
    
    size_t getNumLayers() const { return num_layers; }
    size_t getDModel() const { return d_model; }
    
    /**
     * @brief Save all encoder weights to binary stream
     */
    void saveWeights(std::ofstream& out) const;
    
    /**
     * @brief Load all encoder weights from binary stream
     */
    void loadWeights(std::ifstream& in);
};

#endif // ENCODER_H
