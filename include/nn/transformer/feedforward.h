#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "../matrix.h"
#include "../layer.h"
#include "../activation.h"
#include <memory>

/**
 * @brief Position-wise Feed-Forward Network
 * 
 * FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
 *        = ReLU(xW_1 + b_1)W_2 + b_2
 * 
 * Two linear transformations with ReLU activation:
 * - First layer: d_model → d_ff (expansion)
 * - Second layer: d_ff → d_model (projection)
 * 
 * Typically d_ff = 4 × d_model
 */
class PositionWiseFeedForward {
private:
    size_t d_model;
    size_t d_ff;
    double dropout_rate;
    
    // Parameters
    Matrix W1;  // (d_model × d_ff)
    Matrix b1;  // (d_ff)
    Matrix W2;  // (d_ff × d_model)
    Matrix b2;  // (d_model)
    
    // Gradients
    Matrix dW1, db1, dW2, db2;
    
    // Activation
    std::unique_ptr<Activation> activation;
    
    // Cached for backward pass
    Matrix cached_input;
    Matrix cached_hidden;      // After first linear + activation
    Matrix cached_hidden_pre;  // Before activation
    Matrix dropout_mask;
    
public:
    /**
     * @brief Constructor
     * @param d_model Model dimension
     * @param d_ff Feed-forward dimension (typically 4 × d_model)
     * @param dropout Dropout rate (default: 0.1)
     * @param activation_fn Activation function (default: ReLU)
     */
    PositionWiseFeedForward(size_t d_model, size_t d_ff, 
                           double dropout = 0.1,
                           Activation* activation_fn = nullptr);
    
    /**
     * @brief Initialize weights
     */
    void initializeWeights();
    
    /**
     * @brief Forward pass
     * @param input Input matrix (batch × seq_len × d_model)
     * @param training Whether in training mode (for dropout)
     * @return Output (batch × seq_len × d_model)
     */
    Matrix forward(const Matrix& input, bool training = true);
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    Matrix backward(const Matrix& grad_output);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Save weights to binary stream
     */
    void saveWeights(std::ofstream& out) const;
    
    /**
     * @brief Load weights from binary stream
     */
    void loadWeights(std::ifstream& in);
    
    /**
     * @brief Reset gradients
     */
    void resetGradients();
    
    int getParameterCount() const {
        return (d_model * d_ff + d_ff) + (d_ff * d_model + d_model);
    }
    
    size_t getDModel() const { return d_model; }
    size_t getDFF() const { return d_ff; }
};

#endif // FEEDFORWARD_H
