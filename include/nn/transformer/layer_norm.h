#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "../matrix.h"
#include "../layer.h"
#include <cmath>

/**
 * @brief Layer Normalization
 * 
 * Normalizes activations across features (not across batch like BatchNorm)
 * LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
 * 
 * Where:
 * - μ: mean across features
 * - σ²: variance across features
 * - γ, β: learnable scale and shift parameters
 * - ε: small constant for numerical stability
 */
class LayerNormalization : public Layer {
private:
    size_t normalized_shape;  // Number of features to normalize
    double epsilon;
    
    // Learnable parameters
    Matrix gamma;  // Scale (initialized to 1)
    Matrix beta;   // Shift (initialized to 0)
    
    // Gradients
    Matrix gamma_grad;
    Matrix beta_grad;
    
    // Cached for backward pass
    Matrix cached_input;
    Matrix cached_mean;
    Matrix cached_std;
    Matrix cached_normalized;
    
public:
    /**
     * @brief Constructor
     * @param normalized_shape Size of the feature dimension
     * @param epsilon Small constant for numerical stability (default: 1e-6)
     */
    LayerNormalization(size_t normalized_shape, double epsilon = 1e-6);
    
    /**
     * @brief Forward pass
     * @param input Input matrix (batch_size × seq_len × features) flattened
     * @return Normalized output
     */
    Matrix forward(const Matrix& input) override;
    
    /**
     * @brief Backward pass
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    Matrix backward(const Matrix& grad_output) override;
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate) override;
    
    std::string getName() const override { return "LayerNorm"; }
    size_t getInputSize() const override { return normalized_shape; }
    size_t getOutputSize() const override { return normalized_shape; }
    int getParameterCount() const override { return 2 * normalized_shape; }
    
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
    
    /**
     * @brief Get parameters
     */
    const Matrix& getGamma() const { return gamma; }
    const Matrix& getBeta() const { return beta; }
    
    void setGamma(const Matrix& g) { gamma = g; }
    void setBeta(const Matrix& b) { beta = b; }
};

#endif // LAYER_NORM_H
