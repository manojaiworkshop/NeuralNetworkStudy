#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include <memory>
#include <string>

/**
 * @brief Base class for neural network layers
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    /**
     * @brief Forward pass through the layer
     * @param input Input matrix
     * @return Output matrix
     */
    virtual Matrix forward(const Matrix& input) = 0;
    
    /**
     * @brief Backward pass (compute gradients)
     * @param output_gradient Gradient from next layer
     * @return Gradient with respect to input
     */
    virtual Matrix backward(const Matrix& output_gradient) = 0;
    
    /**
     * @brief Update layer parameters
     * @param learning_rate Learning rate for update
     */
    virtual void updateParameters(double learning_rate) = 0;
    
    /**
     * @brief Get layer name/type
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get input size
     */
    virtual size_t getInputSize() const = 0;
    
    /**
     * @brief Get output size
     */
    virtual size_t getOutputSize() const = 0;
    
    /**
     * @brief Get number of trainable parameters
     */
    virtual int getParameterCount() const = 0;
    
    /**
     * @brief Get weights (if applicable)
     */
    virtual Matrix getWeights() const { return Matrix(); }
    
    /**
     * @brief Get biases (if applicable)
     */
    virtual Matrix getBiases() const { return Matrix(); }
    
    /**
     * @brief Set weights (if applicable)
     */
    virtual void setWeights(const Matrix& weights) {}
    
    /**
     * @brief Set biases (if applicable)
     */
    virtual void setBiases(const Matrix& biases) {}
};

/**
 * @brief Dense (Fully Connected) Layer
 */
class DenseLayer : public Layer {
private:
    size_t input_size;
    size_t output_size;
    
    Matrix weights;           // (output_size x input_size)
    Matrix biases;            // (output_size x 1)
    
    Matrix weight_gradients;  // Accumulated gradients for weights
    Matrix bias_gradients;    // Accumulated gradients for biases
    
    Matrix cached_input;      // Cached for backward pass
    
    std::unique_ptr<Activation> activation;
    Matrix cached_z;          // Pre-activation values
    
public:
    /**
     * @brief Constructor
     * @param input_size Number of input features
     * @param output_size Number of output features
     * @param activation Activation function (nullptr for linear)
     */
    DenseLayer(size_t input_size, size_t output_size, Activation* activation = nullptr);
    
    /**
     * @brief Initialize weights using specific strategy
     * @param strategy "xavier", "he", "random", or "zeros"
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& output_gradient) override;
    void updateParameters(double learning_rate) override;
    
    std::string getName() const override { return "Dense"; }
    size_t getInputSize() const override { return input_size; }
    size_t getOutputSize() const override { return output_size; }
    int getParameterCount() const override { 
        return (input_size * output_size) + output_size; 
    }
    
    Matrix getWeights() const override { return weights; }
    Matrix getBiases() const override { return biases; }
    void setWeights(const Matrix& w) override { weights = w; }
    void setBiases(const Matrix& b) override { biases = b; }
    
    /**
     * @brief Get weight gradients (useful for custom optimizers)
     */
    Matrix getWeightGradients() const { return weight_gradients; }
    
    /**
     * @brief Get bias gradients (useful for custom optimizers)
     */
    Matrix getBiasGradients() const { return bias_gradients; }
    
    /**
     * @brief Reset gradients to zero
     */
    void resetGradients();
};

#endif // LAYER_H
