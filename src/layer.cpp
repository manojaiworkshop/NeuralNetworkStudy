#include "nn/layer.h"
#include <stdexcept>

// Constructor
DenseLayer::DenseLayer(size_t input_size, size_t output_size, Activation* activation)
    : input_size(input_size), output_size(output_size),
      weights(output_size, input_size),
      biases(output_size, 1),
      weight_gradients(output_size, input_size),
      bias_gradients(output_size, 1),
      activation(activation) {
    
    // Initialize weights and biases
    initializeWeights("xavier");
    biases.zeros();
    
    // Initialize gradients to zero
    weight_gradients.zeros();
    bias_gradients.zeros();
}

// Initialize weights
void DenseLayer::initializeWeights(const std::string& strategy) {
    if (strategy == "xavier" || strategy == "glorot") {
        weights.xavierInit(input_size, output_size);
    } else if (strategy == "he") {
        weights.heInit(input_size);
    } else if (strategy == "random") {
        weights.randomize(-0.5, 0.5);
    } else if (strategy == "zeros") {
        weights.zeros();
    } else {
        throw std::invalid_argument("Unknown weight initialization strategy: " + strategy);
    }
}

// Forward pass
Matrix DenseLayer::forward(const Matrix& input) {
    // Check dimensions
    if (input.getCols() != input_size) {
        throw std::invalid_argument("Input size mismatch in DenseLayer::forward");
    }
    
    // Cache input for backward pass
    cached_input = input;
    
    // Linear transformation: Z = X * W^T + b
    // X: (batch_size x input_size)
    // W: (output_size x input_size)
    // W^T: (input_size x output_size)
    // Z: (batch_size x output_size)
    
    Matrix z = input * weights.transpose();
    
    // Add bias (broadcast across batch)
    for (size_t i = 0; i < z.getRows(); ++i) {
        for (size_t j = 0; j < z.getCols(); ++j) {
            z.set(i, j, z.get(i, j) + biases.get(j, 0));
        }
    }
    
    cached_z = z;
    
    // Apply activation if present
    if (activation) {
        return activation->forward(z);
    }
    
    return z;
}

// Backward pass
Matrix DenseLayer::backward(const Matrix& output_gradient) {
    // Apply activation gradient if present
    Matrix delta = output_gradient;
    if (activation) {
        delta = activation->backward(cached_z, output_gradient);
    }
    
    // Gradient with respect to weights: dL/dW = delta^T * input
    // delta: (batch_size x output_size)
    // input: (batch_size x input_size)
    // dL/dW: (output_size x input_size)
    weight_gradients = delta.transpose() * cached_input;
    
    // Gradient with respect to biases: dL/db = sum(delta, axis=0)
    // Sum across batch dimension
    bias_gradients = Matrix(output_size, 1);
    for (size_t j = 0; j < output_size; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < delta.getRows(); ++i) {
            sum += delta.get(i, j);
        }
        bias_gradients.set(j, 0, sum);
    }
    
    // Gradient with respect to input: dL/dX = delta * W
    // delta: (batch_size x output_size)
    // W: (output_size x input_size)
    // dL/dX: (batch_size x input_size)
    Matrix input_gradient = delta * weights;
    
    return input_gradient;
}

// Update parameters
void DenseLayer::updateParameters(double learning_rate) {
    // Simple gradient descent: θ = θ - α * ∇θ
    weights = weights - weight_gradients * learning_rate;
    biases = biases - bias_gradients * learning_rate;
    
    // Reset gradients
    resetGradients();
}

// Reset gradients
void DenseLayer::resetGradients() {
    weight_gradients.zeros();
    bias_gradients.zeros();
}
