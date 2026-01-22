#include "../../include/nn/transformer/layer_norm.h"
#include <fstream>
#include <cmath>
#include <stdexcept>

LayerNormalization::LayerNormalization(size_t normalized_shape, double eps)
    : normalized_shape(normalized_shape), epsilon(eps) {
    
    // Initialize learnable parameters
    gamma = Matrix(1, normalized_shape, 1.0);  // Scale (initialized to 1)
    beta = Matrix(1, normalized_shape, 0.0);   // Shift (initialized to 0)
    
    // Initialize gradients
    gamma_grad = Matrix(1, normalized_shape, 0.0);
    beta_grad = Matrix(1, normalized_shape, 0.0);
}

Matrix LayerNormalization::forward(const Matrix& input) {
    size_t batch_seq = input.getRows();
    size_t features = input.getCols();
    
    if (features != normalized_shape) {
        throw std::runtime_error("Input feature dimension mismatch");
    }
    
    // Cache input for backward
    cached_input = input;
    
    // Initialize output
    Matrix output(batch_seq, features);
    
    // Compute mean and variance for each sample
    cached_mean = Matrix(batch_seq, 1, 0.0);
    cached_std = Matrix(batch_seq, 1, 0.0);
    cached_normalized = Matrix(batch_seq, features);
    
    // Compute statistics and normalize
    for (size_t i = 0; i < batch_seq; i++) {
        // Compute mean
        double mean = 0.0;
        for (size_t j = 0; j < features; j++) {
            mean += input.get(i, j);
        }
        mean /= features;
        cached_mean.set(i, 0, mean);
        
        // Compute variance
        double var = 0.0;
        for (size_t j = 0; j < features; j++) {
            double diff = input.get(i, j) - mean;
            var += diff * diff;
        }
        var /= features;
        double std_dev = std::sqrt(var + epsilon);
        cached_std.set(i, 0, std_dev);
        
        // Normalize and scale/shift
        double std_inv = 1.0 / std_dev;
        for (size_t j = 0; j < features; j++) {
            // Normalize
            double normalized = (input.get(i, j) - mean) * std_inv;
            cached_normalized.set(i, j, normalized);
            
            // Apply affine transformation
            output.set(i, j, gamma.get(0, j) * normalized + beta.get(0, j));
        }
    }
    
    return output;
}

Matrix LayerNormalization::backward(const Matrix& grad_output) {
    size_t batch_seq = grad_output.getRows();
    size_t features = grad_output.getCols();
    
    Matrix grad_input(batch_seq, features, 0.0);
    
    // Accumulate parameter gradients
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < features; j++) {
            double dg = gamma_grad.get(0, j) + grad_output.get(i, j) * cached_normalized.get(i, j);
            gamma_grad.set(0, j, dg);
            double db = beta_grad.get(0, j) + grad_output.get(i, j);
            beta_grad.set(0, j, db);
        }
    }
    
    // Compute input gradient
    for (size_t i = 0; i < batch_seq; i++) {
        double mean = cached_mean.get(i, 0);
        double std_dev = cached_std.get(i, 0);
        double std_inv = 1.0 / std_dev;
        double var = std_dev * std_dev - epsilon;
        
        // Gradient w.r.t normalized values
        std::vector<double> grad_normalized(features);
        for (size_t j = 0; j < features; j++) {
            grad_normalized[j] = grad_output.get(i, j) * gamma.get(0, j);
        }
        
        // Gradient w.r.t variance
        double grad_var = 0.0;
        for (size_t j = 0; j < features; j++) {
            double x_centered = cached_input.get(i, j) - mean;
            grad_var += grad_normalized[j] * x_centered * (-0.5) * std::pow(var + epsilon, -1.5);
        }
        
        // Gradient w.r.t mean
        double grad_mean = 0.0;
        for (size_t j = 0; j < features; j++) {
            grad_mean += grad_normalized[j] * (-std_inv);
        }
        
        double sum_centered = 0.0;
        for (size_t j = 0; j < features; j++) {
            sum_centered += cached_input.get(i, j) - mean;
        }
        grad_mean += grad_var * (-2.0 / features) * sum_centered;
        
        // Gradient w.r.t input
        for (size_t j = 0; j < features; j++) {
            double x_centered = cached_input.get(i, j) - mean;
            double grad_val = grad_normalized[j] * std_inv +
                             grad_var * (2.0 / features) * x_centered +
                             grad_mean / features;
            grad_input.set(i, j, grad_val);
        }
    }
    
    return grad_input;
}

void LayerNormalization::updateParameters(double learning_rate) {
    for (size_t j = 0; j < normalized_shape; j++) {
        gamma.set(0, j, gamma.get(0, j) - learning_rate * gamma_grad.get(0, j));
        beta.set(0, j, beta.get(0, j) - learning_rate * beta_grad.get(0, j));
        
        // Reset gradients
        gamma_grad.set(0, j, 0.0);
        beta_grad.set(0, j, 0.0);
    }
}

void LayerNormalization::saveWeights(std::ofstream& out) const {
    // Save gamma
    size_t rows = gamma.getRows();
    size_t cols = gamma.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = gamma.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
    
    // Save beta
    rows = beta.getRows();
    cols = beta.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = beta.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
}

void LayerNormalization::loadWeights(std::ifstream& in) {
    size_t rows, cols;
    
    // Load gamma
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            gamma.set(i, j, val);
        }
    }
    
    // Load beta
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            beta.set(i, j, val);
        }
    }
}
