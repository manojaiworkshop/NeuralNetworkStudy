#include "nn/activation.h"
#include <cmath>
#include <algorithm>

// ==================== Sigmoid ====================

Matrix Sigmoid::forward(const Matrix& input) const {
    return input.apply([](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
}

Matrix Sigmoid::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix activated = forward(input);
    // Derivative: σ'(x) = σ(x) * (1 - σ(x))
    Matrix derivative = activated.hadamard(activated.apply([](double x) {
        return 1.0 - x;
    }));
    return derivative.hadamard(output_gradient);
}

std::unique_ptr<Activation> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>();
}

// ==================== Tanh ====================

Matrix Tanh::forward(const Matrix& input) const {
    return input.apply([](double x) {
        return std::tanh(x);
    });
}

Matrix Tanh::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix activated = forward(input);
    // Derivative: tanh'(x) = 1 - tanh²(x)
    Matrix derivative = activated.apply([](double x) {
        return 1.0 - x * x;
    });
    return derivative.hadamard(output_gradient);
}

std::unique_ptr<Activation> Tanh::clone() const {
    return std::make_unique<Tanh>();
}

// ==================== ReLU ====================

Matrix ReLU::forward(const Matrix& input) const {
    return input.apply([](double x) {
        return std::max(0.0, x);
    });
}

Matrix ReLU::backward(const Matrix& input, const Matrix& output_gradient) const {
    // Derivative: 1 if x > 0, else 0
    Matrix derivative = input.apply([](double x) {
        return (x > 0.0) ? 1.0 : 0.0;
    });
    return derivative.hadamard(output_gradient);
}

std::unique_ptr<Activation> ReLU::clone() const {
    return std::make_unique<ReLU>();
}

// ==================== Leaky ReLU ====================

Matrix LeakyReLU::forward(const Matrix& input) const {
    return input.apply([this](double x) {
        return (x > 0.0) ? x : alpha * x;
    });
}

Matrix LeakyReLU::backward(const Matrix& input, const Matrix& output_gradient) const {
    // Derivative: 1 if x > 0, else alpha
    Matrix derivative = input.apply([this](double x) {
        return (x > 0.0) ? 1.0 : alpha;
    });
    return derivative.hadamard(output_gradient);
}

std::unique_ptr<Activation> LeakyReLU::clone() const {
    return std::make_unique<LeakyReLU>(alpha);
}

// ==================== Linear ====================

Matrix Linear::forward(const Matrix& input) const {
    return input;
}

Matrix Linear::backward(const Matrix& input, const Matrix& output_gradient) const {
    // Derivative of identity is 1
    return output_gradient;
}

std::unique_ptr<Activation> Linear::clone() const {
    return std::make_unique<Linear>();
}

// ==================== Softmax ====================

Matrix Softmax::forward(const Matrix& input) const {
    Matrix result(input.getRows(), input.getCols());
    
    // Process each row independently (each sample)
    for (size_t i = 0; i < input.getRows(); ++i) {
        // Find max for numerical stability
        double max_val = input.get(i, 0);
        for (size_t j = 1; j < input.getCols(); ++j) {
            max_val = std::max(max_val, input.get(i, j));
        }
        
        // Compute exp(x - max) and sum
        double sum = 0.0;
        std::vector<double> exp_vals(input.getCols());
        for (size_t j = 0; j < input.getCols(); ++j) {
            exp_vals[j] = std::exp(input.get(i, j) - max_val);
            sum += exp_vals[j];
        }
        
        // Normalize
        for (size_t j = 0; j < input.getCols(); ++j) {
            result.set(i, j, exp_vals[j] / sum);
        }
    }
    
    return result;
}

Matrix Softmax::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix output = forward(input);
    Matrix result(input.getRows(), input.getCols());
    
    // For each sample in the batch
    for (size_t i = 0; i < input.getRows(); ++i) {
        // Compute Jacobian for this sample
        // ∂softmax(x_i)/∂x_j = softmax(x_i) * (δ_ij - softmax(x_j))
        for (size_t j = 0; j < input.getCols(); ++j) {
            double grad = 0.0;
            for (size_t k = 0; k < input.getCols(); ++k) {
                double s_i = output.get(i, j);
                double s_k = output.get(i, k);
                double kronecker = (j == k) ? 1.0 : 0.0;
                grad += output_gradient.get(i, k) * s_i * (kronecker - s_k);
            }
            result.set(i, j, grad);
        }
    }
    
    return result;
}

std::unique_ptr<Activation> Softmax::clone() const {
    return std::make_unique<Softmax>();
}
