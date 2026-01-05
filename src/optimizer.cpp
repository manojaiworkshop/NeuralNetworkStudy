#include "nn/optimizer.h"
#include <cmath>

// ==================== SGD ====================

Matrix SGD::update(const Matrix& parameters, const Matrix& gradients, 
                   const std::string& param_id) {
    // Simple gradient descent: θ = θ - α * ∇θ
    return parameters - gradients * learning_rate;
}

// ==================== Momentum ====================

Matrix Momentum::update(const Matrix& parameters, const Matrix& gradients, 
                       const std::string& param_id) {
    // Initialize velocity if not exists
    if (velocity.find(param_id) == velocity.end()) {
        velocity[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        velocity[param_id].zeros();
    }
    
    // Update velocity: v = β * v + ∇θ
    velocity[param_id] = velocity[param_id] * beta + gradients;
    
    // Update parameters: θ = θ - α * v
    return parameters - velocity[param_id] * learning_rate;
}

// ==================== RMSprop ====================

Matrix RMSprop::update(const Matrix& parameters, const Matrix& gradients, 
                      const std::string& param_id) {
    // Initialize cache if not exists
    if (cache.find(param_id) == cache.end()) {
        cache[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        cache[param_id].zeros();
    }
    
    // Update cache: v = β * v + (1 - β) * ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    cache[param_id] = cache[param_id] * beta + grad_squared * (1.0 - beta);
    
    // Compute update: θ = θ - α * ∇θ / (√v + ε)
    Matrix denominator = cache[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    Matrix update = gradients.divide(denominator) * learning_rate;
    return parameters - update;
}

// ==================== Adam ====================

Matrix Adam::update(const Matrix& parameters, const Matrix& gradients, 
                   const std::string& param_id) {
    // Initialize moments if not exists
    if (m.find(param_id) == m.end()) {
        m[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        m[param_id].zeros();
        v[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        v[param_id].zeros();
        t[param_id] = 0;
    }
    
    // Increment time step
    t[param_id]++;
    int time_step = t[param_id];
    
    // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * ∇θ
    m[param_id] = m[param_id] * beta1 + gradients * (1.0 - beta1);
    
    // Update biased second raw moment estimate: v = β₂ * v + (1 - β₂) * ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    v[param_id] = v[param_id] * beta2 + grad_squared * (1.0 - beta2);
    
    // Compute bias-corrected first moment estimate: m̂ = m / (1 - β₁^t)
    double bias_correction1 = 1.0 - std::pow(beta1, time_step);
    Matrix m_hat = m[param_id] / bias_correction1;
    
    // Compute bias-corrected second raw moment estimate: v̂ = v / (1 - β₂^t)
    double bias_correction2 = 1.0 - std::pow(beta2, time_step);
    Matrix v_hat = v[param_id] / bias_correction2;
    
    // Compute update: θ = θ - α * m̂ / (√v̂ + ε)
    Matrix denominator = v_hat.apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    Matrix update = m_hat.divide(denominator) * learning_rate;
    return parameters - update;
}

// ==================== AdaGrad ====================

Matrix AdaGrad::update(const Matrix& parameters, const Matrix& gradients, 
                      const std::string& param_id) {
    // Initialize accumulated gradients if not exists
    if (accumulated_gradients.find(param_id) == accumulated_gradients.end()) {
        accumulated_gradients[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        accumulated_gradients[param_id].zeros();
    }
    
    // Accumulate squared gradients: G = G + ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    accumulated_gradients[param_id] = accumulated_gradients[param_id] + grad_squared;
    
    // Compute update: θ = θ - α * ∇θ / (√G + ε)
    Matrix denominator = accumulated_gradients[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    Matrix update = gradients.divide(denominator) * learning_rate;
    return parameters - update;
}
