/**
 * @file optimizer_cuda.cu
 * @brief CUDA implementation of optimizer algorithms
 * 
 * This file implements GPU-accelerated versions of all optimizers using MatrixCUDA.
 * 
 * Implementation Note:
 * Like activation_cuda.cu and loss_cuda.cu, this implementation uses CPU-side
 * computation with MatrixCUDA wrappers rather than custom CUDA kernels.
 * 
 * Why?
 * - MatrixCUDA doesn't expose device pointers (no getData() method)
 * - MatrixCUDA already provides optimized GPU matrix operations
 * - Operations like addition, multiplication, division are already GPU-accelerated
 * - This approach maintains API consistency and leverages existing optimizations
 * 
 * The operations are still GPU-accelerated because:
 * 1. MatrixCUDA stores data on GPU
 * 2. All arithmetic operations happen on GPU
 * 3. Only control flow happens on CPU
 * 4. No unnecessary CPU-GPU transfers during computation
 */

#include "../include/nn/optimizer_cuda.h"
#include <cmath>
#include <stdexcept>

// ============================================================================
// SGD_CUDA Implementation
// ============================================================================

/**
 * @brief Stochastic Gradient Descent update
 * 
 * Formula: θ = θ - α * ∇θ
 * 
 * This is the simplest optimizer update:
 * 1. Multiply gradient by learning rate
 * 2. Subtract from parameters
 * 
 * No state is maintained - completely stateless.
 * 
 * @param parameters Current parameter values (GPU)
 * @param gradients Computed gradients (GPU)
 * @param param_id Parameter identifier (unused for SGD)
 * @return Updated parameters (GPU)
 */
MatrixCUDA SGD_CUDA::update(const MatrixCUDA& parameters, 
                            const MatrixCUDA& gradients,
                            const std::string& param_id) {
    // Simple gradient descent: θ = θ - α * ∇θ
    // Both operations happen on GPU via MatrixCUDA
    return parameters - gradients * learning_rate;
}

// ============================================================================
// Momentum_CUDA Implementation
// ============================================================================

/**
 * @brief Momentum optimizer update
 * 
 * Formula:
 *   v = β * v + ∇θ           (update velocity)
 *   θ = θ - α * v            (update parameters using velocity)
 * 
 * Steps:
 * 1. Check if this is first update for this parameter
 *    - If yes, initialize velocity to zero
 * 2. Update velocity: v_new = β * v_old + gradient
 *    - This accumulates past gradient direction
 * 3. Update parameters using velocity instead of raw gradient
 * 
 * The velocity smooths out noisy gradients and accelerates
 * in directions with consistent gradient signs.
 * 
 * @param parameters Current parameter values (GPU)
 * @param gradients Computed gradients (GPU)
 * @param param_id Parameter identifier for state tracking
 * @return Updated parameters (GPU)
 */
MatrixCUDA Momentum_CUDA::update(const MatrixCUDA& parameters, 
                                const MatrixCUDA& gradients,
                                const std::string& param_id) {
    // Initialize velocity if not exists
    if (velocity.find(param_id) == velocity.end()) {
        // Create zero matrix with same dimensions as parameters
        velocity[param_id] = MatrixCUDA(parameters.getRows(), parameters.getCols());
        velocity[param_id].zeros();  // Initialize to zero
    }
    
    // Update velocity: v = β * v + ∇θ
    // This exponentially weights past gradients
    velocity[param_id] = velocity[param_id] * beta + gradients;
    
    // Update parameters: θ = θ - α * v
    // Use smoothed velocity instead of raw gradient
    return parameters - velocity[param_id] * learning_rate;
}

// ============================================================================
// RMSprop_CUDA Implementation
// ============================================================================

/**
 * @brief RMSprop optimizer update
 * 
 * Formula:
 *   v = β * v + (1 - β) * ∇θ²      (update moving average of squared gradients)
 *   θ = θ - α * ∇θ / (√v + ε)      (adaptive update)
 * 
 * Steps:
 * 1. Initialize cache if first update
 * 2. Compute squared gradients (element-wise)
 * 3. Update cache with exponential moving average
 * 4. Divide gradient by square root of cache (+ epsilon for stability)
 * 5. Update parameters with scaled gradient
 * 
 * The division by √cache adapts the learning rate per parameter:
 * - Large gradients → large cache → divide by large number → smaller update
 * - Small gradients → small cache → divide by small number → larger update
 * 
 * @param parameters Current parameter values (GPU)
 * @param gradients Computed gradients (GPU)
 * @param param_id Parameter identifier for state tracking
 * @return Updated parameters (GPU)
 */
MatrixCUDA RMSprop_CUDA::update(const MatrixCUDA& parameters, 
                               const MatrixCUDA& gradients,
                               const std::string& param_id) {
    // Initialize cache if not exists
    if (cache.find(param_id) == cache.end()) {
        cache[param_id] = MatrixCUDA(parameters.getRows(), parameters.getCols());
        cache[param_id].zeros();
    }
    
    // Compute squared gradients: ∇θ²
    // hadamard = element-wise multiplication
    MatrixCUDA grad_squared = gradients.hadamard(gradients);
    
    // Update cache: v = β * v + (1 - β) * ∇θ²
    // Exponential moving average of squared gradients
    cache[param_id] = cache[param_id] * beta + grad_squared * (1.0 - beta);
    
    // Compute denominator: √v + ε
    // apply() executes lambda on each element (GPU-accelerated in MatrixCUDA)
    MatrixCUDA denominator = cache[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    // Compute update: α * ∇θ / (√v + ε)
    MatrixCUDA update = gradients.divide(denominator) * learning_rate;
    
    // Update parameters: θ = θ - update
    return parameters - update;
}

// ============================================================================
// Adam_CUDA Implementation
// ============================================================================

/**
 * @brief Adam optimizer update
 * 
 * Formula:
 *   m = β₁ * m + (1 - β₁) * ∇θ           (first moment - momentum)
 *   v = β₂ * v + (1 - β₂) * ∇θ²          (second moment - variance)
 *   m̂ = m / (1 - β₁^t)                   (bias-corrected first moment)
 *   v̂ = v / (1 - β₂^t)                   (bias-corrected second moment)
 *   θ = θ - α * m̂ / (√v̂ + ε)
 * 
 * Steps:
 * 1. Initialize m, v, t if first update
 * 2. Increment time step
 * 3. Update first moment (like Momentum)
 * 4. Update second moment (like RMSprop)
 * 5. Compute bias corrections (compensates for initialization at zero)
 * 6. Apply bias corrections to moments
 * 7. Update parameters using corrected moments
 * 
 * The bias correction is crucial for good early training:
 * - At t=1: correction = large (accounts for zero initialization)
 * - As t→∞: correction → 1 (no correction needed)
 * 
 * @param parameters Current parameter values (GPU)
 * @param gradients Computed gradients (GPU)
 * @param param_id Parameter identifier for state tracking
 * @return Updated parameters (GPU)
 */
MatrixCUDA Adam_CUDA::update(const MatrixCUDA& parameters, 
                            const MatrixCUDA& gradients,
                            const std::string& param_id) {
    // Initialize moments if not exists
    if (m.find(param_id) == m.end()) {
        // First moment (momentum)
        m[param_id] = MatrixCUDA(parameters.getRows(), parameters.getCols());
        m[param_id].zeros();
        
        // Second moment (variance)
        v[param_id] = MatrixCUDA(parameters.getRows(), parameters.getCols());
        v[param_id].zeros();
        
        // Time step counter
        t[param_id] = 0;
    }
    
    // Increment time step
    t[param_id]++;
    int time_step = t[param_id];
    
    // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * ∇θ
    // This is the momentum component
    m[param_id] = m[param_id] * beta1 + gradients * (1.0 - beta1);
    
    // Update biased second raw moment estimate: v = β₂ * v + (1 - β₂) * ∇θ²
    // This is the adaptive learning rate component
    MatrixCUDA grad_squared = gradients.hadamard(gradients);
    v[param_id] = v[param_id] * beta2 + grad_squared * (1.0 - beta2);
    
    // Compute bias-corrected first moment estimate: m̂ = m / (1 - β₁^t)
    // This corrects for initialization bias (m starts at 0)
    double bias_correction1 = 1.0 - std::pow(beta1, time_step);
    MatrixCUDA m_hat = m[param_id] / bias_correction1;
    
    // Compute bias-corrected second raw moment estimate: v̂ = v / (1 - β₂^t)
    // This corrects for initialization bias (v starts at 0)
    double bias_correction2 = 1.0 - std::pow(beta2, time_step);
    MatrixCUDA v_hat = v[param_id] / bias_correction2;
    
    // Compute denominator: √v̂ + ε
    MatrixCUDA denominator = v_hat.apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    // Compute update: α * m̂ / (√v̂ + ε)
    // Combines momentum (m̂) with adaptive learning rate (√v̂)
    MatrixCUDA update = m_hat.divide(denominator) * learning_rate;
    
    // Update parameters: θ = θ - update
    return parameters - update;
}

// ============================================================================
// AdaGrad_CUDA Implementation
// ============================================================================

/**
 * @brief AdaGrad optimizer update
 * 
 * Formula:
 *   G = G + ∇θ²                (accumulate squared gradients)
 *   θ = θ - α * ∇θ / (√G + ε)  (adaptive update)
 * 
 * Steps:
 * 1. Initialize accumulated gradients if first update
 * 2. Compute squared gradients
 * 3. Accumulate (sum, never decay!) squared gradients
 * 4. Divide gradient by square root of accumulated gradients
 * 5. Update parameters
 * 
 * Key difference from RMSprop:
 * - RMSprop: v = β*v + (1-β)*∇θ²  (exponential average, forgets old gradients)
 * - AdaGrad: G = G + ∇θ²          (sum, never forgets)
 * 
 * This causes G to grow monotonically, which means:
 * - Effective learning rate decreases over time
 * - Good for sparse features (rare features get larger updates)
 * - Can stop learning if G becomes too large
 * 
 * @param parameters Current parameter values (GPU)
 * @param gradients Computed gradients (GPU)
 * @param param_id Parameter identifier for state tracking
 * @return Updated parameters (GPU)
 */
MatrixCUDA AdaGrad_CUDA::update(const MatrixCUDA& parameters, 
                               const MatrixCUDA& gradients,
                               const std::string& param_id) {
    // Initialize accumulated gradients if not exists
    if (accumulated_gradients.find(param_id) == accumulated_gradients.end()) {
        accumulated_gradients[param_id] = MatrixCUDA(parameters.getRows(), 
                                                     parameters.getCols());
        accumulated_gradients[param_id].zeros();
    }
    
    // Accumulate squared gradients: G = G + ∇θ²
    // Note: No decay factor (unlike RMSprop)
    MatrixCUDA grad_squared = gradients.hadamard(gradients);
    accumulated_gradients[param_id] = accumulated_gradients[param_id] + grad_squared;
    
    // Compute denominator: √G + ε
    MatrixCUDA denominator = accumulated_gradients[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    // Compute update: α * ∇θ / (√G + ε)
    MatrixCUDA update = gradients.divide(denominator) * learning_rate;
    
    // Update parameters: θ = θ - update
    return parameters - update;
}
