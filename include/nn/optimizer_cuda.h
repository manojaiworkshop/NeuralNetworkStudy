/**
 * @file optimizer_cuda.h
 * @brief CUDA-accelerated optimizer implementations for neural network training
 * 
 * This file contains GPU-accelerated versions of all optimizer algorithms:
 * - SGD_CUDA: Stochastic Gradient Descent
 * - Momentum_CUDA: SGD with momentum
 * - RMSprop_CUDA: Adaptive learning rates with exponential moving average
 * - Adam_CUDA: Adaptive Moment Estimation (momentum + RMSprop + bias correction)
 * - AdaGrad_CUDA: Adaptive learning rates with accumulated gradients
 * 
 * All optimizers inherit from Optimizer_CUDA base class and use MatrixCUDA
 * for GPU computation. The implementations maintain the same mathematical
 * formulas as CPU versions but leverage GPU parallelism for faster updates.
 * 
 * Usage:
 *   MatrixCUDA weights(rows, cols);
 *   MatrixCUDA gradients(rows, cols);
 *   Adam_CUDA optimizer(0.001);
 *   MatrixCUDA new_weights = optimizer.update(weights, gradients, "layer1");
 */

#ifndef OPTIMIZER_CUDA_H
#define OPTIMIZER_CUDA_H

#include "matrix_cuda.h"
#include "optimizer.h"  // For base interface consistency
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @brief Base class for CUDA-accelerated optimizers
 * 
 * This abstract base class defines the interface for all GPU-accelerated
 * optimizer implementations. It manages learning rate and provides virtual
 * methods for parameter updates and state management.
 * 
 * All derived classes must implement:
 * - update(): Performs the parameter update using gradients
 * - getName(): Returns the optimizer name for identification
 * - reset(): Clears accumulated state (optional, default does nothing)
 */
class Optimizer_CUDA {
protected:
    double learning_rate;  ///< Step size for parameter updates
    
public:
    /**
     * @brief Constructor
     * @param learning_rate Initial learning rate (default: 0.01)
     */
    explicit Optimizer_CUDA(double learning_rate = 0.01) 
        : learning_rate(learning_rate) {}
    
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~Optimizer_CUDA() = default;
    
    /**
     * @brief Update parameters using gradients (pure virtual)
     * @param parameters Current parameter values (GPU matrix)
     * @param gradients Computed gradients (GPU matrix)
     * @param param_id Unique identifier for parameter state tracking
     * @return Updated parameters (GPU matrix)
     * 
     * This method performs one optimization step. The param_id allows
     * tracking separate state (velocity, moments) for each parameter.
     */
    virtual MatrixCUDA update(const MatrixCUDA& parameters, 
                              const MatrixCUDA& gradients,
                              const std::string& param_id) = 0;
    
    /**
     * @brief Get optimizer name
     * @return String identifier (e.g., "SGD_CUDA", "Adam_CUDA")
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Reset accumulated state
     * 
     * Clears all cached values (velocities, moments, etc.).
     * Call this when switching to a new training task.
     */
    virtual void reset() {}
    
    /**
     * @brief Set learning rate
     * @param lr New learning rate value
     */
    void setLearningRate(double lr) { learning_rate = lr; }
    
    /**
     * @brief Get current learning rate
     * @return Current learning rate value
     */
    double getLearningRate() const { return learning_rate; }
};

// ============================================================================
// SGD_CUDA: Stochastic Gradient Descent
// ============================================================================

/**
 * @brief CUDA-accelerated Stochastic Gradient Descent optimizer
 * 
 * Mathematical Formula:
 *   θ_new = θ_old - α * ∇θ
 * 
 * Where:
 *   θ = parameters
 *   α = learning_rate
 *   ∇θ = gradients
 * 
 * This is the simplest optimizer - directly descends the gradient.
 * No state is maintained between updates (stateless).
 * 
 * Pros:
 *   - Simple and fast
 *   - Low memory usage
 *   - Stable and predictable
 * 
 * Cons:
 *   - Can be slow to converge
 *   - Sensitive to learning rate
 *   - No momentum or adaptive rates
 * 
 * Typical learning rate: 0.01 - 0.1
 */
class SGD_CUDA : public Optimizer_CUDA {
public:
    explicit SGD_CUDA(double learning_rate = 0.01) 
        : Optimizer_CUDA(learning_rate) {}
    
    MatrixCUDA update(const MatrixCUDA& parameters, 
                     const MatrixCUDA& gradients,
                     const std::string& param_id) override;
    
    std::string getName() const override { return "SGD_CUDA"; }
};

// ============================================================================
// Momentum_CUDA: SGD with Momentum
// ============================================================================

/**
 * @brief CUDA-accelerated Momentum optimizer
 * 
 * Mathematical Formula:
 *   v_new = β * v_old + ∇θ
 *   θ_new = θ_old - α * v_new
 * 
 * Where:
 *   v = velocity (accumulated gradient direction)
 *   β = momentum coefficient (typically 0.9)
 *   α = learning_rate
 *   ∇θ = gradients
 * 
 * Momentum accumulates past gradients to smooth optimization path and
 * accelerate convergence, especially in directions with consistent gradients.
 * 
 * Physical Analogy: Ball rolling downhill - builds up speed in consistent
 * direction, dampens oscillations in inconsistent directions.
 * 
 * Pros:
 *   - Faster convergence than SGD
 *   - Smooths noisy gradients
 *   - Can escape shallow local minima
 * 
 * Cons:
 *   - Extra memory for velocity
 *   - Can overshoot minimum
 *   - One more hyperparameter (β)
 * 
 * Typical settings: α = 0.01, β = 0.9
 */
class Momentum_CUDA : public Optimizer_CUDA {
private:
    double beta;  ///< Momentum coefficient (default: 0.9)
    std::unordered_map<std::string, MatrixCUDA> velocity;  ///< Per-parameter velocities
    
public:
    explicit Momentum_CUDA(double learning_rate = 0.01, double beta = 0.9)
        : Optimizer_CUDA(learning_rate), beta(beta) {}
    
    MatrixCUDA update(const MatrixCUDA& parameters, 
                     const MatrixCUDA& gradients,
                     const std::string& param_id) override;
    
    std::string getName() const override { return "Momentum_CUDA"; }
    
    void reset() override { velocity.clear(); }
};

// ============================================================================
// RMSprop_CUDA: Root Mean Square Propagation
// ============================================================================

/**
 * @brief CUDA-accelerated RMSprop optimizer
 * 
 * Mathematical Formula:
 *   v_new = β * v_old + (1 - β) * ∇θ²
 *   θ_new = θ_old - α * ∇θ / (√v_new + ε)
 * 
 * Where:
 *   v = moving average of squared gradients
 *   β = decay rate (typically 0.9)
 *   α = learning_rate
 *   ε = small constant for numerical stability (1e-8)
 *   ∇θ² = element-wise squared gradients
 * 
 * RMSprop adapts the learning rate per parameter based on gradient magnitude.
 * Parameters with large gradients get smaller effective learning rates,
 * and vice versa.
 * 
 * Pros:
 *   - Adaptive per-parameter learning rates
 *   - Good for RNNs and non-stationary problems
 *   - Handles different scales automatically
 * 
 * Cons:
 *   - More complex than SGD
 *   - Extra memory for cache
 *   - Can be sensitive to learning rate
 * 
 * Typical settings: α = 0.001, β = 0.9, ε = 1e-8
 */
class RMSprop_CUDA : public Optimizer_CUDA {
private:
    double beta;     ///< Decay rate for moving average (default: 0.9)
    double epsilon;  ///< Numerical stability constant (default: 1e-8)
    std::unordered_map<std::string, MatrixCUDA> cache;  ///< Per-parameter caches
    
public:
    explicit RMSprop_CUDA(double learning_rate = 0.001, 
                         double beta = 0.9, 
                         double epsilon = 1e-8)
        : Optimizer_CUDA(learning_rate), beta(beta), epsilon(epsilon) {}
    
    MatrixCUDA update(const MatrixCUDA& parameters, 
                     const MatrixCUDA& gradients,
                     const std::string& param_id) override;
    
    std::string getName() const override { return "RMSprop_CUDA"; }
    
    void reset() override { cache.clear(); }
};

// ============================================================================
// Adam_CUDA: Adaptive Moment Estimation
// ============================================================================

/**
 * @brief CUDA-accelerated Adam optimizer
 * 
 * Mathematical Formula:
 *   m_new = β₁ * m_old + (1 - β₁) * ∇θ           (first moment)
 *   v_new = β₂ * v_old + (1 - β₂) * ∇θ²          (second moment)
 *   m_hat = m_new / (1 - β₁^t)                   (bias correction)
 *   v_hat = v_new / (1 - β₂^t)                   (bias correction)
 *   θ_new = θ_old - α * m_hat / (√v_hat + ε)
 * 
 * Where:
 *   m = first moment estimate (mean of gradients) - like Momentum
 *   v = second moment estimate (variance of gradients) - like RMSprop
 *   β₁ = decay rate for first moment (typically 0.9)
 *   β₂ = decay rate for second moment (typically 0.999)
 *   t = time step (iteration counter)
 *   α = learning_rate
 *   ε = numerical stability constant (1e-8)
 * 
 * Adam combines the best of Momentum and RMSprop:
 * - Momentum: smooths gradient direction (m)
 * - RMSprop: adapts per-parameter learning rates (v)
 * - Bias correction: accounts for initialization at zero
 * 
 * This is the most popular optimizer in deep learning due to its
 * robustness and good performance across many problem domains.
 * 
 * Pros:
 *   - Excellent out-of-the-box performance
 *   - Combines momentum and adaptive rates
 *   - Robust to hyperparameter choices
 *   - Works well for most problems
 * 
 * Cons:
 *   - More memory (stores m, v, t)
 *   - More computation per step
 *   - Sometimes worse generalization than SGD+Momentum
 * 
 * Typical settings: α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
 * (These are the defaults from the original paper and work well!)
 */
class Adam_CUDA : public Optimizer_CUDA {
private:
    double beta1;    ///< First moment decay rate (default: 0.9)
    double beta2;    ///< Second moment decay rate (default: 0.999)
    double epsilon;  ///< Numerical stability constant (default: 1e-8)
    
    std::unordered_map<std::string, MatrixCUDA> m;  ///< First moment estimates
    std::unordered_map<std::string, MatrixCUDA> v;  ///< Second moment estimates
    std::unordered_map<std::string, int> t;         ///< Time steps per parameter
    
public:
    explicit Adam_CUDA(double learning_rate = 0.001,
                      double beta1 = 0.9,
                      double beta2 = 0.999,
                      double epsilon = 1e-8)
        : Optimizer_CUDA(learning_rate), 
          beta1(beta1), beta2(beta2), epsilon(epsilon) {}
    
    MatrixCUDA update(const MatrixCUDA& parameters, 
                     const MatrixCUDA& gradients,
                     const std::string& param_id) override;
    
    std::string getName() const override { return "Adam_CUDA"; }
    
    void reset() override { 
        m.clear(); 
        v.clear(); 
        t.clear(); 
    }
};

// ============================================================================
// AdaGrad_CUDA: Adaptive Gradient Algorithm
// ============================================================================

/**
 * @brief CUDA-accelerated AdaGrad optimizer
 * 
 * Mathematical Formula:
 *   G_new = G_old + ∇θ²                          (accumulate squared gradients)
 *   θ_new = θ_old - α * ∇θ / (√G_new + ε)
 * 
 * Where:
 *   G = accumulated sum of squared gradients (never decays!)
 *   α = learning_rate
 *   ε = numerical stability constant (1e-8)
 *   ∇θ² = element-wise squared gradients
 * 
 * AdaGrad adapts learning rates based on ALL historical gradients.
 * Unlike RMSprop, it never forgets - G only grows.
 * 
 * This makes it excellent for sparse data:
 * - Frequent parameters: large G → small effective learning rate
 * - Rare parameters: small G → large effective learning rate
 * 
 * Common in NLP and recommendation systems where features have
 * vastly different frequencies.
 * 
 * Pros:
 *   - Perfect for sparse data
 *   - Automatic per-parameter rate adaptation
 *   - No decay hyperparameter needed
 * 
 * Cons:
 *   - Learning rate monotonically decreases
 *   - Can stop learning too early
 *   - Not suitable for non-convex optimization
 * 
 * Typical settings: α = 0.01, ε = 1e-8
 */
class AdaGrad_CUDA : public Optimizer_CUDA {
private:
    double epsilon;  ///< Numerical stability constant (default: 1e-8)
    std::unordered_map<std::string, MatrixCUDA> accumulated_gradients;  ///< G matrices
    
public:
    explicit AdaGrad_CUDA(double learning_rate = 0.01, 
                         double epsilon = 1e-8)
        : Optimizer_CUDA(learning_rate), epsilon(epsilon) {}
    
    MatrixCUDA update(const MatrixCUDA& parameters, 
                     const MatrixCUDA& gradients,
                     const std::string& param_id) override;
    
    std::string getName() const override { return "AdaGrad_CUDA"; }
    
    void reset() override { accumulated_gradients.clear(); }
};

#endif // OPTIMIZER_CUDA_H
