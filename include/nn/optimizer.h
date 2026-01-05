#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

/**
 * @brief Base class for optimization algorithms
 */
class Optimizer {
protected:
    double learning_rate;
    
public:
    explicit Optimizer(double learning_rate = 0.01) : learning_rate(learning_rate) {}
    virtual ~Optimizer() = default;
    
    /**
     * @brief Update parameters using computed gradients
     * @param parameters Current parameter values
     * @param gradients Computed gradients
     * @param param_id Unique identifier for parameter (for stateful optimizers)
     * @return Updated parameters
     */
    virtual Matrix update(const Matrix& parameters, const Matrix& gradients, 
                         const std::string& param_id) = 0;
    
    /**
     * @brief Get optimizer name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Reset optimizer state
     */
    virtual void reset() {}
    
    /**
     * @brief Set learning rate
     */
    void setLearningRate(double lr) { learning_rate = lr; }
    
    /**
     * @brief Get learning rate
     */
    double getLearningRate() const { return learning_rate; }
};

/**
 * @brief Stochastic Gradient Descent optimizer
 * Update: θ = θ - α * ∇θ
 */
class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01) : Optimizer(learning_rate) {}
    
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
    
    std::string getName() const override { return "SGD"; }
};

/**
 * @brief SGD with Momentum
 * v = β * v + ∇θ
 * θ = θ - α * v
 */
class Momentum : public Optimizer {
private:
    double beta;  // Momentum coefficient (typically 0.9)
    std::unordered_map<std::string, Matrix> velocity;
    
public:
    explicit Momentum(double learning_rate = 0.01, double beta = 0.9)
        : Optimizer(learning_rate), beta(beta) {}
    
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
    
    std::string getName() const override { return "Momentum"; }
    
    void reset() override { velocity.clear(); }
};

/**
 * @brief RMSprop optimizer
 * v = β * v + (1 - β) * ∇θ²
 * θ = θ - α * ∇θ / (√v + ε)
 */
class RMSprop : public Optimizer {
private:
    double beta;     // Decay rate (typically 0.9)
    double epsilon;  // Small constant for numerical stability
    std::unordered_map<std::string, Matrix> cache;
    
public:
    explicit RMSprop(double learning_rate = 0.001, double beta = 0.9, double epsilon = 1e-8)
        : Optimizer(learning_rate), beta(beta), epsilon(epsilon) {}
    
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
    
    std::string getName() const override { return "RMSprop"; }
    
    void reset() override { cache.clear(); }
};

/**
 * @brief Adam optimizer (Adaptive Moment Estimation)
 * Combines momentum and RMSprop
 * 
 * m = β₁ * m + (1 - β₁) * ∇θ
 * v = β₂ * v + (1 - β₂) * ∇θ²
 * m̂ = m / (1 - β₁^t)
 * v̂ = v / (1 - β₂^t)
 * θ = θ - α * m̂ / (√v̂ + ε)
 */
class Adam : public Optimizer {
private:
    double beta1;    // First moment decay rate (typically 0.9)
    double beta2;    // Second moment decay rate (typically 0.999)
    double epsilon;  // Small constant for numerical stability
    
    std::unordered_map<std::string, Matrix> m;  // First moment
    std::unordered_map<std::string, Matrix> v;  // Second moment
    std::unordered_map<std::string, int> t;     // Time step
    
public:
    explicit Adam(double learning_rate = 0.001, double beta1 = 0.9, 
                 double beta2 = 0.999, double epsilon = 1e-8)
        : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}
    
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
    
    std::string getName() const override { return "Adam"; }
    
    void reset() override { 
        m.clear(); 
        v.clear(); 
        t.clear(); 
    }
};

/**
 * @brief AdaGrad optimizer
 * G = G + ∇θ²
 * θ = θ - α * ∇θ / (√G + ε)
 */
class AdaGrad : public Optimizer {
private:
    double epsilon;  // Small constant for numerical stability
    std::unordered_map<std::string, Matrix> accumulated_gradients;
    
public:
    explicit AdaGrad(double learning_rate = 0.01, double epsilon = 1e-8)
        : Optimizer(learning_rate), epsilon(epsilon) {}
    
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
    
    std::string getName() const override { return "AdaGrad"; }
    
    void reset() override { accumulated_gradients.clear(); }
};

#endif // OPTIMIZER_H
