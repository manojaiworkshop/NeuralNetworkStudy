#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"
#include <memory>
#include <string>

/**
 * @brief Base class for activation functions
 */
class Activation {
public:
    virtual ~Activation() = default;
    
    /**
     * @brief Forward pass through activation function
     * @param input Input matrix
     * @return Activated output
     */
    virtual Matrix forward(const Matrix& input) const = 0;
    
    /**
     * @brief Backward pass (compute gradient)
     * @param input Original input to forward pass
     * @param output_gradient Gradient from next layer
     * @return Gradient with respect to input
     */
    virtual Matrix backward(const Matrix& input, const Matrix& output_gradient) const = 0;
    
    /**
     * @brief Get name of activation function
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Clone the activation function
     */
    virtual std::unique_ptr<Activation> clone() const = 0;
};

/**
 * @brief Sigmoid activation function
 * σ(x) = 1 / (1 + e^(-x))
 */
class Sigmoid : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Sigmoid"; }
    std::unique_ptr<Activation> clone() const override;
};

/**
 * @brief Hyperbolic tangent activation function
 * tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
class Tanh : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Tanh"; }
    std::unique_ptr<Activation> clone() const override;
};

/**
 * @brief Rectified Linear Unit activation function
 * ReLU(x) = max(0, x)
 */
class ReLU : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "ReLU"; }
    std::unique_ptr<Activation> clone() const override;
};

/**
 * @brief Leaky ReLU activation function
 * LeakyReLU(x) = x if x > 0, else alpha * x
 */
class LeakyReLU : public Activation {
private:
    double alpha;
    
public:
    explicit LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
    
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "LeakyReLU"; }
    std::unique_ptr<Activation> clone() const override;
};

/**
 * @brief Linear (identity) activation function
 * Linear(x) = x
 */
class Linear : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Linear"; }
    std::unique_ptr<Activation> clone() const override;
};

/**
 * @brief Softmax activation function (for multi-class classification)
 * softmax(x_i) = e^(x_i) / sum(e^(x_j))
 */
class Softmax : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Softmax"; }
    std::unique_ptr<Activation> clone() const override;
};

#endif // ACTIVATION_H
