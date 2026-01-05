#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <memory>
#include <string>

/**
 * @brief Base class for loss functions
 */
class Loss {
public:
    virtual ~Loss() = default;
    
    /**
     * @brief Calculate loss value
     * @param predictions Predicted values
     * @param targets True values
     * @return Loss value
     */
    virtual double calculate(const Matrix& predictions, const Matrix& targets) const = 0;
    
    /**
     * @brief Calculate gradient of loss with respect to predictions
     * @param predictions Predicted values
     * @param targets True values
     * @return Gradient matrix
     */
    virtual Matrix gradient(const Matrix& predictions, const Matrix& targets) const = 0;
    
    /**
     * @brief Get name of loss function
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Clone the loss function
     */
    virtual std::unique_ptr<Loss> clone() const = 0;
};

/**
 * @brief Mean Squared Error loss function
 * MSE = (1/n) * Σ(y - ŷ)²
 */
class MSELoss : public Loss {
public:
    double calculate(const Matrix& predictions, const Matrix& targets) const override;
    Matrix gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::string getName() const override { return "MSE"; }
    std::unique_ptr<Loss> clone() const override;
};

/**
 * @brief Binary Cross-Entropy loss function
 * BCE = -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
 */
class BinaryCrossEntropyLoss : public Loss {
private:
    double epsilon;  // Small value to prevent log(0)
    
public:
    explicit BinaryCrossEntropyLoss(double epsilon = 1e-7) : epsilon(epsilon) {}
    
    double calculate(const Matrix& predictions, const Matrix& targets) const override;
    Matrix gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::string getName() const override { return "BinaryCrossEntropy"; }
    std::unique_ptr<Loss> clone() const override;
};

/**
 * @brief Categorical Cross-Entropy loss function
 * CCE = -(1/n) * ΣΣ y_ij * log(ŷ_ij)
 */
class CategoricalCrossEntropyLoss : public Loss {
private:
    double epsilon;  // Small value to prevent log(0)
    
public:
    explicit CategoricalCrossEntropyLoss(double epsilon = 1e-7) : epsilon(epsilon) {}
    
    double calculate(const Matrix& predictions, const Matrix& targets) const override;
    Matrix gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::string getName() const override { return "CategoricalCrossEntropy"; }
    std::unique_ptr<Loss> clone() const override;
};

/**
 * @brief Mean Absolute Error loss function
 * MAE = (1/n) * Σ|y - ŷ|
 */
class MAELoss : public Loss {
public:
    double calculate(const Matrix& predictions, const Matrix& targets) const override;
    Matrix gradient(const Matrix& predictions, const Matrix& targets) const override;
    std::string getName() const override { return "MAE"; }
    std::unique_ptr<Loss> clone() const override;
};

#endif // LOSS_H
