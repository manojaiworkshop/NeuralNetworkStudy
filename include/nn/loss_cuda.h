/**
 * CUDA-accelerated Loss Functions for Neural Networks
 * 
 * This file provides GPU-accelerated implementations of loss functions
 * using NVIDIA CUDA for parallel computation.
 * 
 * Features:
 * - Parallel computation on GPU for large matrices
 * - Inherits from base Loss class for compatibility
 * - Supports all standard loss functions (MSE, MAE, BCE, CCE)
 * - Automatic memory management with MatrixCUDA
 */

#ifndef NN_LOSS_CUDA_H
#define NN_LOSS_CUDA_H

#include "loss.h"
#include "matrix_cuda.h"
#include <memory>
#include <string>

/**
 * Base class for CUDA-accelerated loss functions
 * Inherits from Loss for polymorphic use
 */
class LossCUDA : public Loss {
public:
    virtual ~LossCUDA() = default;
    
    // CPU interface (inherited)
    virtual double calculate(const Matrix& predictions, const Matrix& targets) const override;
    virtual Matrix gradient(const Matrix& predictions, const Matrix& targets) const override;
    
    // CUDA interface (new methods)
    virtual double calculateCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const = 0;
    virtual MatrixCUDA gradientCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const = 0;
    
    virtual std::string getName() const override = 0;
    virtual std::unique_ptr<Loss> clone() const override = 0;
};

/**
 * Mean Squared Error Loss - CUDA Implementation
 * 
 * Formula: MSE = (1/n) * Σ(y - ŷ)²
 * 
 * Use for: Regression problems
 * Properties:
 * - Penalizes large errors exponentially
 * - Always positive
 * - Smooth gradient
 */
class MSELossCUDA : public LossCUDA {
public:
    double calculateCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    MatrixCUDA gradientCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    
    std::string getName() const override { return "MSELossCUDA"; }
    std::unique_ptr<Loss> clone() const override {
        return std::make_unique<MSELossCUDA>(*this);
    }
};

/**
 * Binary Cross-Entropy Loss - CUDA Implementation
 * 
 * Formula: BCE = -(1/n) * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
 * 
 * Use for: Binary classification (2 classes)
 * Properties:
 * - Works with probabilities [0, 1]
 * - Use with Sigmoid activation
 * - Epsilon clipping prevents log(0)
 */
class BinaryCrossEntropyLossCUDA : public LossCUDA {
private:
    double epsilon;

public:
    explicit BinaryCrossEntropyLossCUDA(double eps = 1e-7) : epsilon(eps) {}
    
    double calculateCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    MatrixCUDA gradientCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    
    double getEpsilon() const { return epsilon; }
    
    std::string getName() const override { return "BinaryCrossEntropyLossCUDA"; }
    std::unique_ptr<Loss> clone() const override {
        return std::make_unique<BinaryCrossEntropyLossCUDA>(*this);
    }
};

/**
 * Categorical Cross-Entropy Loss - CUDA Implementation
 * 
 * Formula: CCE = -(1/n) * ΣΣ y_ij · log(ŷ_ij)
 * 
 * Use for: Multi-class classification (>2 classes)
 * Properties:
 * - Works with probability distributions
 * - Use with Softmax activation
 * - Targets should be one-hot encoded
 */
class CategoricalCrossEntropyLossCUDA : public LossCUDA {
private:
    double epsilon;

public:
    explicit CategoricalCrossEntropyLossCUDA(double eps = 1e-7) : epsilon(eps) {}
    
    double calculateCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    MatrixCUDA gradientCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    
    double getEpsilon() const { return epsilon; }
    
    std::string getName() const override { return "CategoricalCrossEntropyLossCUDA"; }
    std::unique_ptr<Loss> clone() const override {
        return std::make_unique<CategoricalCrossEntropyLossCUDA>(*this);
    }
};

/**
 * Mean Absolute Error Loss - CUDA Implementation
 * 
 * Formula: MAE = (1/n) * Σ|y - ŷ|
 * 
 * Use for: Robust regression
 * Properties:
 * - Linear penalty (treats all errors equally)
 * - Robust to outliers
 * - Good for median estimation
 */
class MAELossCUDA : public LossCUDA {
public:
    double calculateCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    MatrixCUDA gradientCUDA(const MatrixCUDA& predictions, const MatrixCUDA& targets) const override;
    
    std::string getName() const override { return "MAELossCUDA"; }
    std::unique_ptr<Loss> clone() const override {
        return std::make_unique<MAELossCUDA>(*this);
    }
};

#endif // NN_LOSS_CUDA_H
