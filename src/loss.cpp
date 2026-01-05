#include "nn/loss.h"
#include <cmath>
#include <stdexcept>

// ==================== MSE Loss ====================

double MSELoss::calculate(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    double sum = 0.0;
    size_t n = predictions.getRows() * predictions.getCols();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double diff = predictions.get(i, j) - targets.get(i, j);
            sum += diff * diff;
        }
    }
    
    return sum / n;
}

Matrix MSELoss::gradient(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    // Gradient: ∂MSE/∂ŷ = 2/n * (ŷ - y)
    // Often we simplify to: ∂MSE/∂ŷ = (ŷ - y)
    size_t n = predictions.getRows() * predictions.getCols();
    return (predictions - targets) * (2.0 / n);
}

std::unique_ptr<Loss> MSELoss::clone() const {
    return std::make_unique<MSELoss>();
}

// ==================== Binary Cross-Entropy Loss ====================

double BinaryCrossEntropyLoss::calculate(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    double sum = 0.0;
    size_t n = predictions.getRows() * predictions.getCols();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            // Clip predictions to prevent log(0)
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            // BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
            sum += -(target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
        }
    }
    
    return sum / n;
}

Matrix BinaryCrossEntropyLoss::gradient(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    Matrix result(predictions.getRows(), predictions.getCols());
    size_t n = predictions.getRows() * predictions.getCols();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            // Clip predictions to prevent division by zero
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            // Gradient: ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
            // Simplified: (ŷ - y) / (ŷ(1-ŷ))
            double grad = -(target / pred - (1.0 - target) / (1.0 - pred)) / n;
            result.set(i, j, grad);
        }
    }
    
    return result;
}

std::unique_ptr<Loss> BinaryCrossEntropyLoss::clone() const {
    return std::make_unique<BinaryCrossEntropyLoss>(epsilon);
}

// ==================== Categorical Cross-Entropy Loss ====================

double CategoricalCrossEntropyLoss::calculate(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    double sum = 0.0;
    size_t n = predictions.getRows();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            // Clip predictions to prevent log(0)
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            // CCE = -Σ y * log(ŷ)
            if (target > 0) {  // Only compute for non-zero targets
                sum += -target * std::log(pred);
            }
        }
    }
    
    return sum / n;
}

Matrix CategoricalCrossEntropyLoss::gradient(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    Matrix result(predictions.getRows(), predictions.getCols());
    size_t n = predictions.getRows();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            // Clip predictions to prevent division by zero
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            // Gradient: ∂CCE/∂ŷ = -y/ŷ
            // When using with Softmax, this simplifies to (ŷ - y)
            double grad = -target / pred / n;
            result.set(i, j, grad);
        }
    }
    
    return result;
}

std::unique_ptr<Loss> CategoricalCrossEntropyLoss::clone() const {
    return std::make_unique<CategoricalCrossEntropyLoss>(epsilon);
}

// ==================== MAE Loss ====================

double MAELoss::calculate(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    double sum = 0.0;
    size_t n = predictions.getRows() * predictions.getCols();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            sum += std::abs(predictions.get(i, j) - targets.get(i, j));
        }
    }
    
    return sum / n;
}

Matrix MAELoss::gradient(const Matrix& predictions, const Matrix& targets) const {
    if (!predictions.sameShape(targets)) {
        throw std::invalid_argument("Predictions and targets must have same shape");
    }
    
    Matrix result(predictions.getRows(), predictions.getCols());
    size_t n = predictions.getRows() * predictions.getCols();
    
    for (size_t i = 0; i < predictions.getRows(); ++i) {
        for (size_t j = 0; j < predictions.getCols(); ++j) {
            double diff = predictions.get(i, j) - targets.get(i, j);
            // Gradient: ∂MAE/∂ŷ = sign(ŷ - y)
            double grad = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
            result.set(i, j, grad / n);
        }
    }
    
    return result;
}

std::unique_ptr<Loss> MAELoss::clone() const {
    return std::make_unique<MAELoss>();
}
