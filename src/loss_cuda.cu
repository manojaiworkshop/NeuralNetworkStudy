/**
 * CUDA Implementation of Loss Functions
 * 
 * This file contains GPU-accelerated implementations of loss functions.
 * 
 * Note: Current implementation uses CPU-side computation with MatrixCUDA wrappers
 * for compatibility with existing MatrixCUDA interface. For production use,
 * extend MatrixCUDA to expose device pointers for direct kernel access.
 */

#include "nn/loss_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>

// =====================================================
// Helper Function: Sum all elements (CPU-side for now)
// =====================================================

double sumAllElements(const MatrixCUDA& matrix) {
    // Convert to CPU and sum
    // In production, this should use GPU reduction
    Matrix cpu_matrix = static_cast<Matrix>(matrix);
    double sum = 0.0;
    for (int i = 0; i < cpu_matrix.getRows(); i++) {
        for (int j = 0; j < cpu_matrix.getCols(); j++) {
            sum += cpu_matrix.get(i, j);
        }
    }
    return sum;
}

// =====================================================
// Base LossCUDA Implementation
// =====================================================

double LossCUDA::calculate(const Matrix& predictions, const Matrix& targets) const {
    // Convert CPU matrices to CUDA
    MatrixCUDA pred_cuda(predictions);
    MatrixCUDA target_cuda(targets);
    
    // Call CUDA implementation
    return calculateCUDA(pred_cuda, target_cuda);
}

Matrix LossCUDA::gradient(const Matrix& predictions, const Matrix& targets) const {
    // Convert CPU matrices to CUDA
    MatrixCUDA pred_cuda(predictions);
    MatrixCUDA target_cuda(targets);
    
    // Call CUDA implementation
    MatrixCUDA grad_cuda = gradientCUDA(pred_cuda, target_cuda);
    
    // Convert back to CPU
    return static_cast<Matrix>(grad_cuda);
}

// =====================================================
// MSELossCUDA Implementation
// =====================================================

double MSELossCUDA::calculateCUDA(const MatrixCUDA& predictions, 
                                 const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "MSELossCUDA: predictions and targets must have same dimensions");
    }
    
    // Convert to CPU for computation (simplified approach)
    // In production, use custom CUDA kernels with device pointer access
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    double sum = 0.0;
    int size = pred_cpu.getRows() * pred_cpu.getCols();
    
    for (int i = 0; i < pred_cpu.getRows(); i++) {
        for (int j = 0; j < pred_cpu.getCols(); j++) {
            double diff = target_cpu.get(i, j) - pred_cpu.get(i, j);
            sum += diff * diff;
        }
    }
    
    return sum / size;
}

MatrixCUDA MSELossCUDA::gradientCUDA(const MatrixCUDA& predictions,
                                    const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "MSELossCUDA: predictions and targets must have same dimensions");
    }
    
    // Convert to CPU, compute gradient, convert back
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    int rows = pred_cpu.getRows();
    int cols = pred_cpu.getCols();
    int size = rows * cols;
    double scale = 2.0 / size;
    
    Matrix grad_cpu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad_cpu.set(i, j, scale * (pred_cpu.get(i, j) - target_cpu.get(i, j)));
        }
    }
    
    return MatrixCUDA(grad_cpu);
}

// =====================================================
// BinaryCrossEntropyLossCUDA Implementation
// =====================================================

double BinaryCrossEntropyLossCUDA::calculateCUDA(const MatrixCUDA& predictions,
                                                 const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "BinaryCrossEntropyLossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    double sum = 0.0;
    int size = pred_cpu.getRows() * pred_cpu.getCols();
    
    for (int i = 0; i < pred_cpu.getRows(); i++) {
        for (int j = 0; j < pred_cpu.getCols(); j++) {
            double pred = pred_cpu.get(i, j);
            double target = target_cpu.get(i, j);
            
            // Clip prediction to avoid log(0)
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            sum += target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred);
        }
    }
    
    return -sum / size;
}

MatrixCUDA BinaryCrossEntropyLossCUDA::gradientCUDA(const MatrixCUDA& predictions,
                                                    const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "BinaryCrossEntropyLossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    int rows = pred_cpu.getRows();
    int cols = pred_cpu.getCols();
    int size = rows * cols;
    double scale = 1.0 / size;
    
    Matrix grad_cpu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pred = pred_cpu.get(i, j);
            double target = target_cpu.get(i, j);
            
            // Clip prediction
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
            
            grad_cpu.set(i, j, -(target / pred - (1.0 - target) / (1.0 - pred)) * scale);
        }
    }
    
    return MatrixCUDA(grad_cpu);
}

// =====================================================
// CategoricalCrossEntropyLossCUDA Implementation
// =====================================================

double CategoricalCrossEntropyLossCUDA::calculateCUDA(const MatrixCUDA& predictions,
                                                      const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "CategoricalCrossEntropyLossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    double sum = 0.0;
    int n_samples = pred_cpu.getRows();
    
    for (int i = 0; i < pred_cpu.getRows(); i++) {
        for (int j = 0; j < pred_cpu.getCols(); j++) {
            double target = target_cpu.get(i, j);
            if (target > 0.0) {
                double pred = std::max(epsilon, pred_cpu.get(i, j));
                sum += target * std::log(pred);
            }
        }
    }
    
    return -sum / n_samples;
}

MatrixCUDA CategoricalCrossEntropyLossCUDA::gradientCUDA(const MatrixCUDA& predictions,
                                                         const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "CategoricalCrossEntropyLossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    int rows = pred_cpu.getRows();
    int cols = pred_cpu.getCols();
    int n_samples = rows;
    
    Matrix grad_cpu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pred = std::max(epsilon, pred_cpu.get(i, j));
            double target = target_cpu.get(i, j);
            
            grad_cpu.set(i, j, -target / (pred * n_samples));
        }
    }
    
    return MatrixCUDA(grad_cpu);
}

// =====================================================
// MAELossCUDA Implementation
// =====================================================

double MAELossCUDA::calculateCUDA(const MatrixCUDA& predictions,
                                 const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "MAELossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    double sum = 0.0;
    int size = pred_cpu.getRows() * pred_cpu.getCols();
    
    for (int i = 0; i < pred_cpu.getRows(); i++) {
        for (int j = 0; j < pred_cpu.getCols(); j++) {
            double diff = target_cpu.get(i, j) - pred_cpu.get(i, j);
            sum += std::abs(diff);
        }
    }
    
    return sum / size;
}

MatrixCUDA MAELossCUDA::gradientCUDA(const MatrixCUDA& predictions,
                                    const MatrixCUDA& targets) const {
    if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
        throw std::invalid_argument(
            "MAELossCUDA: predictions and targets must have same dimensions");
    }
    
    Matrix pred_cpu = static_cast<Matrix>(predictions);
    Matrix target_cpu = static_cast<Matrix>(targets);
    
    int rows = pred_cpu.getRows();
    int cols = pred_cpu.getCols();
    int size = rows * cols;
    double scale = 1.0 / size;
    
    Matrix grad_cpu(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = pred_cpu.get(i, j) - target_cpu.get(i, j);
            double sign = (diff > 0.0) ? 1.0 : ((diff < 0.0) ? -1.0 : 0.0);
            grad_cpu.set(i, j, scale * sign);
        }
    }
    
    return MatrixCUDA(grad_cpu);
}
