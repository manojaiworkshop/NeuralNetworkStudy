#include "../../include/nn/transformer/feedforward.h"
#include <fstream>
#include <cmath>
#include <algorithm>

PositionWiseFeedForward::PositionWiseFeedForward(size_t d_model, size_t d_ff, double dropout, Activation* activation_fn)
    : d_model(d_model), d_ff(d_ff), dropout_rate(dropout) {
    
    // Set activation function (default to ReLU if none provided)
    if (activation_fn == nullptr) {
        activation = std::make_unique<ReLU>();
    } else {
        // Note: We'll need to clone the activation or handle ownership appropriately
        activation = std::unique_ptr<Activation>(activation_fn);
    }
    
    // Initialize weight matrices
    W1 = Matrix(d_model, d_ff);
    b1 = Matrix(1, d_ff, 0.0);
    W2 = Matrix(d_ff, d_model);
    b2 = Matrix(1, d_model, 0.0);
    
    // He initialization (good for ReLU)
    double std1 = std::sqrt(2.0 / d_model);
    double std2 = std::sqrt(2.0 / d_ff);
    
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            W1.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * std1);
        }
    }
    
    for (size_t i = 0; i < d_ff; i++) {
        for (size_t j = 0; j < d_model; j++) {
            W2.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * std2);
        }
    }
    
    // Initialize gradients
    dW1 = Matrix(d_model, d_ff, 0.0);
    db1 = Matrix(1, d_ff, 0.0);
    dW2 = Matrix(d_ff, d_model, 0.0);
    db2 = Matrix(1, d_model, 0.0);
}

void PositionWiseFeedForward::initializeWeights() {
    // Already initialized in constructor
}

Matrix PositionWiseFeedForward::forward(const Matrix& input, bool training) {
    size_t batch_seq = input.getRows();
    
    // Cache input
    cached_input = input;
    
    // First linear layer: input * W1 + b1
    cached_hidden_pre = input * W1;
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            cached_hidden_pre.set(i, j, cached_hidden_pre.get(i, j) + b1.get(0, j));
        }
    }
    
    // Apply activation function (ReLU)
    cached_hidden = activation->forward(cached_hidden_pre);
    
    // Dropout
    Matrix hidden_after_dropout = cached_hidden;
    if (training && dropout_rate > 0.0) {
        dropout_mask = Matrix(batch_seq, d_ff);
        
        for (size_t i = 0; i < batch_seq; i++) {
            for (size_t j = 0; j < d_ff; j++) {
                double rand_val = (double)rand() / RAND_MAX;
                if (rand_val < dropout_rate) {
                    dropout_mask.set(i, j, 0.0);
                    hidden_after_dropout.set(i, j, 0.0);
                } else {
                    double mask_val = 1.0 / (1.0 - dropout_rate);
                    dropout_mask.set(i, j, mask_val);
                    hidden_after_dropout.set(i, j, hidden_after_dropout.get(i, j) * mask_val);
                }
            }
        }
    }
    
    // Second linear layer: hidden * W2 + b2
    Matrix output = hidden_after_dropout * W2;
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b2.get(0, j));
        }
    }
    
    return output;
}

Matrix PositionWiseFeedForward::backward(const Matrix& grad_output) {
    size_t batch_seq = grad_output.getRows();
    
    // Gradient w.r.t W2 and b2 (use hidden with dropout applied if it was used)
    Matrix hidden_for_grad = cached_hidden;
    if (dropout_rate > 0.0 && dropout_mask.getRows() > 0) {
        for (size_t i = 0; i < batch_seq; i++) {
            for (size_t j = 0; j < d_ff; j++) {
                hidden_for_grad.set(i, j, hidden_for_grad.get(i, j) * dropout_mask.get(i, j));
            }
        }
    }
    
    dW2 = dW2 + (hidden_for_grad.transpose() * grad_output);
    for (size_t j = 0; j < d_model; j++) {
        for (size_t i = 0; i < batch_seq; i++) {
            db2.set(0, j, db2.get(0, j) + grad_output.get(i, j));
        }
    }
    
    // Gradient w.r.t hidden
    Matrix grad_hidden = grad_output * W2.transpose();
    
    // Backward through dropout
    if (dropout_rate > 0.0 && dropout_mask.getRows() > 0) {
        for (size_t i = 0; i < batch_seq; i++) {
            for (size_t j = 0; j < d_ff; j++) {
                grad_hidden.set(i, j, grad_hidden.get(i, j) * dropout_mask.get(i, j));
            }
        }
    }
    
    // Backward through activation
    Matrix grad_hidden_pre = activation->backward(cached_hidden_pre, grad_hidden);
    
    // Gradient w.r.t W1 and b1
    dW1 = dW1 + (cached_input.transpose() * grad_hidden_pre);
    for (size_t j = 0; j < d_ff; j++) {
        for (size_t i = 0; i < batch_seq; i++) {
            db1.set(0, j, db1.get(0, j) + grad_hidden_pre.get(i, j));
        }
    }
    
    // Gradient w.r.t input
    Matrix grad_input = grad_hidden_pre * W1.transpose();
    
    return grad_input;
}

void PositionWiseFeedForward::updateParameters(double learning_rate) {
    // Update W1
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            W1.set(i, j, W1.get(i, j) - learning_rate * dW1.get(i, j));
            dW1.set(i, j, 0.0);
        }
    }
    
    // Update b1
    for (size_t j = 0; j < d_ff; j++) {
        b1.set(0, j, b1.get(0, j) - learning_rate * db1.get(0, j));
        db1.set(0, j, 0.0);
    }
    
    // Update W2
    for (size_t i = 0; i < d_ff; i++) {
        for (size_t j = 0; j < d_model; j++) {
            W2.set(i, j, W2.get(i, j) - learning_rate * dW2.get(i, j));
            dW2.set(i, j, 0.0);
        }
    }
    
    // Update b2
    for (size_t j = 0; j < d_model; j++) {
        b2.set(0, j, b2.get(0, j) - learning_rate * db2.get(0, j));
        db2.set(0, j, 0.0);
    }
}

void PositionWiseFeedForward::saveWeights(std::ofstream& out) const {
    // Save W1
    size_t rows = W1.getRows();
    size_t cols = W1.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = W1.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
    
    // Save b1
    rows = b1.getRows();
    cols = b1.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = b1.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
    
    // Save W2
    rows = W2.getRows();
    cols = W2.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = W2.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
    
    // Save b2
    rows = b2.getRows();
    cols = b2.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = b2.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
}

void PositionWiseFeedForward::loadWeights(std::ifstream& in) {
    size_t rows, cols;
    
    // Load W1
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            W1.set(i, j, val);
        }
    }
    
    // Load b1
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            b1.set(i, j, val);
        }
    }
    
    // Load W2
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            W2.set(i, j, val);
        }
    }
    
    // Load b2
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            b2.set(i, j, val);
        }
    }
}
