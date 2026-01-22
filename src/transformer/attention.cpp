#include "../../include/nn/transformer/attention.h"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <algorithm>

// ============ Utility Functions ============

Matrix createPaddingMask(const std::vector<std::vector<int>>& tokens, int pad_id) {
    size_t batch_size = tokens.size();
    size_t seq_len = tokens[0].size();
    
    // Create mask: 1 for valid tokens, 0 for padding
    Matrix mask(batch_size, seq_len, 1.0);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            if (tokens[b][t] == pad_id) {
                mask.set(b, t, 0.0);
            }
        }
    }
    
    return mask;
}

Matrix createCausalMask(size_t seq_len) {
    // Lower triangular matrix (can attend to current and previous positions)
    Matrix mask(seq_len, seq_len, 0.0);
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j <= i; j++) {
            mask.set(i, j, 1.0);
        }
    }
    
    return mask;
}

// ============ ScaledDotProductAttention ============

ScaledDotProductAttention::ScaledDotProductAttention(size_t d_k, double dropout)
    : dropout_rate(dropout), scale_factor(1.0 / std::sqrt(static_cast<double>(d_k))) {}

Matrix ScaledDotProductAttention::forward(const Matrix& Q, const Matrix& K,
                                         const Matrix& V, const Matrix* mask,
                                         bool training) {
    size_t batch_heads_seq = Q.getRows();
    
    // Compute attention scores: QK^T / sqrt(d_k)
    Matrix K_T = K.transpose();
    Matrix scores = Q * K_T;
    
    // Scale scores
    for (size_t i = 0; i < scores.getRows(); i++) {
        for (size_t j = 0; j < scores.getCols(); j++) {
            scores.set(i, j, scores.get(i, j) * scale_factor);
        }
    }
    
    // Apply mask (set masked positions to large negative value)
    if (mask != nullptr) {
        for (size_t i = 0; i < scores.getRows(); i++) {
            for (size_t j = 0; j < scores.getCols(); j++) {
                if (mask->get(i % mask->getRows(), j % mask->getCols()) == 0.0) {
                    scores.set(i, j, -1e9);
                }
            }
        }
    }
    
    // Softmax over last dimension
    cached_attention_weights = Matrix(scores.getRows(), scores.getCols());
    for (size_t i = 0; i < scores.getRows(); i++) {
        // Find max for numerical stability
        double max_score = scores.get(i, 0);
        for (size_t j = 1; j < scores.getCols(); j++) {
            max_score = std::max(max_score, scores.get(i, j));
        }
        
        // Compute exp and sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < scores.getCols(); j++) {
            double val = std::exp(scores.get(i, j) - max_score);
            cached_attention_weights.set(i, j, val);
            sum_exp += val;
        }
        
        // Normalize
        for (size_t j = 0; j < scores.getCols(); j++) {
            cached_attention_weights.set(i, j, cached_attention_weights.get(i, j) / sum_exp);
        }
    }
    
    // Apply dropout to attention weights
    if (training && dropout_rate > 0.0) {
        dropout_mask = Matrix(cached_attention_weights.getRows(), cached_attention_weights.getCols());
        
        for (size_t i = 0; i < cached_attention_weights.getRows(); i++) {
            for (size_t j = 0; j < cached_attention_weights.getCols(); j++) {
                double rand_val = (double)rand() / RAND_MAX;
                if (rand_val < dropout_rate) {
                    dropout_mask.set(i, j, 0.0);
                    cached_attention_weights.set(i, j, 0.0);
                } else {
                    double scale = 1.0 / (1.0 - dropout_rate);
                    dropout_mask.set(i, j, scale);
                    cached_attention_weights.set(i, j, cached_attention_weights.get(i, j) * scale);
                }
            }
        }
    }
    
    // Cache for backward
    cached_Q = Q;
    cached_K = K;
    cached_V = V;
    
    // Compute output: attention_weights * V
    Matrix output = cached_attention_weights * V;
    
    return output;
}

void ScaledDotProductAttention::backward(
    const Matrix& grad_output, Matrix& dQ, Matrix& dK, Matrix& dV) {
    
    // Gradient w.r.t V
    dV = cached_attention_weights.transpose() * grad_output;
    
    // Gradient w.r.t attention_weights
    Matrix grad_attn = grad_output * cached_V.transpose();
    
    // Backward through dropout
    if (dropout_rate > 0.0 && dropout_mask.getRows() > 0) {
        for (size_t i = 0; i < grad_attn.getRows(); i++) {
            for (size_t j = 0; j < grad_attn.getCols(); j++) {
                grad_attn.set(i, j, grad_attn.get(i, j) * dropout_mask.get(i, j));
            }
        }
    }
    
    // Backward through softmax
    Matrix grad_scores(grad_attn.getRows(), grad_attn.getCols(), 0.0);
    for (size_t i = 0; i < grad_attn.getRows(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < grad_attn.getCols(); j++) {
            sum += grad_attn.get(i, j) * cached_attention_weights.get(i, j);
        }
        
        for (size_t j = 0; j < grad_attn.getCols(); j++) {
            double val = cached_attention_weights.get(i, j) * (grad_attn.get(i, j) - sum);
            grad_scores.set(i, j, val * scale_factor);
        }
    }
    
    // Gradient w.r.t Q and K
    dQ = grad_scores * cached_K;
    dK = grad_scores.transpose() * cached_Q;
}

// ============ MultiHeadAttention ============

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads, double dropout)
    : d_model(d_model), num_heads(num_heads), dropout_rate(dropout) {
    
    if (d_model % num_heads != 0) {
        throw std::runtime_error("d_model must be divisible by num_heads");
    }
    
    d_k = d_model / num_heads;
    d_v = d_k;  // Usually same as d_k
    
    // Initialize projection matrices (single matrices, not vectors)
    W_Q.resize(num_heads);
    W_K.resize(num_heads);
    W_V.resize(num_heads);
    
    dW_Q.resize(num_heads);
    dW_K.resize(num_heads);
    dW_V.resize(num_heads);
    
    // Xavier initialization
    double std = std::sqrt(2.0 / (2.0 * d_model));
    for (size_t h = 0; h < num_heads; h++) {
        W_Q[h] = Matrix(d_model, d_k);
        W_K[h] = Matrix(d_model, d_k);
        W_V[h] = Matrix(d_model, d_v);
        
        W_Q[h].randomNormal(0.0, std);
        W_K[h].randomNormal(0.0, std);
        W_V[h].randomNormal(0.0, std);
        
        dW_Q[h] = Matrix(d_model, d_k, 0.0);
        dW_K[h] = Matrix(d_model, d_k, 0.0);
        dW_V[h] = Matrix(d_model, d_v, 0.0);
    }
    
    W_O = Matrix(d_model, d_model);
    W_O.randomNormal(0.0, std);
    dW_O = Matrix(d_model, d_model, 0.0);
    
    // Initialize attention mechanisms for each head
    for (size_t h = 0; h < num_heads; h++) {
        attention_heads.push_back(std::make_unique<ScaledDotProductAttention>(d_k, dropout));
    }
}

Matrix MultiHeadAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V,
                                   const Matrix* mask, bool training) {
    size_t batch_seq = Q.getRows();
    
    // Cache inputs
    cached_Q = Q;
    cached_K = K;
    cached_V = V;
    cached_heads.clear();
    
    // Process each head
    std::vector<Matrix> head_outputs;
    for (size_t h = 0; h < num_heads; h++) {
        // Project Q, K, V for this head
        Matrix Q_h = Q * W_Q[h];  // (batch_seq × d_k)
        Matrix K_h = K * W_K[h];
        Matrix V_h = V * W_V[h];
        
        // Apply attention for this head
        Matrix head_out = attention_heads[h]->forward(Q_h, K_h, V_h, mask, training);
        head_outputs.push_back(head_out);
    }
    
    // Concatenate all heads
    Matrix concat(batch_seq, d_model);
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t j = 0; j < d_k; j++) {
                concat.set(i, h * d_k + j, head_outputs[h].get(i, j));
            }
        }
    }
    
    cached_concat = concat;
    cached_heads = head_outputs;
    
    // Final linear projection
    Matrix output = concat * W_O;
    
    return output;
}

void MultiHeadAttention::backward(const Matrix& grad_output,
                                 Matrix& dQ, Matrix& dK, Matrix& dV) {
    size_t batch_seq = cached_Q.getRows();
    
    // Gradient w.r.t W_O
    dW_O = dW_O + (cached_concat.transpose() * grad_output);
    
    // Gradient w.r.t concatenated heads
    Matrix grad_concat = grad_output * W_O.transpose();
    
    // Initialize output gradients
    dQ = Matrix(batch_seq, d_model, 0.0);
    dK = Matrix(batch_seq, d_model, 0.0);
    dV = Matrix(batch_seq, d_model, 0.0);
    
    // Backprop through each head
    for (size_t h = 0; h < num_heads; h++) {
        // Extract gradient for this head
        Matrix grad_head(batch_seq, d_k);
        for (size_t i = 0; i < batch_seq; i++) {
            for (size_t j = 0; j < d_k; j++) {
                grad_head.set(i, j, grad_concat.get(i, h * d_k + j));
            }
        }
        
        // Backward through attention
        Matrix dQ_h, dK_h, dV_h;
        attention_heads[h]->backward(grad_head, dQ_h, dK_h, dV_h);
        
        // Gradient w.r.t projection weights
        Matrix Q_h = cached_Q * W_Q[h];
        Matrix K_h = cached_K * W_K[h];
        Matrix V_h = cached_V * W_V[h];
        
        dW_Q[h] = dW_Q[h] + (cached_Q.transpose() * dQ_h);
        dW_K[h] = dW_K[h] + (cached_K.transpose() * dK_h);
        dW_V[h] = dW_V[h] + (cached_V.transpose() * dV_h);
        
        // Accumulate gradients for inputs
        Matrix dQ_proj = dQ_h * W_Q[h].transpose();
        Matrix dK_proj = dK_h * W_K[h].transpose();
        Matrix dV_proj = dV_h * W_V[h].transpose();
        
        dQ = dQ + dQ_proj;
        dK = dK + dK_proj;
        dV = dV + dV_proj;
    }
}

void MultiHeadAttention::updateParameters(double learning_rate) {
    // Update all head projections
    for (size_t h = 0; h < num_heads; h++) {
        for (size_t i = 0; i < W_Q[h].getRows(); i++) {
            for (size_t j = 0; j < W_Q[h].getCols(); j++) {
                W_Q[h].set(i, j, W_Q[h].get(i, j) - learning_rate * dW_Q[h].get(i, j));
                W_K[h].set(i, j, W_K[h].get(i, j) - learning_rate * dW_K[h].get(i, j));
                W_V[h].set(i, j, W_V[h].get(i, j) - learning_rate * dW_V[h].get(i, j));
                
                dW_Q[h].set(i, j, 0.0);
                dW_K[h].set(i, j, 0.0);
                dW_V[h].set(i, j, 0.0);
            }
        }
    }
    
    // Update output projection
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < d_model; j++) {
            W_O.set(i, j, W_O.get(i, j) - learning_rate * dW_O.get(i, j));
            dW_O.set(i, j, 0.0);
        }
    }
}

std::vector<Matrix> MultiHeadAttention::getAllAttentionWeights() const {
    std::vector<Matrix> weights;
    for (const auto& head : attention_heads) {
        weights.push_back(head->getAttentionWeights());
    }
    return weights;
}

void MultiHeadAttention::saveWeights(std::ofstream& out) const {
    // Save number of heads (for verification)
    size_t num_heads_save = num_heads;
    out.write(reinterpret_cast<const char*>(&num_heads_save), sizeof(size_t));
    
    // Save Q, K, V projections for each head
    for (size_t h = 0; h < num_heads; h++) {
        size_t rows = W_Q[h].getRows();
        size_t cols = W_Q[h].getCols();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = W_Q[h].get(i, j);
                out.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = W_K[h].get(i, j);
                out.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val = W_V[h].get(i, j);
                out.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
    }
    
    // Save output projection W_O
    size_t rows = W_O.getRows();
    size_t cols = W_O.getCols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = W_O.get(i, j);
            out.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }
    }
}

void MultiHeadAttention::loadWeights(std::ifstream& in) {
    // Load and verify number of heads
    size_t num_heads_load;
    in.read(reinterpret_cast<char*>(&num_heads_load), sizeof(size_t));
    if (num_heads_load != num_heads) {
        throw std::runtime_error("Mismatch in number of attention heads");
    }
    
    // Load Q, K, V projections for each head
    for (size_t h = 0; h < num_heads; h++) {
        size_t rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val;
                in.read(reinterpret_cast<char*>(&val), sizeof(double));
                W_Q[h].set(i, j, val);
            }
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val;
                in.read(reinterpret_cast<char*>(&val), sizeof(double));
                W_K[h].set(i, j, val);
            }
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                double val;
                in.read(reinterpret_cast<char*>(&val), sizeof(double));
                W_V[h].set(i, j, val);
            }
        }
    }
    
    // Load output projection W_O
    size_t rows, cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val;
            in.read(reinterpret_cast<char*>(&val), sizeof(double));
            W_O.set(i, j, val);
        }
    }
}
