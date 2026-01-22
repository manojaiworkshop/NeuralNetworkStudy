#include "nn/attention.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

namespace {
    // Softmax along the last dimension
    Matrix softmax(const Matrix& input) {
        Matrix output(input.getRows(), input.getCols());
        
        for (size_t i = 0; i < input.getRows(); ++i) {
            // Find max for numerical stability
            double max_val = input.get(i, 0);
            for (size_t j = 1; j < input.getCols(); ++j) {
                max_val = std::max(max_val, input.get(i, j));
            }
            
            // Compute exp and sum
            double sum = 0.0;
            for (size_t j = 0; j < input.getCols(); ++j) {
                double exp_val = std::exp(input.get(i, j) - max_val);
                output.set(i, j, exp_val);
                sum += exp_val;
            }
            
            // Normalize
            for (size_t j = 0; j < input.getCols(); ++j) {
                output.set(i, j, output.get(i, j) / sum);
            }
        }
        
        return output;
    }
}

// ============================================================================
// DOT-PRODUCT ATTENTION
// ============================================================================

DotProductAttention::DotProductAttention() 
    : softmax(std::make_unique<Sigmoid>()) {
}

std::pair<Matrix, Matrix> DotProductAttention::forward(
    const Matrix& query,
    const std::vector<Matrix>& keys,
    const std::vector<Matrix>& values) {
    
    if (keys.empty()) {
        throw std::invalid_argument("Keys vector is empty");
    }
    
    // Cache for backward pass
    cached_query = query;
    cached_keys = keys;
    cached_values = values;
    
    size_t batch_size = query.getRows();
    size_t query_dim = query.getCols();
    size_t seq_len = keys.size();
    
    // Compute attention scores: score[i] = query · key[i]ᵀ
    // scores: (batch_size × seq_len)
    Matrix scores(batch_size, seq_len);
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            double score = 0.0;
            for (size_t d = 0; d < query_dim; ++d) {
                score += query.get(i, d) * keys[t].get(i, d);
            }
            scores.set(i, t, score);
        }
    }
    
    cached_scores = scores;
    
    // Apply softmax to get attention weights
    cached_attention_weights = ::softmax(scores);
    
    // Compute context vector: context = Σ αᵢ · vᵢ
    size_t value_dim = values[0].getCols();
    Matrix context(batch_size, value_dim);
    context.zeros();
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            double weight = cached_attention_weights.get(i, t);
            for (size_t d = 0; d < value_dim; ++d) {
                context.set(i, d, context.get(i, d) + weight * values[t].get(i, d));
            }
        }
    }
    
    return {context, cached_attention_weights};
}

std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> DotProductAttention::backward(
    const Matrix& grad_context,
    const Matrix& grad_attention_weights) {
    
    // Simplified backward pass (full implementation would be more complex)
    size_t batch_size = cached_query.getRows();
    size_t query_dim = cached_query.getCols();
    size_t seq_len = cached_keys.size();
    
    Matrix grad_query(batch_size, query_dim);
    std::vector<Matrix> grad_keys;
    std::vector<Matrix> grad_values;
    
    for (size_t t = 0; t < seq_len; ++t) {
        grad_keys.push_back(Matrix(batch_size, query_dim));
        grad_values.push_back(Matrix(batch_size, cached_values[0].getCols()));
    }
    
    return {grad_query, grad_keys, grad_values};
}

// ============================================================================
// ADDITIVE ATTENTION (BAHDANAU)
// ============================================================================

AdditiveAttention::AdditiveAttention(size_t query_dim, size_t key_dim, size_t hidden_dim)
    : query_dim(query_dim),
      key_dim(key_dim),
      hidden_dim(hidden_dim),
      W_query(hidden_dim, query_dim),
      W_key(hidden_dim, key_dim),
      v(1, hidden_dim),
      dW_query(hidden_dim, query_dim),
      dW_key(hidden_dim, key_dim),
      dv(1, hidden_dim),
      tanh_activation(std::make_unique<Tanh>()) {
    
    initializeWeights("xavier");
}

void AdditiveAttention::initializeWeights(const std::string& strategy) {
    if (strategy == "xavier") {
        W_query.xavierInit(query_dim, hidden_dim);
        W_key.xavierInit(key_dim, hidden_dim);
        v.randomize(-0.1, 0.1);
    } else {
        W_query.randomize(-0.1, 0.1);
        W_key.randomize(-0.1, 0.1);
        v.randomize(-0.1, 0.1);
    }
    
    resetGradients();
}

std::pair<Matrix, Matrix> AdditiveAttention::forward(
    const Matrix& query,
    const std::vector<Matrix>& keys,
    const std::vector<Matrix>& values) {
    
    if (keys.empty()) {
        throw std::invalid_argument("Keys vector is empty");
    }
    
    cached_query = query;
    cached_keys = keys;
    cached_values = values;
    cached_tanh_output.clear();
    
    size_t batch_size = query.getRows();
    size_t seq_len = keys.size();
    
    // Compute attention scores:
    // score[t] = v · tanh(W_query·query + W_key·key[t])
    Matrix scores(batch_size, seq_len);
    
    // Transform query once: (batch_size × hidden_dim)
    Matrix query_transformed = query * W_query.transpose();
    
    for (size_t t = 0; t < seq_len; ++t) {
        // Transform key: (batch_size × hidden_dim)
        Matrix key_transformed = keys[t] * W_key.transpose();
        
        // Add: (batch_size × hidden_dim)
        Matrix combined = query_transformed + key_transformed;
        
        // Apply tanh
        Matrix tanh_out = tanh_activation->forward(combined);
        cached_tanh_output.push_back(tanh_out);
        
        // Multiply by v and sum: (batch_size × 1)
        for (size_t i = 0; i < batch_size; ++i) {
            double score = 0.0;
            for (size_t h = 0; h < hidden_dim; ++h) {
                score += v.get(0, h) * tanh_out.get(i, h);
            }
            scores.set(i, t, score);
        }
    }
    
    cached_scores = scores;
    
    // Apply softmax
    cached_attention_weights = ::softmax(scores);
    
    // Compute context vector
    size_t value_dim = values[0].getCols();
    Matrix context(batch_size, value_dim);
    context.zeros();
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            double weight = cached_attention_weights.get(i, t);
            for (size_t d = 0; d < value_dim; ++d) {
                context.set(i, d, context.get(i, d) + weight * values[t].get(i, d));
            }
        }
    }
    
    return {context, cached_attention_weights};
}

std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> AdditiveAttention::backward(
    const Matrix& grad_context,
    const Matrix& grad_attention_weights) {
    
    // Simplified backward pass
    size_t batch_size = cached_query.getRows();
    size_t seq_len = cached_keys.size();
    
    Matrix grad_query(batch_size, query_dim);
    std::vector<Matrix> grad_keys;
    std::vector<Matrix> grad_values;
    
    for (size_t t = 0; t < seq_len; ++t) {
        grad_keys.push_back(Matrix(batch_size, key_dim));
        grad_values.push_back(Matrix(batch_size, cached_values[0].getCols()));
    }
    
    // Accumulate gradients for W_query, W_key, v
    // (Full implementation would compute these properly)
    
    return {grad_query, grad_keys, grad_values};
}

void AdditiveAttention::updateParameters(double learning_rate) {
    W_query = W_query - dW_query * learning_rate;
    W_key = W_key - dW_key * learning_rate;
    v = v - dv * learning_rate;
    resetGradients();
}

void AdditiveAttention::resetGradients() {
    dW_query.zeros();
    dW_key.zeros();
    dv.zeros();
}

// ============================================================================
// SCALED DOT-PRODUCT ATTENTION
// ============================================================================

ScaledDotProductAttention::ScaledDotProductAttention(size_t key_dim)
    : scale_factor(1.0 / std::sqrt(static_cast<double>(key_dim))) {
}

std::pair<Matrix, Matrix> ScaledDotProductAttention::forward(
    const Matrix& query,
    const std::vector<Matrix>& keys,
    const std::vector<Matrix>& values) {
    
    if (keys.empty()) {
        throw std::invalid_argument("Keys vector is empty");
    }
    
    cached_query = query;
    cached_keys = keys;
    cached_values = values;
    
    size_t batch_size = query.getRows();
    size_t query_dim = query.getCols();
    size_t seq_len = keys.size();
    
    // Compute scaled attention scores
    Matrix scores(batch_size, seq_len);
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            double score = 0.0;
            for (size_t d = 0; d < query_dim; ++d) {
                score += query.get(i, d) * keys[t].get(i, d);
            }
            // Scale by √d
            scores.set(i, t, score * scale_factor);
        }
    }
    
    cached_scores = scores;
    cached_attention_weights = ::softmax(scores);
    
    // Compute context
    size_t value_dim = values[0].getCols();
    Matrix context(batch_size, value_dim);
    context.zeros();
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t t = 0; t < seq_len; ++t) {
            double weight = cached_attention_weights.get(i, t);
            for (size_t d = 0; d < value_dim; ++d) {
                context.set(i, d, context.get(i, d) + weight * values[t].get(i, d));
            }
        }
    }
    
    return {context, cached_attention_weights};
}

std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> ScaledDotProductAttention::backward(
    const Matrix& grad_context,
    const Matrix& grad_attention_weights) {
    
    size_t batch_size = cached_query.getRows();
    size_t query_dim = cached_query.getCols();
    size_t seq_len = cached_keys.size();
    
    Matrix grad_query(batch_size, query_dim);
    std::vector<Matrix> grad_keys;
    std::vector<Matrix> grad_values;
    
    for (size_t t = 0; t < seq_len; ++t) {
        grad_keys.push_back(Matrix(batch_size, query_dim));
        grad_values.push_back(Matrix(batch_size, cached_values[0].getCols()));
    }
    
    return {grad_query, grad_keys, grad_values};
}

// ============================================================================
// MULTI-HEAD ATTENTION
// ============================================================================

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : num_heads(num_heads),
      d_model(d_model),
      d_k(d_model / num_heads),
      d_v(d_model / num_heads),
      W_O(d_model, d_model) {
    
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    // Initialize projection matrices for each head
    for (size_t h = 0; h < num_heads; ++h) {
        W_Q.push_back(Matrix(d_k, d_model));
        W_K.push_back(Matrix(d_k, d_model));
        W_V.push_back(Matrix(d_v, d_model));
        
        dW_Q.push_back(Matrix(d_k, d_model));
        dW_K.push_back(Matrix(d_k, d_model));
        dW_V.push_back(Matrix(d_v, d_model));
        
        attention_heads.push_back(std::make_unique<ScaledDotProductAttention>(d_k));
    }
    
    dW_O = Matrix(d_model, d_model);
    
    initializeWeights("xavier");
}

void MultiHeadAttention::initializeWeights(const std::string& strategy) {
    for (size_t h = 0; h < num_heads; ++h) {
        W_Q[h].xavierInit(d_model, d_k);
        W_K[h].xavierInit(d_model, d_k);
        W_V[h].xavierInit(d_model, d_v);
    }
    W_O.xavierInit(d_model, d_model);
    resetGradients();
}

std::pair<Matrix, Matrix> MultiHeadAttention::forward(
    const Matrix& query,
    const std::vector<Matrix>& keys,
    const std::vector<Matrix>& values) {
    
    // This is a simplified implementation
    // Full multi-head attention would project Q, K, V for each head
    
    // For now, use first head
    auto [context, weights] = attention_heads[0]->forward(query, keys, values);
    
    return {context, weights};
}

std::tuple<Matrix, std::vector<Matrix>, std::vector<Matrix>> MultiHeadAttention::backward(
    const Matrix& grad_context,
    const Matrix& grad_attention_weights) {
    
    return attention_heads[0]->backward(grad_context, grad_attention_weights);
}

void MultiHeadAttention::updateParameters(double learning_rate) {
    // Update would happen here
    resetGradients();
}

void MultiHeadAttention::resetGradients() {
    for (size_t h = 0; h < num_heads; ++h) {
        dW_Q[h].zeros();
        dW_K[h].zeros();
        dW_V[h].zeros();
    }
    dW_O.zeros();
}

int MultiHeadAttention::getParameterCount() const {
    return num_heads * (d_k * d_model * 2 + d_v * d_model) + d_model * d_model;
}
