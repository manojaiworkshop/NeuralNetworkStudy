#include "../../include/nn/transformer/embedding.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

// ============ TokenEmbedding ============

TokenEmbedding::TokenEmbedding(size_t vocab_size, size_t embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    // Initialize embedding matrix: (vocab_size × embedding_dim)
    embeddings = Matrix(vocab_size, embedding_dim);
    
    // Xavier initialization
    double std = std::sqrt(2.0 / (vocab_size + embedding_dim));
    embeddings.randomNormal(0.0, std);
    
    // Initialize gradient matrix
    gradients = Matrix(vocab_size, embedding_dim, 0.0);
}

void TokenEmbedding::initializeEmbeddings(const std::string& strategy) {
    if (strategy == "random") {
        for (size_t i = 0; i < vocab_size; i++) {
            for (size_t j = 0; j < embedding_dim; j++) {
                embeddings.set(i, j, ((double)rand() / RAND_MAX) * 2.0 - 1.0);
            }
        }
    } else if (strategy == "normal") {
        embeddings.randomNormal(0.0, 1.0);
    } else if (strategy == "xavier") {
        double std = std::sqrt(2.0 / (vocab_size + embedding_dim));
        embeddings.randomNormal(0.0, std);
    }
}

Matrix TokenEmbedding::forward(const std::vector<std::vector<int>>& token_ids) {
    size_t batch_size = token_ids.size();
    size_t seq_len = token_ids[0].size();
    
    // Output: (batch_size × seq_len × embedding_dim)
    // Flattened as (batch_size * seq_len × embedding_dim)
    Matrix output(batch_size * seq_len, embedding_dim);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = token_ids[b][t];
            
            if (token_id < 0 || token_id >= (int)vocab_size) {
                throw std::runtime_error("Token ID out of vocabulary range");
            }
            
            // Copy embedding vector
            for (size_t d = 0; d < embedding_dim; d++) {
                output.set(b * seq_len + t, d, embeddings.get(token_id, d));
            }
        }
    }
    
    return output;
}

Matrix TokenEmbedding::forward(const std::vector<int>& token_ids) {
    // Single sequence forward
    std::vector<std::vector<int>> batch = {token_ids};
    return forward(batch);
}

void TokenEmbedding::backward(const Matrix& grad_output, 
                             const std::vector<std::vector<int>>& token_ids) {
    size_t batch_size = token_ids.size();
    size_t seq_len = token_ids[0].size();
    
    // Accumulate gradients for each token
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = token_ids[b][t];
            
            for (size_t d = 0; d < embedding_dim; d++) {
                double grad = gradients.get(token_id, d) + grad_output.get(b * seq_len + t, d);
                gradients.set(token_id, d, grad);
            }
        }
    }
}

void TokenEmbedding::updateParameters(double learning_rate) {
    for (size_t i = 0; i < vocab_size; i++) {
        for (size_t j = 0; j < embedding_dim; j++) {
            double val = embeddings.get(i, j) - learning_rate * gradients.get(i, j);
            embeddings.set(i, j, val);
            gradients.set(i, j, 0.0);  // Reset gradient
        }
    }
}

// ============ PositionalEncoding ============

PositionalEncoding::PositionalEncoding(size_t max_seq_len, size_t d_model)
    : max_seq_len(max_seq_len), d_model(d_model) {
    
    // Precompute positional encodings
    encoding = Matrix(max_seq_len, d_model);
    generateEncodings();
}

void PositionalEncoding::generateEncodings() {
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / d_model);
            
            if (i % 2 == 0) {
                encoding.set(pos, i, std::sin(angle));
            } else {
                encoding.set(pos, i, std::cos(angle));
            }
        }
    }
}

Matrix PositionalEncoding::forward(const Matrix& x, size_t seq_len) const {
    if (seq_len > max_seq_len) {
        throw std::runtime_error("Sequence length exceeds maximum");
    }
    
    size_t batch_size = x.getRows() / seq_len;
    Matrix output = x;  // Copy input
    
    // Add positional encoding to each position
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t d = 0; d < d_model; d++) {
                double val = output.get(b * seq_len + t, d) + encoding.get(t, d);
                output.set(b * seq_len + t, d, val);
            }
        }
    }
    
    return output;
}

Matrix PositionalEncoding::backward(const Matrix& grad_output) const {
    // Positional encoding has no learnable parameters
    // Gradient passes through unchanged
    return grad_output;
}

// ============ TransformerEmbedding ============

TransformerEmbedding::TransformerEmbedding(size_t vocab_size, size_t d_model,
                                         size_t max_seq_len, double dropout)
    : d_model(d_model), dropout_rate(dropout), training(false) {
    
    token_emb = std::make_unique<TokenEmbedding>(vocab_size, d_model);
    pos_enc = std::make_unique<PositionalEncoding>(max_seq_len, d_model);
    scale_factor = std::sqrt(static_cast<double>(d_model));
}

Matrix TransformerEmbedding::forward(const std::vector<std::vector<int>>& token_ids,
                                    bool training) {
    this->training = training;
    size_t seq_len = token_ids[0].size();
    
    // Token embedding
    Matrix embedded = token_emb->forward(token_ids);
    
    // Scale by sqrt(d_model) as in original paper
    for (size_t i = 0; i < embedded.getRows(); i++) {
        for (size_t j = 0; j < embedded.getCols(); j++) {
            embedded.set(i, j, embedded.get(i, j) * scale_factor);
        }
    }
    
    // Add positional encoding
    Matrix output = pos_enc->forward(embedded, seq_len);
    
    // Apply dropout
    if (training && dropout_rate > 0.0) {
        dropout_mask = Matrix(output.getRows(), output.getCols());
        
        for (size_t i = 0; i < output.getRows(); i++) {
            for (size_t j = 0; j < output.getCols(); j++) {
                double rand_val = (double)rand() / RAND_MAX;
                if (rand_val < dropout_rate) {
                    dropout_mask.set(i, j, 0.0);
                    output.set(i, j, 0.0);
                } else {
                    double mask_val = 1.0 / (1.0 - dropout_rate);
                    dropout_mask.set(i, j, mask_val);
                    output.set(i, j, output.get(i, j) * mask_val);
                }
            }
        }
    }
    
    return output;
}

Matrix TransformerEmbedding::forward(const std::vector<int>& token_ids, bool training) {
    // Single sequence forward
    std::vector<std::vector<int>> batch = {token_ids};
    return forward(batch, training);
}

void TransformerEmbedding::backward(const Matrix& grad_output, 
                                   const std::vector<std::vector<int>>& token_ids) {
    Matrix grad = grad_output;
    
    // Backward through dropout
    if (training && dropout_rate > 0.0) {
        for (size_t i = 0; i < grad.getRows(); i++) {
            for (size_t j = 0; j < grad.getCols(); j++) {
                grad.set(i, j, grad.get(i, j) * dropout_mask.get(i, j));
            }
        }
    }
    
    // Backward through positional encoding (no-op)
    grad = pos_enc->backward(grad);
    
    // Backward through scaling
    for (size_t i = 0; i < grad.getRows(); i++) {
        for (size_t j = 0; j < grad.getCols(); j++) {
            grad.set(i, j, grad.get(i, j) * scale_factor);
        }
    }
    
    // Backward through token embedding
    token_emb->backward(grad, token_ids);
}

void TransformerEmbedding::updateParameters(double learning_rate) {
    token_emb->updateParameters(learning_rate);
}

Matrix& TransformerEmbedding::getTokenEmbeddings() {
    return token_emb->getEmbeddings();
}

size_t TransformerEmbedding::getVocabSize() const {
    return token_emb->getVocabSize();
}

size_t TransformerEmbedding::getDModel() const {
    return d_model;
}
