#include "../../include/nn/transformer/encoder.h"
#include <fstream>
#include <stdexcept>

// ============ EncoderLayer ============

EncoderLayer::EncoderLayer(size_t d_model, size_t num_heads, size_t d_ff, double dropout)
    : d_model(d_model) {
    
    // Initialize components
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads, dropout);
    feed_forward = std::make_unique<PositionWiseFeedForward>(d_model, d_ff, dropout);
    norm1 = std::make_unique<LayerNormalization>(d_model);
    norm2 = std::make_unique<LayerNormalization>(d_model);
}

Matrix EncoderLayer::forward(const Matrix& x, const Matrix* mask, bool training) {
    // Self-attention with residual connection and layer norm
    Matrix dQ, dK, dV;
    Matrix attn_out = self_attention->forward(x, x, x, mask, training);
    
    // Add & Norm (residual connection)
    Matrix residual1(x.getRows(), x.getCols());
    for (size_t i = 0; i < x.getRows(); i++) {
        for (size_t j = 0; j < x.getCols(); j++) {
            residual1.set(i, j, x.get(i, j) + attn_out.get(i, j));
        }
    }
    Matrix norm1_out = norm1->forward(residual1);
    
    // Feed-forward with residual connection and layer norm
    Matrix ff_out = feed_forward->forward(norm1_out, training);
    
    // Add & Norm
    Matrix residual2(norm1_out.getRows(), norm1_out.getCols());
    for (size_t i = 0; i < norm1_out.getRows(); i++) {
        for (size_t j = 0; j < norm1_out.getCols(); j++) {
            residual2.set(i, j, norm1_out.get(i, j) + ff_out.get(i, j));
        }
    }
    Matrix output = norm2->forward(residual2);
    
    // Cache for backward
    cached_input = x;
    cached_attn_output = attn_out;
    cached_ffn_input = norm1_out;
    
    return output;
}

Matrix EncoderLayer::backward(const Matrix& grad_output) {
    size_t rows = grad_output.getRows();
    size_t cols = grad_output.getCols();
    
    // Backward through norm2
    Matrix grad_residual2 = norm2->backward(grad_output);
    
    // Split gradient for residual connection
    Matrix grad_norm1_out(rows, cols);
    Matrix grad_ff_out = grad_residual2;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm1_out.set(i, j, grad_residual2.get(i, j));
        }
    }
    
    // Backward through feed-forward
    Matrix grad_ff = feed_forward->backward(grad_ff_out);
    
    // Add gradient from residual
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm1_out.set(i, j, grad_norm1_out.get(i, j) + grad_ff.get(i, j));
        }
    }
    
    // Backward through norm1
    Matrix grad_residual1 = norm1->backward(grad_norm1_out);
    
    // Split gradient for residual connection
    Matrix grad_x(rows, cols);
    Matrix grad_attn = grad_residual1;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_x.set(i, j, grad_residual1.get(i, j));
        }
    }
    
    // Backward through self-attention
    Matrix grad_q(rows, cols);
    Matrix grad_k(rows, cols);
    Matrix grad_v(rows, cols);
    self_attention->backward(grad_attn, grad_q, grad_k, grad_v);
    
    // Self-attention has Q=K=V=x, so sum all gradients
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_x.set(i, j, grad_x.get(i, j) + grad_q.get(i, j) + grad_k.get(i, j) + grad_v.get(i, j));
        }
    }
    
    return grad_x;
}

void EncoderLayer::updateParameters(double learning_rate) {
    self_attention->updateParameters(learning_rate);
    feed_forward->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
}

std::vector<Matrix> EncoderLayer::getAttentionWeights() const {
    return self_attention->getAllAttentionWeights();
}

void EncoderLayer::saveWeights(std::ofstream& out) const {
    self_attention->saveWeights(out);
    feed_forward->saveWeights(out);
    norm1->saveWeights(out);
    norm2->saveWeights(out);
}

void EncoderLayer::loadWeights(std::ifstream& in) {
    self_attention->loadWeights(in);
    feed_forward->loadWeights(in);
    norm1->loadWeights(in);
    norm2->loadWeights(in);
}

// ============ TransformerEncoder ============

TransformerEncoder::TransformerEncoder(size_t num_layers, size_t d_model,
                                       size_t num_heads, size_t d_ff, double dropout)
    : num_layers(num_layers), d_model(d_model) {
    
    // Create encoder layers
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<EncoderLayer>(d_model, num_heads, d_ff, dropout));
    }
}

Matrix TransformerEncoder::forward(const Matrix& x, const Matrix* mask, bool training) {
    Matrix output = x;
    
    // Pass through each encoder layer
    for (size_t i = 0; i < num_layers; i++) {
        output = layers[i]->forward(output, mask, training);
    }
    
    return output;
}

Matrix TransformerEncoder::backward(const Matrix& grad_output) {
    Matrix grad = grad_output;
    
    // Backward through layers in reverse order
    for (int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
    }
    
    return grad;
}

void TransformerEncoder::updateParameters(double learning_rate) {
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->updateParameters(learning_rate);
    }
}

std::vector<std::vector<Matrix>> TransformerEncoder::getAllAttentionWeights() const {
    std::vector<std::vector<Matrix>> attention_weights;
    
    for (size_t i = 0; i < num_layers; i++) {
        attention_weights.push_back(layers[i]->getAttentionWeights());
    }
    
    return attention_weights;
}

void TransformerEncoder::saveWeights(std::ofstream& out) const {
    // Save number of layers (for verification)
    size_t num_layers_save = num_layers;
    out.write(reinterpret_cast<const char*>(&num_layers_save), sizeof(size_t));
    
    // Save each layer
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->saveWeights(out);
    }
}

void TransformerEncoder::loadWeights(std::ifstream& in) {
    // Load and verify number of layers
    size_t num_layers_load;
    in.read(reinterpret_cast<char*>(&num_layers_load), sizeof(size_t));
    if (num_layers_load != num_layers) {
        throw std::runtime_error("Mismatch in number of encoder layers");
    }
    
    // Load each layer
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->loadWeights(in);
    }
}
