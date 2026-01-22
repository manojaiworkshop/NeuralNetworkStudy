#include "../../include/nn/transformer/decoder.h"
#include <stdexcept>

// ============ DecoderLayer ============

DecoderLayer::DecoderLayer(size_t d_model, size_t num_heads, size_t d_ff, double dropout)
    : d_model(d_model) {
    
    // Initialize components
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads, dropout);
    cross_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads, dropout);
    feed_forward = std::make_unique<PositionWiseFeedForward>(d_model, d_ff, dropout);
    norm1 = std::make_unique<LayerNormalization>(d_model);
    norm2 = std::make_unique<LayerNormalization>(d_model);
    norm3 = std::make_unique<LayerNormalization>(d_model);
}

Matrix DecoderLayer::forward(const Matrix& x, const Matrix& encoder_output,
                            const Matrix* self_mask, const Matrix* cross_mask,
                            bool training) {
    // Masked self-attention with residual connection and layer norm
    Matrix self_attn_out = self_attention->forward(x, x, x, self_mask, training);
    
    Matrix residual1(x.getRows(), x.getCols());
    for (size_t i = 0; i < x.getRows(); i++) {
        for (size_t j = 0; j < x.getCols(); j++) {
            residual1.set(i, j, x.get(i, j) + self_attn_out.get(i, j));
        }
    }
    Matrix norm1_out = norm1->forward(residual1);
    
    // Cross-attention with residual connection and layer norm
    Matrix cross_attn_out = cross_attention->forward(
        norm1_out, encoder_output, encoder_output, cross_mask, training);
    
    Matrix residual2(norm1_out.getRows(), norm1_out.getCols());
    for (size_t i = 0; i < norm1_out.getRows(); i++) {
        for (size_t j = 0; j < norm1_out.getCols(); j++) {
            residual2.set(i, j, norm1_out.get(i, j) + cross_attn_out.get(i, j));
        }
    }
    Matrix norm2_out = norm2->forward(residual2);
    
    // Feed-forward with residual connection and layer norm
    Matrix ff_out = feed_forward->forward(norm2_out, training);
    
    Matrix residual3(norm2_out.getRows(), norm2_out.getCols());
    for (size_t i = 0; i < norm2_out.getRows(); i++) {
        for (size_t j = 0; j < norm2_out.getCols(); j++) {
            residual3.set(i, j, norm2_out.get(i, j) + ff_out.get(i, j));
        }
    }
    Matrix output = norm3->forward(residual3);
    
    // Cache for backward
    cached_input = x;
    cached_encoder_output = encoder_output;
    cached_self_attn_output = self_attn_out;
    cached_cross_attn_output = cross_attn_out;
    cached_ffn_input = norm2_out;
    
    return output;
}

Matrix DecoderLayer::backward(const Matrix& grad_output, Matrix& grad_encoder) {
    size_t rows = grad_output.getRows();
    size_t cols = grad_output.getCols();
    
    // Backward through norm3
    Matrix grad_residual3 = norm3->backward(grad_output);
    
    // Split gradient for residual connection
    Matrix grad_norm2_out(rows, cols);
    Matrix grad_ff_out = grad_residual3;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm2_out.set(i, j, grad_residual3.get(i, j));
        }
    }
    
    // Backward through feed-forward
    Matrix grad_ff = feed_forward->backward(grad_ff_out);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm2_out.set(i, j, grad_norm2_out.get(i, j) + grad_ff.get(i, j));
        }
    }
    
    // Backward through norm2
    Matrix grad_residual2 = norm2->backward(grad_norm2_out);
    
    // Split gradient for residual connection
    Matrix grad_norm1_out(rows, cols);
    Matrix grad_cross_attn = grad_residual2;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm1_out.set(i, j, grad_residual2.get(i, j));
        }
    }
    
    // Backward through cross-attention
    Matrix grad_q_cross(rows, cols);
    Matrix grad_k_cross(cached_encoder_output.getRows(), cached_encoder_output.getCols());
    Matrix grad_v_cross(cached_encoder_output.getRows(), cached_encoder_output.getCols());
    cross_attention->backward(grad_cross_attn, grad_q_cross, grad_k_cross, grad_v_cross);
    
    // grad_q goes to decoder, grad_k and grad_v go to encoder
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_norm1_out.set(i, j, grad_norm1_out.get(i, j) + grad_q_cross.get(i, j));
        }
    }
    
    // Gradient for encoder output
    grad_encoder = Matrix(grad_k_cross.getRows(), grad_k_cross.getCols());
    for (size_t i = 0; i < grad_encoder.getRows(); i++) {
        for (size_t j = 0; j < grad_encoder.getCols(); j++) {
            grad_encoder.set(i, j, grad_k_cross.get(i, j) + grad_v_cross.get(i, j));
        }
    }
    
    // Backward through norm1
    Matrix grad_residual1 = norm1->backward(grad_norm1_out);
    
    // Split gradient for residual connection
    Matrix grad_x(rows, cols);
    Matrix grad_self_attn = grad_residual1;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_x.set(i, j, grad_residual1.get(i, j));
        }
    }
    
    // Backward through self-attention
    Matrix grad_q_self(rows, cols);
    Matrix grad_k_self(rows, cols);
    Matrix grad_v_self(rows, cols);
    self_attention->backward(grad_self_attn, grad_q_self, grad_k_self, grad_v_self);
    
    // Self-attention has Q=K=V=x, so sum all gradients
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            grad_x.set(i, j, grad_x.get(i, j) + grad_q_self.get(i, j) + grad_k_self.get(i, j) + grad_v_self.get(i, j));
        }
    }
    
    return grad_x;
}

void DecoderLayer::updateParameters(double learning_rate) {
    self_attention->updateParameters(learning_rate);
    cross_attention->updateParameters(learning_rate);
    feed_forward->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
    norm3->updateParameters(learning_rate);
}

std::vector<Matrix> DecoderLayer::getSelfAttentionWeights() const {
    return self_attention->getAllAttentionWeights();
}

std::vector<Matrix> DecoderLayer::getCrossAttentionWeights() const {
    return cross_attention->getAllAttentionWeights();
}

// ============ TransformerDecoder ============

TransformerDecoder::TransformerDecoder(size_t num_layers, size_t d_model,
                                       size_t num_heads, size_t d_ff, double dropout)
    : num_layers(num_layers), d_model(d_model) {
    
    // Create decoder layers
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<DecoderLayer>(d_model, num_heads, d_ff, dropout));
    }
}

Matrix TransformerDecoder::forward(const Matrix& x, const Matrix& encoder_output,
                                   const Matrix* self_mask, const Matrix* cross_mask,
                                   bool training) {
    Matrix output = x;
    
    // Pass through each decoder layer
    for (size_t i = 0; i < num_layers; i++) {
        output = layers[i]->forward(output, encoder_output, self_mask, cross_mask, training);
    }
    
    return output;
}

Matrix TransformerDecoder::backward(const Matrix& grad_output, Matrix& grad_encoder) {
    Matrix grad_decoder = grad_output;
    grad_encoder = Matrix(grad_output.getRows(), grad_output.getCols(), 0.0);
    
    // Backward through layers in reverse order
    for (int i = num_layers - 1; i >= 0; i--) {
        Matrix grad_enc(grad_decoder.getRows(), grad_decoder.getCols());
        grad_decoder = layers[i]->backward(grad_decoder, grad_enc);
        
        // Accumulate encoder gradients from all layers
        for (size_t r = 0; r < grad_encoder.getRows(); r++) {
            for (size_t c = 0; c < grad_encoder.getCols(); c++) {
                grad_encoder.set(r, c, grad_encoder.get(r, c) + grad_enc.get(r, c));
            }
        }
    }
    
    return grad_decoder;
}

void TransformerDecoder::updateParameters(double learning_rate) {
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->updateParameters(learning_rate);
    }
}
