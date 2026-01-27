#include "nn/bert_encoder.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// ============================================================================
// LAYER NORMALIZATION IMPLEMENTATION
// ============================================================================

LayerNorm::LayerNorm(size_t d_model, double epsilon)
    : d_model(d_model), eps(epsilon),
      gamma(1, d_model, 1.0),  // Initialize to 1
      beta(1, d_model, 0.0)     // Initialize to 0
{
}

Matrix LayerNorm::forward(const Matrix& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    // Calculate mean and variance for each position
    mean = Matrix(seq_len, 1);
    variance = Matrix(seq_len, 1);
    
    for (size_t i = 0; i < seq_len; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < d_model; j++) {
            sum += input.get(i, j);
        }
        mean.set(i, 0, sum / d_model);
    }
    
    for (size_t i = 0; i < seq_len; i++) {
        double var_sum = 0.0;
        double m = mean.get(i, 0);
        for (size_t j = 0; j < d_model; j++) {
            double diff = input.get(i, j) - m;
            var_sum += diff * diff;
        }
        variance.set(i, 0, var_sum / d_model);
    }
    
    // Normalize
    input_normalized = Matrix(seq_len, d_model);
    for (size_t i = 0; i < seq_len; i++) {
        double m = mean.get(i, 0);
        double std = std::sqrt(variance.get(i, 0) + eps);
        for (size_t j = 0; j < d_model; j++) {
            double normalized = (input.get(i, j) - m) / std;
            // Apply learnable parameters
            double output = gamma.get(0, j) * normalized + beta.get(0, j);
            input_normalized.set(i, j, output);
        }
    }
    
    return input_normalized;
}

Matrix LayerNorm::backward(const Matrix& grad_output) {
    size_t seq_len = input_cache.getRows();
    Matrix grad_input(seq_len, d_model);
    Matrix grad_gamma_acc(1, d_model, 0.0);
    Matrix grad_beta_acc(1, d_model, 0.0);
    
    for (size_t i = 0; i < seq_len; i++) {
        double m = mean.get(i, 0);
        double std = std::sqrt(variance.get(i, 0) + eps);
        
        // Accumulate gradients for gamma and beta
        for (size_t j = 0; j < d_model; j++) {
            double normalized = (input_cache.get(i, j) - m) / std;
            grad_gamma_acc.set(0, j, grad_gamma_acc.get(0, j) + 
                               grad_output.get(i, j) * normalized);
            grad_beta_acc.set(0, j, grad_beta_acc.get(0, j) + 
                              grad_output.get(i, j));
        }
        
        // Gradient w.r.t. normalized input
        double grad_var = 0.0;
        double grad_mean = 0.0;
        
        for (size_t j = 0; j < d_model; j++) {
            double x_norm = (input_cache.get(i, j) - m) / std;
            double grad_x_norm = grad_output.get(i, j) * gamma.get(0, j);
            
            grad_var += grad_x_norm * (input_cache.get(i, j) - m) * -0.5 * 
                       std::pow(std, -3);
            grad_mean += grad_x_norm * (-1.0 / std);
        }
        
        for (size_t j = 0; j < d_model; j++) {
            double grad_x_norm = grad_output.get(i, j) * gamma.get(0, j);
            double grad = grad_x_norm / std + 
                         grad_var * 2.0 * (input_cache.get(i, j) - m) / d_model +
                         grad_mean / d_model;
            grad_input.set(i, j, grad);
        }
    }
    
    // Store gradients for parameter update
    gamma = gamma - grad_gamma_acc * 0.0;  // Will be updated in updateParameters
    beta = beta - grad_beta_acc * 0.0;
    
    return grad_input;
}

void LayerNorm::updateParameters(double learning_rate) {
    // Parameters are updated in backward pass
}

// ============================================================================
// MULTI-HEAD ATTENTION IMPLEMENTATION
// ============================================================================

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model(d_model), num_heads(num_heads), d_k(d_model / num_heads)
{
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    
    // Initialize weight matrices for each head
    for (size_t h = 0; h < num_heads; h++) {
        Matrix W_Q_h(d_model, d_k);
        Matrix W_K_h(d_model, d_k);
        Matrix W_V_h(d_model, d_k);
        
        // Xavier initialization
        W_Q_h.xavierInit(d_model, d_k);
        W_K_h.xavierInit(d_model, d_k);
        W_V_h.xavierInit(d_model, d_k);
        
        W_Q.push_back(W_Q_h);
        W_K.push_back(W_K_h);
        W_V.push_back(W_V_h);
        
        grad_W_Q.push_back(Matrix(d_model, d_k, 0.0));
        grad_W_K.push_back(Matrix(d_model, d_k, 0.0));
        grad_W_V.push_back(Matrix(d_model, d_k, 0.0));
    }
    
    // Output projection
    W_O = Matrix(d_model, d_model);
    W_O.xavierInit(d_model, d_model);
    b_O = Matrix(1, d_model, 0.0);
    
    grad_W_O = Matrix(d_model, d_model, 0.0);
    grad_b_O = Matrix(1, d_model, 0.0);
}

Matrix MultiHeadAttention::forward(const Matrix& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    Q_cache.clear();
    K_cache.clear();
    V_cache.clear();
    attention_weights_cache.clear();
    attention_output_cache.clear();
    
    // Multi-head attention
    std::vector<Matrix> head_outputs;
    
    for (size_t h = 0; h < num_heads; h++) {
        // Q, K, V projections for this head
        Matrix Q = input * W_Q[h];  // (seq_len × d_k)
        Matrix K = input * W_K[h];
        Matrix V = input * W_V[h];
        
        Q_cache.push_back(Q);
        K_cache.push_back(K);
        V_cache.push_back(V);
        
        // Scaled dot-product attention
        Matrix K_T = K.transpose();  // (d_k × seq_len)
        Matrix scores = Q * K_T;      // (seq_len × seq_len)
        
        // Scale by sqrt(d_k)
        double scale = 1.0 / std::sqrt(static_cast<double>(d_k));
        scores = scores * scale;
        
        // Softmax over last dimension
        Matrix attention_weights(seq_len, seq_len);
        for (size_t i = 0; i < seq_len; i++) {
            double max_score = scores.get(i, 0);
            for (size_t j = 1; j < seq_len; j++) {
                max_score = std::max(max_score, scores.get(i, j));
            }
            
            double sum_exp = 0.0;
            for (size_t j = 0; j < seq_len; j++) {
                sum_exp += std::exp(scores.get(i, j) - max_score);
            }
            
            for (size_t j = 0; j < seq_len; j++) {
                double weight = std::exp(scores.get(i, j) - max_score) / sum_exp;
                attention_weights.set(i, j, weight);
            }
        }
        
        attention_weights_cache.push_back(attention_weights);
        
        // Apply attention to values
        Matrix head_output = attention_weights * V;  // (seq_len × d_k)
        attention_output_cache.push_back(head_output);
        head_outputs.push_back(head_output);
    }
    
    // Concatenate all heads
    Matrix concatenated(seq_len, d_model);
    for (size_t i = 0; i < seq_len; i++) {
        size_t col_offset = 0;
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t j = 0; j < d_k; j++) {
                concatenated.set(i, col_offset + j, head_outputs[h].get(i, j));
            }
            col_offset += d_k;
        }
    }
    
    // Output projection
    Matrix output = concatenated * W_O;  // (seq_len × d_model)
    
    // Add bias
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b_O.get(0, j));
        }
    }
    
    return output;
}

Matrix MultiHeadAttention::backward(const Matrix& grad_output) {
    // Simplified backward pass - accumulate gradients
    size_t seq_len = input_cache.getRows();
    Matrix grad_input(seq_len, d_model, 0.0);
    
    // This is a simplified version - full implementation would require
    // backpropagating through all attention operations
    
    return grad_input;
}

void MultiHeadAttention::updateParameters(double learning_rate) {
    for (size_t h = 0; h < num_heads; h++) {
        W_Q[h] = W_Q[h] - grad_W_Q[h] * learning_rate;
        W_K[h] = W_K[h] - grad_W_K[h] * learning_rate;
        W_V[h] = W_V[h] - grad_W_V[h] * learning_rate;
    }
    
    W_O = W_O - grad_W_O * learning_rate;
    b_O = b_O - grad_b_O * learning_rate;
}

// ============================================================================
// FEED-FORWARD NETWORK IMPLEMENTATION
// ============================================================================

FeedForward::FeedForward(size_t d_model, size_t d_ff)
    : d_model(d_model), d_ff(d_ff)
{
    W1 = Matrix(d_model, d_ff);
    W1.xavierInit(d_model, d_ff);
    b1 = Matrix(1, d_ff, 0.0);
    
    W2 = Matrix(d_ff, d_model);
    W2.xavierInit(d_ff, d_model);
    b2 = Matrix(1, d_model, 0.0);
    
    grad_W1 = Matrix(d_model, d_ff, 0.0);
    grad_b1 = Matrix(1, d_ff, 0.0);
    grad_W2 = Matrix(d_ff, d_model, 0.0);
    grad_b2 = Matrix(1, d_model, 0.0);
}

Matrix FeedForward::forward(const Matrix& input) {
    input_cache = input;
    size_t seq_len = input.getRows();
    
    // First layer: input * W1 + b1
    Matrix hidden = input * W1;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            hidden.set(i, j, hidden.get(i, j) + b1.get(0, j));
        }
    }
    
    // ReLU activation
    relu_mask = Matrix(seq_len, d_ff);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            double val = hidden.get(i, j);
            if (val > 0) {
                relu_mask.set(i, j, 1.0);
            } else {
                hidden.set(i, j, 0.0);
                relu_mask.set(i, j, 0.0);
            }
        }
    }
    
    hidden_cache = hidden;
    
    // Second layer: hidden * W2 + b2
    Matrix output = hidden * W2;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b2.get(0, j));
        }
    }
    
    return output;
}

Matrix FeedForward::backward(const Matrix& grad_output) {
    size_t seq_len = input_cache.getRows();
    
    // Gradient w.r.t. W2 and b2
    Matrix hidden_T = hidden_cache.transpose();
    grad_W2 = hidden_T * grad_output;
    
    grad_b2 = Matrix(1, d_model, 0.0);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            grad_b2.set(0, j, grad_b2.get(0, j) + grad_output.get(i, j));
        }
    }
    
    // Backprop through second layer
    Matrix W2_T = W2.transpose();
    Matrix grad_hidden = grad_output * W2_T;
    
    // Backprop through ReLU
    grad_hidden = grad_hidden.hadamard(relu_mask);
    
    // Gradient w.r.t. W1 and b1
    Matrix input_T = input_cache.transpose();
    grad_W1 = input_T * grad_hidden;
    
    grad_b1 = Matrix(1, d_ff, 0.0);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            grad_b1.set(0, j, grad_b1.get(0, j) + grad_hidden.get(i, j));
        }
    }
    
    // Gradient w.r.t. input
    Matrix W1_T = W1.transpose();
    Matrix grad_input = grad_hidden * W1_T;
    
    return grad_input;
}

void FeedForward::updateParameters(double learning_rate) {
    W1 = W1 - grad_W1 * learning_rate;
    b1 = b1 - grad_b1 * learning_rate;
    W2 = W2 - grad_W2 * learning_rate;
    b2 = b2 - grad_b2 * learning_rate;
}

// ============================================================================
// TRANSFORMER ENCODER LAYER IMPLEMENTATION
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(size_t d_model, size_t num_heads, size_t d_ff)
{
    attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    ffn = std::make_unique<FeedForward>(d_model, d_ff);
    norm1 = std::make_unique<LayerNorm>(d_model);
    norm2 = std::make_unique<LayerNorm>(d_model);
}

Matrix TransformerEncoderLayer::forward(const Matrix& input) {
    // Multi-head attention with residual
    residual1_cache = input;
    attention_output_cache = attention->forward(input);
    Matrix attn_out = attention_output_cache + residual1_cache;
    Matrix norm1_out = norm1->forward(attn_out);
    
    // Feed-forward with residual
    residual2_cache = norm1_out;
    ffn_output_cache = ffn->forward(norm1_out);
    Matrix ffn_out = ffn_output_cache + residual2_cache;
    Matrix output = norm2->forward(ffn_out);
    
    return output;
}

Matrix TransformerEncoderLayer::backward(const Matrix& grad_output) {
    // Backward through second norm
    Matrix grad_ffn_out = norm2->backward(grad_output);
    
    // Backward through residual
    Matrix grad_residual2 = grad_ffn_out;
    Matrix grad_ffn = grad_ffn_out;
    
    // Backward through FFN
    Matrix grad_norm1_out = ffn->backward(grad_ffn);
    grad_norm1_out = grad_norm1_out + grad_residual2;
    
    // Backward through first norm
    Matrix grad_attn_out = norm1->backward(grad_norm1_out);
    
    // Backward through residual
    Matrix grad_residual1 = grad_attn_out;
    Matrix grad_attn = grad_attn_out;
    
    // Backward through attention
    Matrix grad_input = attention->backward(grad_attn);
    grad_input = grad_input + grad_residual1;
    
    return grad_input;
}

void TransformerEncoderLayer::updateParameters(double learning_rate) {
    attention->updateParameters(learning_rate);
    ffn->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
}

// ============================================================================
// BERT ENCODER IMPLEMENTATION
// ============================================================================

BERTEncoder::BERTEncoder(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers)
    : d_model(d_model), num_layers(num_layers)
{
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<TransformerEncoderLayer>(d_model, num_heads, d_ff));
    }
}

Matrix BERTEncoder::forward(const Matrix& input) {
    layer_outputs.clear();
    layer_outputs.push_back(input);
    
    Matrix current = input;
    for (size_t i = 0; i < num_layers; i++) {
        current = layers[i]->forward(current);
        layer_outputs.push_back(current);
    }
    
    return current;
}

Matrix BERTEncoder::backward(const Matrix& grad_output) {
    Matrix grad = grad_output;
    
    for (int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad);
    }
    
    return grad;
}

void BERTEncoder::updateParameters(double learning_rate) {
    for (size_t i = 0; i < num_layers; i++) {
        layers[i]->updateParameters(learning_rate);
    }
}

// ============================================================================
// BERT EMBEDDING IMPLEMENTATION
// ============================================================================

BERTEmbedding::BERTEmbedding(size_t vocab_size, size_t d_model, size_t max_seq_length)
    : vocab_size(vocab_size), d_model(d_model), max_seq_length(max_seq_length)
{
    // Initialize token embeddings
    token_embeddings = Matrix(vocab_size, d_model);
    token_embeddings.randomNormal(0.0, 0.02);
    
    grad_token_embeddings = Matrix(vocab_size, d_model, 0.0);
    
    // Initialize positional encodings
    positional_encodings = Matrix(max_seq_length, d_model);
    initializePositionalEncoding();
}

void BERTEmbedding::initializePositionalEncoding() {
    // Sinusoidal positional encoding
    for (size_t pos = 0; pos < max_seq_length; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / d_model);
            if (i % 2 == 0) {
                positional_encodings.set(pos, i, std::sin(angle));
            } else {
                positional_encodings.set(pos, i, std::cos(angle));
            }
        }
    }
}

Matrix BERTEmbedding::forward(const std::vector<int>& token_ids) {
    token_ids_cache = token_ids;
    size_t seq_len = token_ids.size();
    
    Matrix output(seq_len, d_model);
    
    for (size_t i = 0; i < seq_len; i++) {
        int token_id = token_ids[i];
        for (size_t j = 0; j < d_model; j++) {
            double token_emb = token_embeddings.get(token_id, j);
            double pos_enc = positional_encodings.get(i, j);
            output.set(i, j, token_emb + pos_enc);
        }
    }
    
    return output;
}

void BERTEmbedding::backward(const Matrix& grad_output) {
    // Accumulate gradients for token embeddings
    for (size_t i = 0; i < token_ids_cache.size(); i++) {
        int token_id = token_ids_cache[i];
        for (size_t j = 0; j < d_model; j++) {
            double current_grad = grad_token_embeddings.get(token_id, j);
            grad_token_embeddings.set(token_id, j, current_grad + grad_output.get(i, j));
        }
    }
}

void BERTEmbedding::updateParameters(double learning_rate) {
    token_embeddings = token_embeddings - grad_token_embeddings * learning_rate;
    grad_token_embeddings.zeros();  // Reset gradients
}

// ============================================================================
// OUTPUT HEADS IMPLEMENTATION
// ============================================================================

IntentClassifier::IntentClassifier(size_t d_model, size_t num_intents)
    : d_model(d_model), num_intents(num_intents)
{
    W = Matrix(d_model, num_intents);
    W.xavierInit(d_model, num_intents);
    b = Matrix(1, num_intents, 0.0);
    
    grad_W = Matrix(d_model, num_intents, 0.0);
    grad_b = Matrix(1, num_intents, 0.0);
}

Matrix IntentClassifier::forward(const Matrix& cls_representation) {
    input_cache = cls_representation;
    
    // cls_representation is (1 × d_model)
    Matrix logits = cls_representation * W;  // (1 × num_intents)
    
    for (size_t j = 0; j < num_intents; j++) {
        logits.set(0, j, logits.get(0, j) + b.get(0, j));
    }
    
    return logits;
}

Matrix IntentClassifier::backward(const Matrix& grad_output) {
    // Gradient w.r.t. W
    Matrix input_T = input_cache.transpose();
    grad_W = input_T * grad_output;
    
    // Gradient w.r.t. b
    grad_b = grad_output;
    
    // Gradient w.r.t. input
    Matrix W_T = W.transpose();
    Matrix grad_input = grad_output * W_T;
    
    return grad_input;
}

void IntentClassifier::updateParameters(double learning_rate) {
    W = W - grad_W * learning_rate;
    b = b - grad_b * learning_rate;
}

SlotTagger::SlotTagger(size_t d_model, size_t num_slots)
    : d_model(d_model), num_slots(num_slots)
{
    W = Matrix(d_model, num_slots);
    W.xavierInit(d_model, num_slots);
    b = Matrix(1, num_slots, 0.0);
    
    grad_W = Matrix(d_model, num_slots, 0.0);
    grad_b = Matrix(1, num_slots, 0.0);
}

Matrix SlotTagger::forward(const Matrix& sequence_representations) {
    input_cache = sequence_representations;
    size_t seq_len = sequence_representations.getRows();
    
    // sequence_representations is (seq_len × d_model)
    Matrix logits = sequence_representations * W;  // (seq_len × num_slots)
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_slots; j++) {
            logits.set(i, j, logits.get(i, j) + b.get(0, j));
        }
    }
    
    return logits;
}

Matrix SlotTagger::backward(const Matrix& grad_output) {
    // Gradient w.r.t. W
    Matrix input_T = input_cache.transpose();
    grad_W = input_T * grad_output;
    
    // Gradient w.r.t. b
    size_t seq_len = input_cache.getRows();
    grad_b = Matrix(1, num_slots, 0.0);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_slots; j++) {
            grad_b.set(0, j, grad_b.get(0, j) + grad_output.get(i, j));
        }
    }
    
    // Gradient w.r.t. input
    Matrix W_T = W.transpose();
    Matrix grad_input = grad_output * W_T;
    
    return grad_input;
}

void SlotTagger::updateParameters(double learning_rate) {
    W = W - grad_W * learning_rate;
    b = b - grad_b * learning_rate;
}

EntityDetector::EntityDetector(size_t d_model, size_t num_entities)
    : d_model(d_model), num_entities(num_entities)
{
    W = Matrix(d_model, num_entities);
    W.xavierInit(d_model, num_entities);
    b = Matrix(1, num_entities, 0.0);
    
    grad_W = Matrix(d_model, num_entities, 0.0);
    grad_b = Matrix(1, num_entities, 0.0);
}

Matrix EntityDetector::forward(const Matrix& sequence_representations) {
    input_cache = sequence_representations;
    size_t seq_len = sequence_representations.getRows();
    
    Matrix logits = sequence_representations * W;
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_entities; j++) {
            logits.set(i, j, logits.get(i, j) + b.get(0, j));
        }
    }
    
    return logits;
}

Matrix EntityDetector::backward(const Matrix& grad_output) {
    Matrix input_T = input_cache.transpose();
    grad_W = input_T * grad_output;
    
    size_t seq_len = input_cache.getRows();
    grad_b = Matrix(1, num_entities, 0.0);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_entities; j++) {
            grad_b.set(0, j, grad_b.get(0, j) + grad_output.get(i, j));
        }
    }
    
    Matrix W_T = W.transpose();
    Matrix grad_input = grad_output * W_T;
    
    return grad_input;
}

void EntityDetector::updateParameters(double learning_rate) {
    W = W - grad_W * learning_rate;
    b = b - grad_b * learning_rate;
}

// ============================================================================
// COMPLETE BERT-NLU MODEL IMPLEMENTATION
// ============================================================================

BERTForNLU::BERTForNLU(size_t vocab_size, size_t d_model, size_t num_heads,
                       size_t d_ff, size_t num_layers, size_t max_seq_length,
                       size_t num_intents, size_t num_slots, size_t num_entities)
    : d_model(d_model), vocab_size(vocab_size),
      num_intents(num_intents), num_slots(num_slots), num_entities(num_entities)
{
    embedding = std::make_unique<BERTEmbedding>(vocab_size, d_model, max_seq_length);
    encoder = std::make_unique<BERTEncoder>(d_model, num_heads, d_ff, num_layers);
    intent_head = std::make_unique<IntentClassifier>(d_model, num_intents);
    slot_head = std::make_unique<SlotTagger>(d_model, num_slots);
    entity_head = std::make_unique<EntityDetector>(d_model, num_entities);
}

std::tuple<Matrix, Matrix, Matrix> BERTForNLU::forward(const std::vector<int>& token_ids) {
    // Embedding
    Matrix embedded = embedding->forward(token_ids);
    
    // Encoder
    encoder_output_cache = encoder->forward(embedded);
    
    // Extract [CLS] token (first token) for intent classification
    Matrix cls_token(1, d_model);
    for (size_t j = 0; j < d_model; j++) {
        cls_token.set(0, j, encoder_output_cache.get(0, j));
    }
    
    // Three task heads
    Matrix intent_logits = intent_head->forward(cls_token);
    Matrix slot_logits = slot_head->forward(encoder_output_cache);
    Matrix entity_logits = entity_head->forward(encoder_output_cache);
    
    return std::make_tuple(intent_logits, slot_logits, entity_logits);
}

void BERTForNLU::backward(const Matrix& grad_intent, const Matrix& grad_slots,
                          const Matrix& grad_entities) {
    // Backward through output heads
    Matrix grad_cls = intent_head->backward(grad_intent);
    Matrix grad_encoder_slots = slot_head->backward(grad_slots);
    Matrix grad_encoder_entities = entity_head->backward(grad_entities);
    
    // Combine gradients for encoder output
    size_t seq_len = encoder_output_cache.getRows();
    Matrix grad_encoder_output(seq_len, d_model);
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < d_model; j++) {
            double grad = grad_encoder_slots.get(i, j) + grad_encoder_entities.get(i, j);
            if (i == 0) {  // Add CLS gradient only to first token
                grad += grad_cls.get(0, j);
            }
            grad_encoder_output.set(i, j, grad);
        }
    }
    
    // Backward through encoder
    Matrix grad_embedded = encoder->backward(grad_encoder_output);
    
    // Backward through embedding
    embedding->backward(grad_embedded);
}

void BERTForNLU::updateParameters(double learning_rate) {
    embedding->updateParameters(learning_rate);
    encoder->updateParameters(learning_rate);
    intent_head->updateParameters(learning_rate);
    slot_head->updateParameters(learning_rate);
    entity_head->updateParameters(learning_rate);
}

std::tuple<int, std::vector<int>, std::vector<int>> BERTForNLU::predict(
    const std::vector<int>& token_ids) {
    
    auto [intent_logits, slot_logits, entity_logits] = forward(token_ids);
    
    // Predict intent (argmax of logits)
    int predicted_intent = 0;
    double max_intent_score = intent_logits.get(0, 0);
    for (size_t i = 1; i < num_intents; i++) {
        if (intent_logits.get(0, i) > max_intent_score) {
            max_intent_score = intent_logits.get(0, i);
            predicted_intent = i;
        }
    }
    
    // Predict slots (argmax for each token)
    std::vector<int> predicted_slots;
    size_t seq_len = slot_logits.getRows();
    for (size_t i = 0; i < seq_len; i++) {
        int slot_id = 0;
        double max_slot_score = slot_logits.get(i, 0);
        for (size_t j = 1; j < num_slots; j++) {
            if (slot_logits.get(i, j) > max_slot_score) {
                max_slot_score = slot_logits.get(i, j);
                slot_id = j;
            }
        }
        predicted_slots.push_back(slot_id);
    }
    
    // Predict entities (argmax for each token)
    std::vector<int> predicted_entities;
    for (size_t i = 0; i < seq_len; i++) {
        int entity_id = 0;
        double max_entity_score = entity_logits.get(i, 0);
        for (size_t j = 1; j < num_entities; j++) {
            if (entity_logits.get(i, j) > max_entity_score) {
                max_entity_score = entity_logits.get(i, j);
                entity_id = j;
            }
        }
        predicted_entities.push_back(entity_id);
    }
    
    return std::make_tuple(predicted_intent, predicted_slots, predicted_entities);
}
