#include "../../include/nn/transformer/transformer.h"
#include "../../include/nn/transformer/attention.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>

// ============ Transformer (Encoder-Decoder) ============

Transformer::Transformer(size_t src_vocab_size, size_t tgt_vocab_size,
                        size_t d_model, size_t num_heads,
                        size_t num_encoder_layers, size_t num_decoder_layers,
                        size_t d_ff, size_t max_seq_len, double dropout,
                        int pad_token_id, int bos_token_id, int eos_token_id)
    : src_vocab_size(src_vocab_size), tgt_vocab_size(tgt_vocab_size),
      d_model(d_model), num_heads(num_heads),
      num_encoder_layers(num_encoder_layers), num_decoder_layers(num_decoder_layers),
      d_ff(d_ff), max_seq_len(max_seq_len), dropout(dropout),
      pad_token_id(pad_token_id), bos_token_id(bos_token_id), eos_token_id(eos_token_id) {
    
    // Initialize embeddings
    src_embedding = std::make_unique<TransformerEmbedding>(src_vocab_size, d_model, max_seq_len, dropout);
    tgt_embedding = std::make_unique<TransformerEmbedding>(tgt_vocab_size, d_model, max_seq_len, dropout);
    
    // Initialize encoder and decoder
    encoder = std::make_unique<TransformerEncoder>(num_encoder_layers, d_model, num_heads, d_ff, dropout);
    decoder = std::make_unique<TransformerDecoder>(num_decoder_layers, d_model, num_heads, d_ff, dropout);
    
    // Initialize output projection
    initializeOutputLayer();
}

void Transformer::initializeOutputLayer() {
    output_projection = Matrix(d_model, tgt_vocab_size);
    output_bias = Matrix(1, tgt_vocab_size, 0.0);
    
    // Xavier initialization
    double std = std::sqrt(2.0 / (d_model + tgt_vocab_size));
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < tgt_vocab_size; j++) {
            output_projection.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * std);
        }
    }
    
    dW_out = Matrix(d_model, tgt_vocab_size, 0.0);
    db_out = Matrix(1, tgt_vocab_size, 0.0);
}

Matrix Transformer::forward(const std::vector<std::vector<int>>& src_tokens,
                           const std::vector<std::vector<int>>& tgt_tokens,
                           bool training) {
    size_t batch_size = src_tokens.size();
    size_t src_seq_len = src_tokens[0].size();
    size_t tgt_seq_len = tgt_tokens[0].size();
    
    // Cache for backward
    cached_src_tokens = Matrix(batch_size, src_seq_len);
    cached_tgt_tokens = Matrix(batch_size, tgt_seq_len);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < src_seq_len; t++) {
            cached_src_tokens.set(b, t, src_tokens[b][t]);
        }
        for (size_t t = 0; t < tgt_seq_len; t++) {
            cached_tgt_tokens.set(b, t, tgt_tokens[b][t]);
        }
    }
    
    // Create masks
    Matrix src_mask = createPaddingMask(src_tokens, pad_token_id);
    Matrix tgt_mask = createCausalMask(tgt_seq_len);
    
    // Combine causal mask with padding mask for target
    Matrix tgt_padding_mask = createPaddingMask(tgt_tokens, pad_token_id);
    for (size_t i = 0; i < tgt_mask.getRows(); i++) {
        for (size_t j = 0; j < tgt_mask.getCols(); j++) {
            if (tgt_mask.get(i, j) == 0.0 || tgt_padding_mask.get(0, j) == 0.0) {
                tgt_mask.set(i, j, 0.0);
            }
        }
    }
    
    // Encode source
    Matrix src_embedded = src_embedding->forward(src_tokens, training);
    cached_encoder_output = encoder->forward(src_embedded, &src_mask, training);
    
    // Decode target
    Matrix tgt_embedded = tgt_embedding->forward(tgt_tokens, training);
    cached_decoder_output = decoder->forward(tgt_embedded, cached_encoder_output,
                                            &tgt_mask, &src_mask, training);
    
    // Project to vocabulary
    Matrix logits = cached_decoder_output * output_projection;
    
    // Add bias
    for (size_t i = 0; i < logits.getRows(); i++) {
        for (size_t j = 0; j < logits.getCols(); j++) {
            logits.set(i, j, logits.get(i, j) + output_bias.get(0, j));
        }
    }
    
    return logits;
}

double Transformer::computeLoss(const Matrix& logits,
                                const std::vector<std::vector<int>>& targets) {
    size_t batch_size = targets.size();
    size_t seq_len = targets[0].size();
    
    double total_loss = 0.0;
    size_t valid_tokens = 0;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int target_id = targets[b][t];
            
            // Skip padding tokens
            if (target_id == pad_token_id) continue;
            
            size_t idx = b * seq_len + t;
            
            // Softmax for this position
            double max_logit = logits.get(idx, 0);
            for (size_t v = 1; v < tgt_vocab_size; v++) {
                max_logit = std::max(max_logit, logits.get(idx, v));
            }
            
            double sum_exp = 0.0;
            for (size_t v = 0; v < tgt_vocab_size; v++) {
                sum_exp += std::exp(logits.get(idx, v) - max_logit);
            }
            
            double log_prob = logits.get(idx, target_id) - max_logit - std::log(sum_exp);
            total_loss -= log_prob;
            valid_tokens++;
        }
    }
    
    return valid_tokens > 0 ? total_loss / valid_tokens : 0.0;
}

void Transformer::backward(const Matrix& logits,
                          const std::vector<std::vector<int>>& targets) {
    size_t batch_size = targets.size();
    size_t seq_len = targets[0].size();
    
    // Compute softmax and gradient
    Matrix grad_logits(logits.getRows(), logits.getCols(), 0.0);
    size_t valid_tokens = 0;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int target_id = targets[b][t];
            if (target_id == pad_token_id) continue;
            
            size_t idx = b * seq_len + t;
            
            // Compute softmax
            double max_logit = logits.get(idx, 0);
            for (size_t v = 1; v < tgt_vocab_size; v++) {
                max_logit = std::max(max_logit, logits.get(idx, v));
            }
            
            double sum_exp = 0.0;
            std::vector<double> probs(tgt_vocab_size);
            for (size_t v = 0; v < tgt_vocab_size; v++) {
                probs[v] = std::exp(logits.get(idx, v) - max_logit);
                sum_exp += probs[v];
            }
            
            // Gradient: softmax - one_hot
            for (size_t v = 0; v < tgt_vocab_size; v++) {
                grad_logits.set(idx, v, probs[v] / sum_exp);
                if ((int)v == target_id) {
                    grad_logits.set(idx, v, grad_logits.get(idx, v) - 1.0);
                }
            }
            
            valid_tokens++;
        }
    }
    
    // Normalize gradient
    if (valid_tokens > 0) {
        for (size_t i = 0; i < grad_logits.getRows(); i++) {
            for (size_t j = 0; j < grad_logits.getCols(); j++) {
                grad_logits.set(i, j, grad_logits.get(i, j) / valid_tokens);
            }
        }
    }
    
    // Gradient for output projection
    dW_out = dW_out + (cached_decoder_output.transpose() * grad_logits);
    for (size_t j = 0; j < tgt_vocab_size; j++) {
        for (size_t i = 0; i < grad_logits.getRows(); i++) {
            db_out.set(0, j, db_out.get(0, j) + grad_logits.get(i, j));
        }
    }
    
    // Backward through output projection
    Matrix grad_decoder_out = grad_logits * output_projection.transpose();
    
    // Backward through decoder
    Matrix grad_enc_from_dec(cached_encoder_output.getRows(), cached_encoder_output.getCols());
    Matrix grad_tgt_emb = decoder->backward(grad_decoder_out, grad_enc_from_dec);
    
    // Convert cached tokens back to vectors for embedding backward
    size_t tgt_seq_len = cached_tgt_tokens.getCols();
    std::vector<std::vector<int>> tgt_tokens_vec(batch_size, std::vector<int>(tgt_seq_len));
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < tgt_seq_len; t++) {
            tgt_tokens_vec[b][t] = (int)cached_tgt_tokens.get(b, t);
        }
    }
    
    // Backward through target embedding
    tgt_embedding->backward(grad_tgt_emb, tgt_tokens_vec);
    
    // Backward through encoder
    Matrix grad_src_emb = encoder->backward(grad_enc_from_dec);
    
    // Convert source tokens
    size_t src_seq_len = cached_src_tokens.getCols();
    std::vector<std::vector<int>> src_tokens_vec(batch_size, std::vector<int>(src_seq_len));
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < src_seq_len; t++) {
            src_tokens_vec[b][t] = (int)cached_src_tokens.get(b, t);
        }
    }
    
    // Backward through source embedding
    src_embedding->backward(grad_src_emb, src_tokens_vec);
}

void Transformer::updateParameters(double learning_rate) {
    // Update embeddings
    src_embedding->updateParameters(learning_rate);
    tgt_embedding->updateParameters(learning_rate);
    
    // Update encoder and decoder
    encoder->updateParameters(learning_rate);
    decoder->updateParameters(learning_rate);
    
    // Update output projection
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < tgt_vocab_size; j++) {
            output_projection.set(i, j, output_projection.get(i, j) - learning_rate * dW_out.get(i, j));
            dW_out.set(i, j, 0.0);
        }
    }
    
    for (size_t j = 0; j < tgt_vocab_size; j++) {
        output_bias.set(0, j, output_bias.get(0, j) - learning_rate * db_out.get(0, j));
        db_out.set(0, j, 0.0);
    }
}

double Transformer::trainStep(const std::vector<std::vector<int>>& src_tokens,
                             const std::vector<std::vector<int>>& tgt_tokens,
                             double learning_rate) {
    // Forward pass
    Matrix logits = forward(src_tokens, tgt_tokens, true);
    
    // Compute loss
    double loss = computeLoss(logits, tgt_tokens);
    
    // Backward pass
    backward(logits, tgt_tokens);
    
    // Update parameters
    updateParameters(learning_rate);
    
    return loss;
}

std::vector<int> Transformer::greedyDecode(const std::vector<int>& src_tokens,
                                          size_t max_length) {
    std::vector<std::vector<int>> src_batch = {src_tokens};
    
    // Encode source
    Matrix src_mask = createPaddingMask(src_batch, pad_token_id);
    Matrix src_embedded = src_embedding->forward(src_batch, false);
    Matrix encoder_output = encoder->forward(src_embedded, &src_mask, false);
    
    // Start with BOS token
    std::vector<int> generated = {bos_token_id};
    
    for (size_t i = 0; i < max_length; i++) {
        std::vector<std::vector<int>> tgt_batch = {generated};
        
        // Decode
        Matrix tgt_mask = createCausalMask(generated.size());
        Matrix tgt_embedded = tgt_embedding->forward(tgt_batch, false);
        Matrix decoder_output = decoder->forward(tgt_embedded, encoder_output,
                                                &tgt_mask, &src_mask, false);
        
        // Get last position logits
        Matrix logits = decoder_output * output_projection;
        size_t last_pos = generated.size() - 1;
        
        // Find argmax
        int next_token = 0;
        double max_logit = logits.get(last_pos, 0) + output_bias.get(0, 0);
        for (size_t v = 1; v < tgt_vocab_size; v++) {
            double logit = logits.get(last_pos, v) + output_bias.get(0, v);
            if (logit > max_logit) {
                max_logit = logit;
                next_token = v;
            }
        }
        
        generated.push_back(next_token);
        
        // Stop if EOS generated
        if (next_token == eos_token_id) break;
    }
    
    return generated;
}

std::vector<int> Transformer::beamSearch(const std::vector<int>& src_tokens,
                                        size_t beam_size, size_t max_length) {
    // Simplified beam search - just return greedy for now
    // Full implementation would maintain beam_size hypotheses
    return greedyDecode(src_tokens, max_length);
}

void Transformer::summary() const {
    std::cout << "=============== Transformer Summary ===============\n";
    std::cout << "Source Vocabulary Size: " << src_vocab_size << "\n";
    std::cout << "Target Vocabulary Size: " << tgt_vocab_size << "\n";
    std::cout << "Model Dimension (d_model): " << d_model << "\n";
    std::cout << "Number of Heads: " << num_heads << "\n";
    std::cout << "Encoder Layers: " << num_encoder_layers << "\n";
    std::cout << "Decoder Layers: " << num_decoder_layers << "\n";
    std::cout << "Feed-Forward Dimension: " << d_ff << "\n";
    std::cout << "Max Sequence Length: " << max_seq_len << "\n";
    std::cout << "Dropout Rate: " << dropout << "\n";
    std::cout << "Total Parameters: " << getParameterCount() << "\n";
    std::cout << "==================================================\n";
}

size_t Transformer::getParameterCount() const {
    size_t total = 0;
    
    // Embeddings
    total += src_vocab_size * d_model;  // Source token embedding
    total += tgt_vocab_size * d_model;  // Target token embedding
    
    // Encoder: num_layers × (attention + ffn + layer_norm)
    size_t encoder_layer_params = 4 * d_model * d_model +  // Q, K, V, O projections
                                  2 * d_model * d_ff +      // FFN
                                  4 * d_model;              // 2 layer norms (gamma, beta)
    total += num_encoder_layers * encoder_layer_params;
    
    // Decoder: num_layers × (self_attn + cross_attn + ffn + layer_norm)
    size_t decoder_layer_params = 8 * d_model * d_model +  // 2 attention modules
                                  2 * d_model * d_ff +      // FFN
                                  6 * d_model;              // 3 layer norms
    total += num_decoder_layers * decoder_layer_params;
    
    // Output projection
    total += d_model * tgt_vocab_size + tgt_vocab_size;
    
    return total;
}

bool Transformer::saveModel(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Save hyperparameters and matrices
    // (Implementation details omitted for brevity)
    
    file.close();
    return true;
}

bool Transformer::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Load hyperparameters and matrices
    // (Implementation details omitted for brevity)
    
    file.close();
    return true;
}

// ============ DecoderOnlyTransformer (GPT-style) ============

DecoderOnlyTransformer::DecoderOnlyTransformer(size_t vocab_size, size_t d_model,
                                             size_t num_layers, size_t num_heads,
                                             size_t d_ff, size_t max_seq_len,
                                             double dropout)
    : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
      num_heads(num_heads), d_ff(d_ff), max_seq_len(max_seq_len), dropout(dropout) {
    
    embedding = std::make_unique<TransformerEmbedding>(vocab_size, d_model, max_seq_len, dropout);
    decoder = std::make_unique<TransformerDecoder>(num_layers, d_model, num_heads, d_ff, dropout);
    
    // Output projection (can tie with embedding)
    output_projection = Matrix(d_model, vocab_size);
    output_bias = Matrix(1, vocab_size, 0.0);
    
    double std = std::sqrt(2.0 / (d_model + vocab_size));
    for (size_t i = 0; i < d_model; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            output_projection.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * std);
        }
    }
}

Matrix DecoderOnlyTransformer::forward(const std::vector<std::vector<int>>& tokens,
                                      bool training) {
    size_t seq_len = tokens[0].size();
    
    // Create causal mask
    Matrix mask = createCausalMask(seq_len);
    
    // Embed
    Matrix embedded = embedding->forward(tokens, training);
    
    // Dummy encoder output for decoder (not used in decoder-only)
    Matrix dummy_encoder = Matrix(1, d_model, 0.0);
    
    // Decode with masked self-attention
    Matrix output = decoder->forward(embedded, dummy_encoder, &mask, nullptr, training);
    
    // Project to vocabulary
    Matrix logits = output * output_projection;
    for (size_t i = 0; i < logits.getRows(); i++) {
        for (size_t j = 0; j < logits.getCols(); j++) {
            logits.set(i, j, logits.get(i, j) + output_bias.get(0, j));
        }
    }
    
    return logits;
}

std::vector<int> DecoderOnlyTransformer::generate(const std::vector<int>& prompt,
                                                 size_t max_length, double temperature) {
    std::vector<int> generated = prompt;
    
    for (size_t i = 0; i < max_length; i++) {
        std::vector<std::vector<int>> batch = {generated};
        Matrix logits = forward(batch, false);
        
        size_t last_pos = generated.size() - 1;
        
        // Apply temperature
        std::vector<double> probs(vocab_size);
        double max_logit = logits.get(last_pos, 0);
        for (size_t v = 1; v < vocab_size; v++) {
            max_logit = std::max(max_logit, logits.get(last_pos, v));
        }
        
        double sum_exp = 0.0;
        for (size_t v = 0; v < vocab_size; v++) {
            probs[v] = std::exp((logits.get(last_pos, v) - max_logit) / temperature);
            sum_exp += probs[v];
        }
        
        // Sample or greedy
        int next_token = 0;
        if (temperature > 0.0) {
            double rand_val = ((double)rand() / RAND_MAX) * sum_exp;
            double cumsum = 0.0;
            for (size_t v = 0; v < vocab_size; v++) {
                cumsum += probs[v];
                if (cumsum >= rand_val) {
                    next_token = v;
                    break;
                }
            }
        } else {
            double max_prob = probs[0];
            for (size_t v = 1; v < vocab_size; v++) {
                if (probs[v] > max_prob) {
                    max_prob = probs[v];
                    next_token = v;
                }
            }
        }
        
        generated.push_back(next_token);
    }
    
    return generated;
}

void DecoderOnlyTransformer::summary() const {
    std::cout << "========== Decoder-Only Transformer (GPT) ==========\n";
    std::cout << "Vocabulary Size: " << vocab_size << "\n";
    std::cout << "Model Dimension: " << d_model << "\n";
    std::cout << "Number of Layers: " << num_layers << "\n";
    std::cout << "Number of Heads: " << num_heads << "\n";
    std::cout << "Feed-Forward Dimension: " << d_ff << "\n";
    std::cout << "====================================================\n";
}

// ============ EncoderOnlyTransformer (BERT-style) ============

EncoderOnlyTransformer::EncoderOnlyTransformer(size_t vocab_size, size_t d_model,
                                             size_t num_layers, size_t num_heads,
                                             size_t d_ff, size_t max_seq_len,
                                             double dropout)
    : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
      num_heads(num_heads), d_ff(d_ff), max_seq_len(max_seq_len), dropout(dropout) {
    
    embedding = std::make_unique<TransformerEmbedding>(vocab_size, d_model, max_seq_len, dropout);
    encoder = std::make_unique<TransformerEncoder>(num_layers, d_model, num_heads, d_ff, dropout);
}

Matrix EncoderOnlyTransformer::forward(const std::vector<std::vector<int>>& tokens,
                                      bool training) {
    // Create padding mask
    Matrix mask = createPaddingMask(tokens, 0);
    
    // Embed
    Matrix embedded = embedding->forward(tokens, training);
    
    // Encode
    Matrix output = encoder->forward(embedded, &mask, training);
    
    return output;
}

Matrix EncoderOnlyTransformer::getPooledOutput(const Matrix& encoder_output) {
    // Return [CLS] token representation (first token)
    Matrix pooled(1, d_model);
    
    for (size_t j = 0; j < d_model; j++) {
        pooled.set(0, j, encoder_output.get(0, j));
    }
    
    return pooled;
}

void EncoderOnlyTransformer::summary() const {
    std::cout << "========== Encoder-Only Transformer (BERT) ==========\n";
    std::cout << "Vocabulary Size: " << vocab_size << "\n";
    std::cout << "Model Dimension: " << d_model << "\n";
    std::cout << "Number of Layers: " << num_layers << "\n";
    std::cout << "Number of Heads: " << num_heads << "\n";
    std::cout << "Feed-Forward Dimension: " << d_ff << "\n";
    std::cout << "=====================================================\n";
}
