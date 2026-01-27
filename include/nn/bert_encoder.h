#ifndef BERT_ENCODER_H
#define BERT_ENCODER_H

#include "matrix.h"
#include "activation.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

/**
 * @file bert_encoder.h
 * @brief BERT Encoder from Scratch for Multi-Task NLU
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * BERT ARCHITECTURE OVERVIEW
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * BERT (Bidirectional Encoder Representations from Transformers)
 * 
 * Input: "show me flights from boston"
 *   ↓
 * Token Embeddings: [tok1, tok2, tok3, tok4, tok5]
 *   ↓
 * + Positional Encodings: [pos1, pos2, pos3, pos4, pos5]
 *   ↓
 * Encoder Layer 1: Multi-Head Self-Attention → FFN
 *   ↓
 * Encoder Layer 2: Multi-Head Self-Attention → FFN
 *   ↓
 * ... (N layers)
 *   ↓
 * Output: [h1, h2, h3, h4, h5]
 *   ↓
 * ┌─────────────┬────────────────┬──────────────┐
 * │   Intent    │  Slot Tagging  │   Entities   │
 * │ Classifier  │    (per-tok)   │  (per-tok)   │
 * └─────────────┴────────────────┴──────────────┘
 */

// ============================================================================
// LAYER NORMALIZATION
// ============================================================================

/**
 * @brief Layer Normalization: normalize across features
 * 
 * For input x of shape (batch, seq_len, d_model):
 * mean = mean(x, axis=-1)
 * var = var(x, axis=-1)
 * x_norm = (x - mean) / sqrt(var + eps)
 * output = gamma * x_norm + beta
 */
class LayerNorm {
private:
    size_t d_model;
    Matrix gamma;  // learnable scale (d_model,)
    Matrix beta;   // learnable shift (d_model,)
    double eps;
    
    // For backward pass
    Matrix input_normalized;
    Matrix mean;
    Matrix variance;
    Matrix input_cache;

public:
    LayerNorm(size_t d_model, double epsilon = 1e-6);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
    
    Matrix getGamma() const { return gamma; }
    Matrix getBeta() const { return beta; }
};

// ============================================================================
// MULTI-HEAD ATTENTION
// ============================================================================

/**
 * @brief Multi-Head Self-Attention
 * 
 * For each head h:
 *   Q = X * W_Q^h
 *   K = X * W_K^h  
 *   V = X * W_V^h
 *   
 *   Attention(Q,K,V) = softmax(Q*K^T / sqrt(d_k)) * V
 * 
 * Concatenate all heads → Linear projection
 */
class MultiHeadAttention {
private:
    size_t d_model;      // Model dimension (e.g., 256)
    size_t num_heads;    // Number of attention heads (e.g., 8)
    size_t d_k;          // Dimension per head (d_model / num_heads)
    
    // Parameters for each head
    std::vector<Matrix> W_Q;  // Query weights (num_heads × d_model × d_k)
    std::vector<Matrix> W_K;  // Key weights
    std::vector<Matrix> W_V;  // Value weights
    
    Matrix W_O;  // Output projection (d_model × d_model)
    Matrix b_O;  // Output bias
    
    // Gradients
    std::vector<Matrix> grad_W_Q;
    std::vector<Matrix> grad_W_K;
    std::vector<Matrix> grad_W_V;
    Matrix grad_W_O;
    Matrix grad_b_O;
    
    // Cache for backward pass
    Matrix input_cache;
    std::vector<Matrix> Q_cache;
    std::vector<Matrix> K_cache;
    std::vector<Matrix> V_cache;
    std::vector<Matrix> attention_weights_cache;
    std::vector<Matrix> attention_output_cache;
    
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
    
    std::vector<Matrix> getAttentionWeights() const { return attention_weights_cache; }
};

// ============================================================================
// POSITION-WISE FEED-FORWARD NETWORK
// ============================================================================

/**
 * @brief Feed-Forward Network
 * 
 * FFN(x) = ReLU(x*W1 + b1) * W2 + b2
 * 
 * Typically: d_ff = 4 * d_model
 */
class FeedForward {
private:
    size_t d_model;
    size_t d_ff;
    
    Matrix W1;  // (d_model × d_ff)
    Matrix b1;
    Matrix W2;  // (d_ff × d_model)
    Matrix b2;
    
    // Gradients
    Matrix grad_W1;
    Matrix grad_b1;
    Matrix grad_W2;
    Matrix grad_b2;
    
    // Cache
    Matrix input_cache;
    Matrix hidden_cache;
    Matrix relu_mask;  // For ReLU backward
    
public:
    FeedForward(size_t d_model, size_t d_ff);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
};

// ============================================================================
// TRANSFORMER ENCODER LAYER
// ============================================================================

/**
 * @brief Single Transformer Encoder Layer
 * 
 * x → Multi-Head Attention → Add & Norm → FFN → Add & Norm → output
 */
class TransformerEncoderLayer {
private:
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> ffn;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;
    
    // Cache for residual connections
    Matrix residual1_cache;
    Matrix residual2_cache;
    Matrix attention_output_cache;
    Matrix ffn_output_cache;
    
public:
    TransformerEncoderLayer(size_t d_model, size_t num_heads, size_t d_ff);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
};

// ============================================================================
// BERT ENCODER (STACK OF TRANSFORMER LAYERS)
// ============================================================================

/**
 * @brief BERT Encoder: Stack of N Transformer layers
 */
class BERTEncoder {
private:
    size_t d_model;
    size_t num_layers;
    std::vector<std::unique_ptr<TransformerEncoderLayer>> layers;
    
    // Cache for backward pass
    std::vector<Matrix> layer_outputs;
    
public:
    BERTEncoder(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
    
    size_t getNumLayers() const { return num_layers; }
};

// ============================================================================
// EMBEDDING LAYER
// ============================================================================

/**
 * @brief Token + Positional Embeddings
 */
class BERTEmbedding {
private:
    size_t vocab_size;
    size_t d_model;
    size_t max_seq_length;
    
    Matrix token_embeddings;      // (vocab_size × d_model)
    Matrix positional_encodings;  // (max_seq_length × d_model)
    
    // Gradients
    Matrix grad_token_embeddings;
    
    // Cache
    std::vector<int> token_ids_cache;
    
public:
    BERTEmbedding(size_t vocab_size, size_t d_model, size_t max_seq_length);
    
    Matrix forward(const std::vector<int>& token_ids);
    void backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
    
    void initializePositionalEncoding();
};

// ============================================================================
// MULTI-TASK OUTPUT HEADS
// ============================================================================

/**
 * @brief Intent Classification Head
 * 
 * Uses [CLS] token representation (first token)
 */
class IntentClassifier {
private:
    size_t d_model;
    size_t num_intents;
    
    Matrix W;  // (d_model × num_intents)
    Matrix b;  // (num_intents,)
    
    Matrix grad_W;
    Matrix grad_b;
    
    Matrix input_cache;
    
public:
    IntentClassifier(size_t d_model, size_t num_intents);
    
    Matrix forward(const Matrix& cls_representation);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
};

/**
 * @brief Slot Tagging Head
 * 
 * Per-token classification
 */
class SlotTagger {
private:
    size_t d_model;
    size_t num_slots;
    
    Matrix W;  // (d_model × num_slots)
    Matrix b;  // (num_slots,)
    
    Matrix grad_W;
    Matrix grad_b;
    
    Matrix input_cache;
    
public:
    SlotTagger(size_t d_model, size_t num_slots);
    
    Matrix forward(const Matrix& sequence_representations);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
};

/**
 * @brief Entity Detection Head
 * 
 * Per-token entity classification
 */
class EntityDetector {
private:
    size_t d_model;
    size_t num_entities;
    
    Matrix W;  // (d_model × num_entities)
    Matrix b;  // (num_entities,)
    
    Matrix grad_W;
    Matrix grad_b;
    
    Matrix input_cache;
    
public:
    EntityDetector(size_t d_model, size_t num_entities);
    
    Matrix forward(const Matrix& sequence_representations);
    Matrix backward(const Matrix& grad_output);
    void updateParameters(double learning_rate);
};

// ============================================================================
// COMPLETE BERT-NLU MODEL
// ============================================================================

/**
 * @brief Complete BERT model for Multi-Task NLU
 * 
 * Tasks:
 * 1. Intent Classification
 * 2. Slot Tagging
 * 3. Entity Detection
 */
class BERTForNLU {
private:
    std::unique_ptr<BERTEmbedding> embedding;
    std::unique_ptr<BERTEncoder> encoder;
    std::unique_ptr<IntentClassifier> intent_head;
    std::unique_ptr<SlotTagger> slot_head;
    std::unique_ptr<EntityDetector> entity_head;
    
    size_t d_model;
    size_t vocab_size;
    size_t num_intents;
    size_t num_slots;
    size_t num_entities;
    
    // Cache
    Matrix encoder_output_cache;
    
public:
    BERTForNLU(size_t vocab_size, size_t d_model, size_t num_heads, 
               size_t d_ff, size_t num_layers, size_t max_seq_length,
               size_t num_intents, size_t num_slots, size_t num_entities);
    
    /**
     * @brief Forward pass for all three tasks
     * @return tuple of (intent_logits, slot_logits, entity_logits)
     */
    std::tuple<Matrix, Matrix, Matrix> forward(const std::vector<int>& token_ids);
    
    /**
     * @brief Backward pass with multi-task gradients
     */
    void backward(const Matrix& grad_intent, const Matrix& grad_slots, 
                  const Matrix& grad_entities);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Predict intent, slots, and entities
     */
    std::tuple<int, std::vector<int>, std::vector<int>> predict(
        const std::vector<int>& token_ids);
};

#endif // BERT_ENCODER_H
