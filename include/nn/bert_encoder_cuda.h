#ifndef BERT_ENCODER_CUDA_H
#define BERT_ENCODER_CUDA_H

#include "matrix_cuda.h"
#include "activation_cuda.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

/**
 * @file bert_encoder_cuda.h
 * @brief GPU-Accelerated BERT Encoder for Multi-Task NLU
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * CUDA BERT ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * All operations run on GPU using CUDA kernels for maximum performance
 */

// ============================================================================
// LAYER NORMALIZATION (CUDA)
// ============================================================================

class LayerNormCUDA {
private:
    size_t d_model;
    MatrixCUDA gamma;
    MatrixCUDA beta;
    float eps;
    
    MatrixCUDA input_cache;
    MatrixCUDA mean_cache;
    MatrixCUDA variance_cache;

public:
    LayerNormCUDA(size_t d_model, float epsilon = 1e-6f);
    
    MatrixCUDA forward(const MatrixCUDA& input);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// MULTI-HEAD ATTENTION (CUDA)
// ============================================================================

class MultiHeadAttentionCUDA {
private:
    size_t d_model;
    size_t num_heads;
    size_t d_k;
    
    std::vector<MatrixCUDA> W_Q;
    std::vector<MatrixCUDA> W_K;
    std::vector<MatrixCUDA> W_V;
    MatrixCUDA W_O;
    MatrixCUDA b_O;
    
    std::vector<MatrixCUDA> grad_W_Q;
    std::vector<MatrixCUDA> grad_W_K;
    std::vector<MatrixCUDA> grad_W_V;
    MatrixCUDA grad_W_O;
    MatrixCUDA grad_b_O;
    
    MatrixCUDA input_cache;
    std::vector<MatrixCUDA> Q_cache;
    std::vector<MatrixCUDA> K_cache;
    std::vector<MatrixCUDA> V_cache;
    std::vector<MatrixCUDA> attention_weights_cache;

public:
    MultiHeadAttentionCUDA(size_t d_model, size_t num_heads);
    
    MatrixCUDA forward(const MatrixCUDA& input);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// FEED-FORWARD NETWORK (CUDA)
// ============================================================================

class FeedForwardCUDA {
private:
    size_t d_model;
    size_t d_ff;
    
    MatrixCUDA W1;
    MatrixCUDA b1;
    MatrixCUDA W2;
    MatrixCUDA b2;
    
    MatrixCUDA grad_W1;
    MatrixCUDA grad_b1;
    MatrixCUDA grad_W2;
    MatrixCUDA grad_b2;
    
    MatrixCUDA input_cache;
    MatrixCUDA hidden_cache;

public:
    FeedForwardCUDA(size_t d_model, size_t d_ff);
    
    MatrixCUDA forward(const MatrixCUDA& input);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// TRANSFORMER ENCODER LAYER (CUDA)
// ============================================================================

class TransformerEncoderLayerCUDA {
private:
    std::unique_ptr<MultiHeadAttentionCUDA> attention;
    std::unique_ptr<FeedForwardCUDA> ffn;
    std::unique_ptr<LayerNormCUDA> norm1;
    std::unique_ptr<LayerNormCUDA> norm2;
    
    MatrixCUDA residual1_cache;
    MatrixCUDA residual2_cache;

public:
    TransformerEncoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff);
    
    MatrixCUDA forward(const MatrixCUDA& input);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// BERT ENCODER (CUDA)
// ============================================================================

class BERTEncoderCUDA {
private:
    size_t d_model;
    size_t num_layers;
    std::vector<std::unique_ptr<TransformerEncoderLayerCUDA>> layers;
    std::vector<MatrixCUDA> layer_outputs;

public:
    BERTEncoderCUDA(size_t d_model, size_t num_heads, size_t d_ff, size_t num_layers);
    
    MatrixCUDA forward(const MatrixCUDA& input);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// EMBEDDING LAYER (CUDA)
// ============================================================================

class BERTEmbeddingCUDA {
private:
    size_t vocab_size;
    size_t d_model;
    size_t max_seq_length;
    
    MatrixCUDA token_embeddings;
    MatrixCUDA positional_encodings;
    MatrixCUDA grad_token_embeddings;
    
    std::vector<int> token_ids_cache;

public:
    BERTEmbeddingCUDA(size_t vocab_size, size_t d_model, size_t max_seq_length);
    
    MatrixCUDA forward(const std::vector<int>& token_ids);
    void backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
    
    void initializePositionalEncoding();
};

// ============================================================================
// OUTPUT HEADS (CUDA)
// ============================================================================

class IntentClassifierCUDA {
private:
    size_t d_model;
    size_t num_intents;
    
    MatrixCUDA W;
    MatrixCUDA b;
    MatrixCUDA grad_W;
    MatrixCUDA grad_b;
    MatrixCUDA input_cache;

public:
    IntentClassifierCUDA(size_t d_model, size_t num_intents);
    
    MatrixCUDA forward(const MatrixCUDA& cls_representation);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

class SlotTaggerCUDA {
private:
    size_t d_model;
    size_t num_slots;
    
    MatrixCUDA W;
    MatrixCUDA b;
    MatrixCUDA grad_W;
    MatrixCUDA grad_b;
    MatrixCUDA input_cache;

public:
    SlotTaggerCUDA(size_t d_model, size_t num_slots);
    
    MatrixCUDA forward(const MatrixCUDA& sequence_representations);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

class EntityDetectorCUDA {
private:
    size_t d_model;
    size_t num_entities;
    
    MatrixCUDA W;
    MatrixCUDA b;
    MatrixCUDA grad_W;
    MatrixCUDA grad_b;
    MatrixCUDA input_cache;

public:
    EntityDetectorCUDA(size_t d_model, size_t num_entities);
    
    MatrixCUDA forward(const MatrixCUDA& sequence_representations);
    MatrixCUDA backward(const MatrixCUDA& grad_output);
    void updateParameters(float learning_rate);
};

// ============================================================================
// COMPLETE BERT-NLU MODEL (CUDA)
// ============================================================================

class BERTForNLUCUDA {
private:
    std::unique_ptr<BERTEmbeddingCUDA> embedding;
    std::unique_ptr<BERTEncoderCUDA> encoder;
    std::unique_ptr<IntentClassifierCUDA> intent_head;
    std::unique_ptr<SlotTaggerCUDA> slot_head;
    std::unique_ptr<EntityDetectorCUDA> entity_head;
    
    size_t d_model;
    size_t vocab_size;
    size_t num_intents;
    size_t num_slots;
    size_t num_entities;
    
    MatrixCUDA encoder_output_cache;

public:
    BERTForNLUCUDA(size_t vocab_size, size_t d_model, size_t num_heads, 
                   size_t d_ff, size_t num_layers, size_t max_seq_length,
                   size_t num_intents, size_t num_slots, size_t num_entities);
    
    std::tuple<MatrixCUDA, MatrixCUDA, MatrixCUDA> forward(const std::vector<int>& token_ids);
    
    void backward(const MatrixCUDA& grad_intent, const MatrixCUDA& grad_slots, 
                  const MatrixCUDA& grad_entities);
    
    void updateParameters(float learning_rate);
    
    std::tuple<int, std::vector<int>, std::vector<int>> predict(const std::vector<int>& token_ids);
};

#endif // BERT_ENCODER_CUDA_H
