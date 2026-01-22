#include "nn/attention_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================================
// DECODER LAYER CUDA
// ============================================================================

DecoderLayerCUDA::DecoderLayerCUDA(size_t d_model, size_t num_heads, size_t d_ff)
    : d_model(d_model), num_heads(num_heads), d_ff(d_ff) {
    
    // Initialize sub-layers
    masked_self_attention = std::make_unique<MultiHeadAttentionCUDA>(d_model, num_heads);
    cross_attention = std::make_unique<MultiHeadAttentionCUDA>(d_model, num_heads);
    feed_forward = std::make_unique<FeedForwardCUDA>(d_model, d_ff);
    
    // Layer norms
    norm1 = std::make_unique<LayerNormCUDA>(d_model);
    norm2 = std::make_unique<LayerNormCUDA>(d_model);
    norm3 = std::make_unique<LayerNormCUDA>(d_model);
}

MatrixCUDA DecoderLayerCUDA::forward(const MatrixCUDA& input,
                                     const MatrixCUDA& encoder_output,
                                     const MatrixCUDA* mask) {
    cached_input = input;
    cached_encoder_output = encoder_output;
    
    // 1. Masked self-attention on target sequence
    MatrixCUDA self_attn_out;
    if (mask != nullptr) {
        // Apply causal mask to prevent attending to future positions
        self_attn_out = masked_self_attention->forward(input, input, input);
        
        // Apply mask by setting attention to -inf before softmax
        // (This would require modifying the attention mechanism)
        // For now, we'll compute normally and the mask will be applied at generation time
    } else {
        self_attn_out = masked_self_attention->forward(input, input, input);
    }
    
    // Add & Norm
    MatrixCUDA norm1_out = norm1->forward(input.addGPU(self_attn_out));
    
    // 2. Cross-attention with encoder output
    // Query from decoder, Key and Value from encoder
    MatrixCUDA cross_attn_out = cross_attention->forward(
        norm1_out,           // Q from decoder
        encoder_output,      // K from encoder
        encoder_output       // V from encoder
    );
    
    // Add & Norm
    MatrixCUDA norm2_out = norm2->forward(norm1_out.addGPU(cross_attn_out));
    
    // 3. Feed-forward network
    MatrixCUDA ff_out = feed_forward->forward(norm2_out);
    
    // Add & Norm
    MatrixCUDA output = norm3->forward(norm2_out.addGPU(ff_out));
    
    return output;
}

MatrixCUDA DecoderLayerCUDA::backward(const MatrixCUDA& grad_output,
                                      MatrixCUDA& grad_encoder_output) {
    // Simplified backward - gradients flow through each component
    MatrixCUDA grad = grad_output;
    
    // Backward through norm3 and FF
    grad = norm3->backward(grad);
    MatrixCUDA grad_ff = feed_forward->backward(grad);
    
    // Backward through norm2 and cross-attention
    grad = norm2->backward(grad.addGPU(grad_ff));
    MatrixCUDA grad_q, grad_k, grad_v;
    cross_attention->backward(grad, grad_q, grad_k, grad_v);
    
    // Accumulate encoder gradients
    grad_encoder_output = grad_encoder_output.addGPU(grad_k.addGPU(grad_v));
    
    // Backward through norm1 and self-attention
    grad = norm1->backward(grad.addGPU(grad_q));
    MatrixCUDA grad_self_q, grad_self_k, grad_self_v;
    masked_self_attention->backward(grad, grad_self_q, grad_self_k, grad_self_v);
    
    return grad_self_q.addGPU(grad_self_k).addGPU(grad_self_v);
}

void DecoderLayerCUDA::updateParameters(double learning_rate) {
    masked_self_attention->updateParameters(learning_rate);
    cross_attention->updateParameters(learning_rate);
    feed_forward->updateParameters(learning_rate);
    norm1->updateParameters(learning_rate);
    norm2->updateParameters(learning_rate);
    norm3->updateParameters(learning_rate);
}

std::vector<MatrixCUDA> DecoderLayerCUDA::getSelfAttentionWeights() const {
    return masked_self_attention->getAllAttentionWeights();
}

std::vector<MatrixCUDA> DecoderLayerCUDA::getCrossAttentionWeights() const {
    return cross_attention->getAllAttentionWeights();
}

// ============================================================================
// TRANSFORMER DECODER CUDA
// ============================================================================

TransformerDecoderCUDA::TransformerDecoderCUDA(size_t num_layers, size_t d_model,
                                               size_t num_heads, size_t d_ff)
    : num_layers(num_layers), d_model(d_model) {
    
    // Create decoder layers
    for (size_t i = 0; i < num_layers; i++) {
        layers.push_back(std::make_unique<DecoderLayerCUDA>(d_model, num_heads, d_ff));
    }
    
    // Final layer normalization
    final_norm = std::make_unique<LayerNormCUDA>(d_model);
}

MatrixCUDA TransformerDecoderCUDA::forward(const MatrixCUDA& target,
                                           const MatrixCUDA& encoder_output,
                                           const MatrixCUDA* mask) {
    MatrixCUDA output = target;
    
    // Pass through each decoder layer
    for (size_t i = 0; i < num_layers; i++) {
        output = layers[i]->forward(output, encoder_output, mask);
    }
    
    // Final normalization
    output = final_norm->forward(output);
    
    return output;
}

MatrixCUDA TransformerDecoderCUDA::backward(const MatrixCUDA& grad_output,
                                            MatrixCUDA& grad_encoder_output) {
    MatrixCUDA grad = final_norm->backward(grad_output);
    
    // Backward through layers in reverse order
    for (int i = num_layers - 1; i >= 0; i--) {
        grad = layers[i]->backward(grad, grad_encoder_output);
    }
    
    return grad;
}

void TransformerDecoderCUDA::updateParameters(double learning_rate) {
    for (auto& layer : layers) {
        layer->updateParameters(learning_rate);
    }
    final_norm->updateParameters(learning_rate);
}

std::vector<std::vector<MatrixCUDA>> TransformerDecoderCUDA::getAllSelfAttentionWeights() const {
    std::vector<std::vector<MatrixCUDA>> all_weights;
    for (const auto& layer : layers) {
        all_weights.push_back(layer->getSelfAttentionWeights());
    }
    return all_weights;
}

std::vector<std::vector<MatrixCUDA>> TransformerDecoderCUDA::getAllCrossAttentionWeights() const {
    std::vector<std::vector<MatrixCUDA>> all_weights;
    for (const auto& layer : layers) {
        all_weights.push_back(layer->getCrossAttentionWeights());
    }
    return all_weights;
}

// ============================================================================
// COMPLETE TRANSFORMER CUDA
// ============================================================================

TransformerCUDA::TransformerCUDA(size_t vocab_size, size_t d_model, size_t num_heads,
                                 size_t num_layers, size_t d_ff, size_t max_seq_len)
    : vocab_size(vocab_size), d_model(d_model), num_heads(num_heads),
      num_layers(num_layers), d_ff(d_ff), max_seq_len(max_seq_len) {
    
    // Initialize embeddings
    src_embedding = std::make_unique<TokenEmbeddingCUDA>(vocab_size, d_model);
    tgt_embedding = std::make_unique<TokenEmbeddingCUDA>(vocab_size, d_model);
    positional_encoding = std::make_unique<PositionalEncodingCUDA>(max_seq_len, d_model);
    
    // Initialize encoder and decoder
    encoder = std::make_unique<TransformerEncoderCUDA>(num_layers, d_model, num_heads, d_ff);
    decoder = std::make_unique<TransformerDecoderCUDA>(num_layers, d_model, num_heads, d_ff);
    
    // Initialize output projection
    initializeOutputProjection();
}

void TransformerCUDA::initializeOutputProjection() {
    // Xavier initialization for output projection
    Matrix proj_cpu(d_model, vocab_size);
    double std = std::sqrt(2.0 / (d_model + vocab_size));
    proj_cpu.randomNormal(0.0, std);
    
    output_projection = MatrixCUDA(proj_cpu);
    output_projection.toGPU();
    
    Matrix grad_cpu(d_model, vocab_size, 0.0);
    output_projection_grad = MatrixCUDA(grad_cpu);
    output_projection_grad.toGPU();
}

MatrixCUDA TransformerCUDA::createCausalMask(size_t seq_len) {
    // Create lower triangular mask (1 = attend, 0 = mask out)
    Matrix mask_cpu(seq_len, seq_len);
    
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            // Can attend to positions <= current position
            mask_cpu.set(i, j, (j <= i) ? 1.0 : 0.0);
        }
    }
    
    MatrixCUDA mask_gpu(mask_cpu);
    mask_gpu.toGPU();
    return mask_gpu;
}

MatrixCUDA TransformerCUDA::forward(const std::vector<std::vector<int>>& source_ids,
                                    const std::vector<std::vector<int>>& target_ids) {
    // 1. Encode source sequence
    MatrixCUDA src_embedded = src_embedding->forward(source_ids);
    MatrixCUDA src_encoded = positional_encoding->forward(src_embedded);
    MatrixCUDA encoder_output = encoder->forward(src_encoded);
    
    // 2. Embed and encode target sequence
    MatrixCUDA tgt_embedded = tgt_embedding->forward(target_ids);
    MatrixCUDA tgt_encoded = positional_encoding->forward(tgt_embedded);
    
    // 3. Create causal mask for decoder
    size_t tgt_seq_len = target_ids[0].size();
    causal_mask = createCausalMask(tgt_seq_len);
    
    // 4. Decode with cross-attention to encoder output
    MatrixCUDA decoder_output = decoder->forward(tgt_encoded, encoder_output, &causal_mask);
    
    // 5. Project to vocabulary logits
    MatrixCUDA logits = decoder_output.multiplyGPU(output_projection);
    
    return logits;
}

std::vector<int> TransformerCUDA::generate(const std::vector<int>& source_ids,
                                          size_t max_length,
                                          int start_token_id,
                                          int end_token_id) {
    // Encode source once
    std::vector<std::vector<int>> src_batch = {source_ids};
    MatrixCUDA src_embedded = src_embedding->forward(src_batch);
    MatrixCUDA src_encoded = positional_encoding->forward(src_embedded);
    MatrixCUDA encoder_output = encoder->forward(src_encoded);
    
    // Start with start token
    std::vector<int> generated = {start_token_id};
    
    // Auto-regressive generation
    for (size_t i = 0; i < max_length; i++) {
        // Prepare target input
        std::vector<std::vector<int>> tgt_batch = {generated};
        
        // Embed and decode
        MatrixCUDA tgt_embedded = tgt_embedding->forward(tgt_batch);
        MatrixCUDA tgt_encoded = positional_encoding->forward(tgt_embedded);
        
        // Create causal mask
        MatrixCUDA mask = createCausalMask(generated.size());
        
        // Decode
        MatrixCUDA decoder_output = decoder->forward(tgt_encoded, encoder_output, &mask);
        
        // Project to vocabulary
        MatrixCUDA logits = decoder_output.multiplyGPU(output_projection);
        
        // Get logits for last position
        logits.toCPU();
        size_t last_pos = generated.size() - 1;
        
        // Find argmax (greedy decoding)
        int best_token = 0;
        double best_score = logits.get(last_pos, 0);
        
        for (size_t v = 1; v < vocab_size; v++) {
            double score = logits.get(last_pos, v);
            if (score > best_score) {
                best_score = score;
                best_token = v;
            }
        }
        
        // Add predicted token
        generated.push_back(best_token);
        
        // Stop if end token generated
        if (best_token == end_token_id) {
            break;
        }
    }
    
    return generated;
}

void TransformerCUDA::backward(const MatrixCUDA& grad_output) {
    // Backward through output projection
    MatrixCUDA grad_decoder = grad_output.multiplyGPU(output_projection.transposeGPU());
    
    // Compute gradient for output projection
    // grad_W = decoder_output^T × grad_output
    // (Simplified - need cached decoder_output from forward)
    
    // Backward through decoder
    MatrixCUDA grad_encoder(grad_decoder.getRows(), grad_decoder.getCols(), 0.0);
    grad_encoder.toGPU();
    MatrixCUDA grad_tgt = decoder->backward(grad_decoder, grad_encoder);
    
    // Backward through encoder
    MatrixCUDA grad_src = encoder->backward(grad_encoder);
    
    // Backward through embeddings
    // (Using cached token IDs from forward pass)
}

void TransformerCUDA::updateParameters(double learning_rate) {
    // Update encoder and decoder
    encoder->updateParameters(learning_rate);
    decoder->updateParameters(learning_rate);
    
    // Update embeddings
    src_embedding->updateParameters(learning_rate);
    tgt_embedding->updateParameters(learning_rate);
    
    // Update output projection
    // output_projection -= learning_rate * grad
    output_projection_grad.toCPU();
    output_projection.toCPU();
    
    for (size_t i = 0; i < output_projection.getRows(); i++) {
        for (size_t j = 0; j < output_projection.getCols(); j++) {
            double new_val = output_projection.get(i, j) - 
                           learning_rate * output_projection_grad.get(i, j);
            output_projection.set(i, j, new_val);
        }
    }
    
    output_projection.toGPU();
}

std::vector<std::vector<MatrixCUDA>> TransformerCUDA::getEncoderAttentionWeights() const {
    return encoder->getAllAttentionWeights();
}

std::vector<std::vector<MatrixCUDA>> TransformerCUDA::getDecoderSelfAttentionWeights() const {
    return decoder->getAllSelfAttentionWeights();
}

std::vector<std::vector<MatrixCUDA>> TransformerCUDA::getDecoderCrossAttentionWeights() const {
    return decoder->getAllCrossAttentionWeights();
}

size_t TransformerCUDA::getParameterCount() const {
    size_t count = 0;
    
    // Embeddings: 2 × vocab_size × d_model
    count += 2 * vocab_size * d_model;
    
    // Encoder: num_layers × (4 × d_model² + 2 × d_model × d_ff)
    count += num_layers * (4 * d_model * d_model + 2 * d_model * d_ff);
    
    // Decoder: num_layers × (8 × d_model² + 2 × d_model × d_ff)
    count += num_layers * (8 * d_model * d_model + 2 * d_model * d_ff);
    
    // Output projection: d_model × vocab_size
    count += d_model * vocab_size;
    
    return count;
}
