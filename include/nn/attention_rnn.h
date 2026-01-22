#ifndef ATTENTION_RNN_H
#define ATTENTION_RNN_H

#include "rnn.h"
#include "lstm.h"
#include "gru.h"
#include "attention.h"
#include <memory>
#include <vector>

/**
 * @file attention_rnn.h
 * @brief RNN/LSTM/GRU with Attention Mechanism
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY ATTENTION FOR RNN?
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * Standard RNN/LSTM:
 * ─────────────────
 * Encoder: [x₁, x₂, ..., xₙ] → h_final (single context)
 *                               ↓
 * Decoder:                   Decode from h_final
 * 
 * Problem: All information compressed into single vector!
 * 
 * With Attention:
 * ──────────────
 * Encoder: [x₁, x₂, ..., xₙ] → [h₁, h₂, ..., hₙ]
 *                               ↓   ↓   ↓    ↓
 * Decoder at step t:         Attention over all hᵢ
 *                               ↓
 *                            context_t (focused context)
 *                               ↓
 *                            Use for prediction
 * 
 * Benefits:
 * ✓ No information bottleneck
 * ✓ Better for long sequences
 * ✓ Can "look back" at all inputs
 * ✓ Interpretable (visualize what it attends to)
 * 
 * Performance Improvements:
 * • Machine Translation: +5-15 BLEU points
 * • Text Summarization: +3-8 ROUGE points  
 * • Speech Recognition: -10-30% WER
 * • Time Series: -5-20% MSE
 */

/**
 * @brief Encoder-Decoder with Attention
 * 
 * Architecture:
 * 1. Encoder RNN/LSTM/GRU processes input sequence
 * 2. Decoder RNN/LSTM/GRU with attention over encoder states
 * 3. Each decoder step computes attention over all encoder outputs
 */
class AttentionRNN {
private:
    // Encoder
    std::unique_ptr<RNNLayer> encoder_rnn;
    std::unique_ptr<LSTMLayer> encoder_lstm;
    std::unique_ptr<GRULayer> encoder_gru;
    std::string encoder_type;  // "rnn", "lstm", or "gru"
    
    // Decoder  
    std::unique_ptr<RNNLayer> decoder_rnn;
    std::unique_ptr<LSTMLayer> decoder_lstm;
    std::unique_ptr<GRULayer> decoder_gru;
    std::string decoder_type;
    
    // Attention mechanism
    std::unique_ptr<Attention> attention;
    
    // Output projection
    Matrix W_out;
    Matrix b_out;
    Matrix dW_out;
    Matrix db_out;
    
    // Cached data
    std::vector<Matrix> encoder_outputs;
    std::vector<Matrix> decoder_outputs;
    std::vector<Matrix> attention_weights_history;
    std::vector<Matrix> context_vectors;
    
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    
public:
    /**
     * @brief Constructor
     * @param input_size Input dimension
     * @param hidden_size Hidden state dimension
     * @param output_size Output dimension
     * @param encoder_type "rnn", "lstm", or "gru"
     * @param decoder_type "rnn", "lstm", or "gru"
     * @param attention_type "dot", "additive", or "scaled"
     */
    AttentionRNN(size_t input_size, size_t hidden_size, size_t output_size,
                 const std::string& encoder_type = "lstm",
                 const std::string& decoder_type = "lstm",
                 const std::string& attention_type = "additive");
    
    /**
     * @brief Encode input sequence
     * @param sequence Input sequence
     * @return Encoder hidden states
     */
    std::vector<Matrix> encode(const std::vector<Matrix>& sequence);
    
    /**
     * @brief Decode with attention
     * @param encoder_outputs All encoder hidden states
     * @param target_sequence Target sequence (for training)
     * @param teacher_forcing Whether to use teacher forcing
     * @return Decoder outputs
     */
    std::vector<Matrix> decode(const std::vector<Matrix>& encoder_outputs,
                                const std::vector<Matrix>& target_sequence,
                                bool teacher_forcing = true);
    
    /**
     * @brief Forward pass (encode + decode)
     * @param input_sequence Input sequence
     * @param target_sequence Target sequence
     * @return Output sequence and attention weights
     */
    std::pair<std::vector<Matrix>, std::vector<Matrix>> forward(
        const std::vector<Matrix>& input_sequence,
        const std::vector<Matrix>& target_sequence,
        bool teacher_forcing = true);
    
    /**
     * @brief Predict (inference mode, no teacher forcing)
     * @param input_sequence Input sequence
     * @param max_length Maximum output length
     * @return Predicted sequence
     */
    std::vector<Matrix> predict(const std::vector<Matrix>& input_sequence,
                                 size_t max_length);
    
    /**
     * @brief Get attention weights from last forward pass
     */
    std::vector<Matrix> getAttentionWeights() const {
        return attention_weights_history;
    }
    
    /**
     * @brief Visualize attention weights
     */
    void visualizeAttention(const std::vector<std::string>& input_labels,
                           const std::vector<std::string>& output_labels) const;
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Reset gradients
     */
    void resetGradients();
    
    /**
     * @brief Get parameter count
     */
    int getParameterCount() const;
};

/**
 * @brief Attention-based LSTM (simplified interface)
 */
class AttentionLSTM {
private:
    LSTMLayer encoder;
    LSTMLayer decoder;
    std::unique_ptr<Attention> attention;
    
    Matrix W_out;
    Matrix b_out;
    
    std::vector<Matrix> encoder_outputs;
    std::vector<Matrix> attention_weights_history;
    
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    
public:
    AttentionLSTM(size_t input_size, size_t hidden_size, size_t output_size,
                  const std::string& attention_type = "additive");
    
    std::pair<std::vector<Matrix>, std::vector<Matrix>> forward(
        const std::vector<Matrix>& input_sequence,
        const std::vector<Matrix>& target_sequence);
    
    std::vector<Matrix> predict(const std::vector<Matrix>& input_sequence,
                                size_t max_length);
    
    std::vector<Matrix> getAttentionWeights() const {
        return attention_weights_history;
    }
    
    void updateParameters(double learning_rate);
    int getParameterCount() const;
};

/**
 * @brief Attention-based GRU (simplified interface)
 */
class AttentionGRU {
private:
    GRULayer encoder;
    GRULayer decoder;
    std::unique_ptr<Attention> attention;
    
    Matrix W_out;
    Matrix b_out;
    
    std::vector<Matrix> encoder_outputs;
    std::vector<Matrix> attention_weights_history;
    
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    
public:
    AttentionGRU(size_t input_size, size_t hidden_size, size_t output_size,
                 const std::string& attention_type = "additive");
    
    std::pair<std::vector<Matrix>, std::vector<Matrix>> forward(
        const std::vector<Matrix>& input_sequence,
        const std::vector<Matrix>& target_sequence);
    
    std::vector<Matrix> predict(const std::vector<Matrix>& input_sequence,
                                size_t max_length);
    
    std::vector<Matrix> getAttentionWeights() const {
        return attention_weights_history;
    }
    
    void updateParameters(double learning_rate);
    int getParameterCount() const;
};

#endif // ATTENTION_RNN_H
