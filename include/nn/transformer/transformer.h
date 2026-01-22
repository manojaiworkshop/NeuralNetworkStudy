#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../matrix.h"
#include "embedding.h"
#include "encoder.h"
#include "decoder.h"
#include <memory>
#include <vector>
#include <string>

/**
 * @brief Complete Transformer Model (Encoder-Decoder)
 * 
 * Full implementation of "Attention Is All You Need" architecture
 * 
 * Components:
 * 1. Source & Target Embeddings (with positional encoding)
 * 2. Encoder Stack
 * 3. Decoder Stack
 * 4. Output Linear + Softmax
 */
class Transformer {
private:
    // Model configuration
    size_t src_vocab_size;
    size_t tgt_vocab_size;
    size_t d_model;
    size_t num_heads;
    size_t num_encoder_layers;
    size_t num_decoder_layers;
    size_t d_ff;
    size_t max_seq_len;
    double dropout;
    
    // Components
    std::unique_ptr<TransformerEmbedding> src_embedding;
    std::unique_ptr<TransformerEmbedding> tgt_embedding;
    std::unique_ptr<TransformerEncoder> encoder;
    std::unique_ptr<TransformerDecoder> decoder;
    
    // Output layer (project to vocabulary)
    Matrix output_projection;  // (d_model × tgt_vocab_size)
    Matrix output_bias;        // (tgt_vocab_size)
    Matrix dW_out, db_out;     // Gradients
    
    // Special tokens
    int pad_token_id;
    int bos_token_id;  // Beginning of sequence
    int eos_token_id;  // End of sequence
    
    // Cached for backward pass
    Matrix cached_src_tokens;
    Matrix cached_tgt_tokens;
    Matrix cached_encoder_output;
    Matrix cached_decoder_output;
    
public:
    /**
     * @brief Constructor
     * @param src_vocab_size Source vocabulary size
     * @param tgt_vocab_size Target vocabulary size
     * @param d_model Model dimension (default: 512)
     * @param num_heads Number of attention heads (default: 8)
     * @param num_encoder_layers Number of encoder layers (default: 6)
     * @param num_decoder_layers Number of decoder layers (default: 6)
     * @param d_ff Feed-forward dimension (default: 2048)
     * @param max_seq_len Maximum sequence length (default: 512)
     * @param dropout Dropout rate (default: 0.1)
     * @param pad_token Beginning of sequence token ID (default: 0)
     * @param bos_token Beginning of sequence token ID (default: 1)
     * @param eos_token End of sequence token ID (default: 2)
     */
    Transformer(size_t src_vocab_size, size_t tgt_vocab_size,
               size_t d_model = 512, size_t num_heads = 8,
               size_t num_encoder_layers = 6, size_t num_decoder_layers = 6,
               size_t d_ff = 2048, size_t max_seq_len = 512,
               double dropout = 0.1,
               int pad_token_id = 0, int bos_token_id = 1, int eos_token_id = 2);
    
    /**
     * @brief Initialize output projection weights
     */
    void initializeOutputLayer();
    
    /**
     * @brief Forward pass
     * @param src_tokens Source token IDs (batch × src_seq_len)
     * @param tgt_tokens Target token IDs (batch × tgt_seq_len)
     * @param training Whether in training mode
     * @return Logits (batch × tgt_seq_len × tgt_vocab_size)
     */
    Matrix forward(const std::vector<std::vector<int>>& src_tokens,
                   const std::vector<std::vector<int>>& tgt_tokens,
                   bool training = true);
    
    /**
     * @brief Compute loss (Cross-Entropy)
     * @param logits Model predictions
     * @param targets Target token IDs
     * @return Loss value
     */
    double computeLoss(const Matrix& logits,
                      const std::vector<std::vector<int>>& targets);
    
    /**
     * @brief Backward pass
     * @param logits Model predictions
     * @param targets Target token IDs
     */
    void backward(const Matrix& logits,
                  const std::vector<std::vector<int>>& targets);
    
    /**
     * @brief Update all parameters
     * @param learning_rate Learning rate
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Train on a batch
     * @param src_tokens Source sequences
     * @param tgt_tokens Target sequences
     * @param learning_rate Learning rate
     * @return Loss value
     */
    double trainStep(const std::vector<std::vector<int>>& src_tokens,
                    const std::vector<std::vector<int>>& tgt_tokens,
                    double learning_rate);
    
    /**
     * @brief Greedy decoding (for inference)
     * @param src_tokens Source sequence (single sample)
     * @param max_length Maximum generation length
     * @return Generated token IDs
     */
    std::vector<int> greedyDecode(const std::vector<int>& src_tokens,
                                   size_t max_length);
    
    /**
     * @brief Beam search decoding
     * @param src_tokens Source sequence
     * @param beam_size Beam size
     * @param max_length Maximum generation length
     * @return Generated token IDs
     */
    std::vector<int> beamSearch(const std::vector<int>& src_tokens,
                                size_t beam_size, size_t max_length);
    
    /**
     * @brief Get encoder-decoder attention weights (for visualization)
     */
    std::vector<std::vector<Matrix>> getEncoderAttentionWeights() const;
    std::vector<std::vector<Matrix>> getDecoderSelfAttentionWeights() const;
    std::vector<std::vector<Matrix>> getDecoderCrossAttentionWeights() const;
    
    /**
     * @brief Save model to file
     */
    bool saveModel(const std::string& filename) const;
    
    /**
     * @brief Load model from file
     */
    bool loadModel(const std::string& filename);
    
    /**
     * @brief Print model summary
     */
    void summary() const;
    
    /**
     * @brief Get total parameter count
     */
    size_t getParameterCount() const;
};

/**
 * @brief Decoder-only Transformer (GPT-style)
 * 
 * For autoregressive language modeling
 */
class DecoderOnlyTransformer {
private:
    size_t vocab_size;
    size_t d_model;
    size_t num_layers;
    size_t num_heads;
    size_t d_ff;
    size_t max_seq_len;
    double dropout;
    
    std::unique_ptr<TransformerEmbedding> embedding;
    std::unique_ptr<TransformerDecoder> decoder;
    
    Matrix output_projection;
    Matrix output_bias;
    
public:
    DecoderOnlyTransformer(size_t vocab_size, size_t d_model = 768,
                          size_t num_layers = 12, size_t num_heads = 12,
                          size_t d_ff = 3072, size_t max_seq_len = 1024,
                          double dropout = 0.1);
    
    /**
     * @brief Forward pass (language modeling)
     * @param tokens Input token IDs
     * @param training Whether in training mode
     * @return Logits for next token prediction
     */
    Matrix forward(const std::vector<std::vector<int>>& tokens,
                   bool training = true);
    
    /**
     * @brief Generate text autoregressively
     * @param prompt Initial token IDs
     * @param max_length Maximum generation length
     * @param temperature Sampling temperature
     * @return Generated token IDs
     */
    std::vector<int> generate(const std::vector<int>& prompt,
                              size_t max_length, double temperature = 1.0);
    
    void summary() const;
};

/**
 * @brief Encoder-only Transformer (BERT-style)
 * 
 * For encoding/classification tasks
 */
class EncoderOnlyTransformer {
private:
    size_t vocab_size;
    size_t d_model;
    size_t num_layers;
    size_t num_heads;
    size_t d_ff;
    size_t max_seq_len;
    double dropout;
    
    std::unique_ptr<TransformerEmbedding> embedding;
    std::unique_ptr<TransformerEncoder> encoder;
    
public:
    EncoderOnlyTransformer(size_t vocab_size, size_t d_model = 768,
                          size_t num_layers = 12, size_t num_heads = 12,
                          size_t d_ff = 3072, size_t max_seq_len = 512,
                          double dropout = 0.1);
    
    /**
     * @brief Encode input sequence
     * @param tokens Input token IDs
     * @param training Whether in training mode
     * @return Encoded representations
     */
    Matrix forward(const std::vector<std::vector<int>>& tokens,
                   bool training = true);
    
    /**
     * @brief Get [CLS] token embedding (for classification)
     */
    Matrix getPooledOutput(const Matrix& encoder_output);
    
    void summary() const;
};

#endif // TRANSFORMER_H
