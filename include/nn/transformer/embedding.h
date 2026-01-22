#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../matrix.h"
#include "../layer.h"
#include <vector>
#include <cmath>
#include <memory>

/**
 * @brief Token Embedding Layer
 * 
 * Converts token IDs to dense vectors
 * Embedding(token_id) = lookup_table[token_id]
 */
class TokenEmbedding {
private:
    size_t vocab_size;
    size_t embedding_dim;
    Matrix embeddings;  // (vocab_size × embedding_dim)
    Matrix gradients;
    
public:
    /**
     * @brief Constructor
     * @param vocab_size Size of vocabulary
     * @param embedding_dim Dimension of embedding vectors
     */
    TokenEmbedding(size_t vocab_size, size_t embedding_dim);
    
    /**
     * @brief Initialize embeddings
     * @param strategy "random", "normal", "xavier"
     */
    void initializeEmbeddings(const std::string& strategy = "normal");
    
    /**
     * @brief Forward pass - convert token IDs to embeddings
     * @param token_ids Vector of token IDs (batch_size × seq_len)
     * @return Matrix of embeddings (batch_size × seq_len × embedding_dim)
     */
    Matrix forward(const std::vector<std::vector<int>>& token_ids);
    
    /**
     * @brief Single sequence forward
     */
    Matrix forward(const std::vector<int>& token_ids);
    
    /**
     * @brief Backward pass - accumulate gradients for embeddings
     * @param grad_output Gradients from next layer
     * @param token_ids Original token IDs used in forward pass
     */
    void backward(const Matrix& grad_output, 
                  const std::vector<std::vector<int>>& token_ids);
    
    /**
     * @brief Update embeddings
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get embedding matrix (for weight tying)
     */
    Matrix& getEmbeddings() { return embeddings; }
    const Matrix& getEmbeddings() const { return embeddings; }
    
    /**
     * @brief Get weights (alias for getEmbeddings, for consistency)
     */
    const Matrix& getWeights() const { return embeddings; }
    
    /**
     * @brief Set weights (for loading saved models)
     */
    void setWeights(const Matrix& weights) { embeddings = weights; }
    
    size_t getVocabSize() const { return vocab_size; }
    size_t getEmbeddingDim() const { return embedding_dim; }
};

/**
 * @brief Positional Encoding Layer
 * 
 * Adds position information using sinusoidal functions:
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 */
class PositionalEncoding {
private:
    size_t max_seq_len;
    size_t d_model;
    Matrix encoding;  // (max_seq_len × d_model)
    
public:
    /**
     * @brief Constructor
     * @param max_seq_len Maximum sequence length
     * @param d_model Model dimension
     */
    PositionalEncoding(size_t max_seq_len, size_t d_model);
    
    /**
     * @brief Generate sinusoidal positional encodings
     */
    void generateEncodings();
    
    /**
     * @brief Add positional encoding to embeddings
     * @param embeddings Input embeddings (batch_size × seq_len × d_model)
     * @return Embeddings + positional encoding
     */
    Matrix forward(const Matrix& embeddings, size_t seq_len) const;
    
    /**
     * @brief Backward pass (positional encoding is not learned)
     */
    Matrix backward(const Matrix& grad_output) const;
    
    const Matrix& getEncodings() const { return encoding; }
};

/**
 * @brief Combined Embedding Layer
 * 
 * Combines token embedding + positional encoding + dropout
 */
class TransformerEmbedding {
private:
    std::unique_ptr<TokenEmbedding> token_emb;
    std::unique_ptr<PositionalEncoding> pos_enc;
    size_t d_model;
    double dropout_rate;
    double scale_factor;  // sqrt(d_model) for scaling
    bool training;
    
    // Dropout mask for backward pass
    Matrix dropout_mask;
    
public:
    /**
     * @brief Constructor
     * @param vocab_size Vocabulary size
     * @param d_model Model dimension
     * @param max_seq_len Maximum sequence length
     * @param dropout Dropout rate (default: 0.1)
     */
    TransformerEmbedding(size_t vocab_size, size_t d_model, 
                        size_t max_seq_len, double dropout = 0.1);
    
    /**
     * @brief Forward pass
     * @param token_ids Input token IDs
     * @param training Whether in training mode (for dropout)
     * @return Embedded and encoded vectors
     */
    Matrix forward(const std::vector<std::vector<int>>& token_ids, 
                   bool training = true);
    
    Matrix forward(const std::vector<int>& token_ids, bool training = true);
    
    /**
     * @brief Backward pass
     */
    void backward(const Matrix& grad_output, 
                  const std::vector<std::vector<int>>& token_ids);
    
    /**
     * @brief Update parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Get token embedding matrix (for weight tying)
     */
    Matrix& getTokenEmbeddings();
    
    size_t getVocabSize() const;
    size_t getDModel() const;
};

#endif // EMBEDDING_H
