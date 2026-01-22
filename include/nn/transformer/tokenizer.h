#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declaration for SentencePiece
namespace sentencepiece {
    class SentencePieceProcessor;
}

/**
 * @brief Tokenizer wrapper for SentencePiece
 * 
 * Handles text ↔ token ID conversion using SentencePiece
 */
class Tokenizer {
private:
#ifdef USE_SENTENCEPIECE
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor;
#else
    void* processor;  // Dummy pointer for fallback mode
#endif
    
    int pad_id;
    int unk_id;
    int bos_id;
    int eos_id;
    
    size_t vocab_size;
    
public:
    /**
     * @brief Constructor - load SentencePiece model
     * @param model_path Path to .model file
     */
    explicit Tokenizer(const std::string& model_path);
    
    /**
     * @brief Destructor
     */
    ~Tokenizer();
    
    /**
     * @brief Encode text to token IDs
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text) const;
    
    /**
     * @brief Encode multiple texts (batched)
     * @param texts Vector of input texts
     * @param max_length Maximum sequence length (pad/truncate)
     * @param add_bos Add beginning of sequence token
     * @param add_eos Add end of sequence token
     * @return Batch of token ID sequences
     */
    std::vector<std::vector<int>> encodeBatch(
        const std::vector<std::string>& texts,
        size_t max_length = 512,
        bool add_bos = false,
        bool add_eos = false) const;
    
    /**
     * @brief Decode token IDs to text
     * @param ids Vector of token IDs
     * @return Decoded text
     */
    std::string decode(const std::vector<int>& ids) const;
    
    /**
     * @brief Decode batch of sequences
     * @param batch_ids Batch of token ID sequences
     * @return Vector of decoded texts
     */
    std::vector<std::string> decodeBatch(
        const std::vector<std::vector<int>>& batch_ids) const;
    
    /**
     * @brief Get vocabulary size
     */
    size_t getVocabSize() const { return vocab_size; }
    
    /**
     * @brief Get special token IDs
     */
    int getPadId() const { return pad_id; }
    int getUnkId() const { return unk_id; }
    int getBosId() const { return bos_id; }
    int getEosId() const { return eos_id; }
    
    /**
     * @brief Get token string from ID
     */
    std::string idToToken(int id) const;
    
    /**
     * @brief Get ID from token string
     */
    int tokenToId(const std::string& token) const;
    
    /**
     * @brief Check if model loaded successfully
     */
    bool isLoaded() const;
    
    /**
     * @brief Print tokenizer info
     */
    void printInfo() const;
};

/**
 * @brief Simple vocabulary-based tokenizer (fallback if no SentencePiece)
 * 
 * Basic word/character level tokenization
 */
class SimpleTokenizer {
private:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
    
    int pad_id;
    int unk_id;
    int bos_id;
    int eos_id;
    
    size_t vocab_size;
    
    enum class TokenizationType {
        WORD,      // Whitespace splitting
        CHAR,      // Character level
        BPE        // Simple BPE (not full SentencePiece)
    };
    
    TokenizationType type;
    
public:
    /**
     * @brief Default constructor
     */
    SimpleTokenizer() : pad_id(0), unk_id(1), bos_id(2), eos_id(3), vocab_size(4), type(TokenizationType::WORD) {}
    
    /**
     * @brief Constructor with special token IDs
     */
    SimpleTokenizer(size_t vocab_size, int pad, int unk, int bos, int eos)
        : pad_id(pad), unk_id(unk), bos_id(bos), eos_id(eos), vocab_size(vocab_size), type(TokenizationType::WORD) {}
    
    /**
     * @brief Constructor
     * @param vocab_file Path to vocabulary file (one token per line)
     * @param type Tokenization type
     */
    SimpleTokenizer(const std::string& vocab_file,
                   const std::string& type = "word");
    
    /**
     * @brief Build vocabulary from text corpus
     * @param corpus_path Path to training corpus
     * @param vocab_size Target vocabulary size
     * @param min_freq Minimum frequency threshold
     */
    void buildVocab(const std::string& corpus_path,
                   size_t vocab_size = 30000,
                   size_t min_freq = 2);
    
    /**
     * @brief Build vocabulary from text string directly (for examples)
     * @param text Text to build vocabulary from
     */
    void buildVocabFromText(const std::string& text);
    
    /**
     * @brief Set vocabulary directly (for loading from saved models)
     * @param new_vocab Vocabulary map
     */
    void setVocab(const std::unordered_map<std::string, int>& new_vocab);
    
    /**
     * @brief Get vocabulary map
     */
    const std::unordered_map<std::string, int>& getVocab() const { return vocab; }
    
    /**
     * @brief Save vocabulary to file
     */
    bool saveVocab(const std::string& output_path) const;
    
    /**
     * @brief Load vocabulary from file
     */
    bool loadVocab(const std::string& vocab_path);
    
    /**
     * @brief Encode text to token IDs
     */
    std::vector<int> encode(const std::string& text) const;
    
    /**
     * @brief Tokenize text into tokens (for inspection, uses same logic as encode)
     */
    std::vector<std::string> tokenize(const std::string& text) const;
    
    /**
     * @brief Encode tokens to IDs
     */
    std::vector<int> encode(const std::vector<std::string>& tokens) const;
    
    /**
     * @brief Decode token IDs to text
     */
    std::string decode(const std::vector<int>& ids) const;
    
    /**
     * @brief Encode batch
     */
    std::vector<std::vector<int>> encodeBatch(
        const std::vector<std::string>& texts,
        size_t max_length = 512,
        bool add_bos = false,
        bool add_eos = false) const;
    
    /**
     * @brief Decode batch
     */
    std::vector<std::string> decodeBatch(
        const std::vector<std::vector<int>>& batch_ids) const;
    
    size_t getVocabSize() const { return vocab_size; }
    int getPadId() const { return pad_id; }
    int getUnkId() const { return unk_id; }
    int getBosId() const { return bos_id; }
    int getEosId() const { return eos_id; }
    
    void printInfo() const;
};

#endif // TOKENIZER_H
