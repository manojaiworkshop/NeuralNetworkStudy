#include "../../include/nn/transformer/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>

// Note: SentencePiece integration requires linking with sentencepiece library
// For now, we'll provide a basic implementation and a fallback SimpleTokenizer

#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>

Tokenizer::Tokenizer(const std::string& model_path) {
    processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
    
    const auto status = processor->Load(model_path);
    if (!status.ok()) {
        std::cerr << "Failed to load SentencePiece model: " << status.ToString() << std::endl;
        processor.reset();
        return;
    }
    
    vocab_size = processor->GetPieceSize();
    pad_id = processor->PieceToId("<pad>");
    unk_id = processor->unk_id();
    bos_id = processor->bos_id();
    eos_id = processor->eos_id();
    
    std::cout << "Loaded SentencePiece model with vocabulary size: " << vocab_size << std::endl;
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!processor) return {};
    
    std::vector<int> ids;
    processor->Encode(text, &ids);
    return ids;
}

std::vector<std::vector<int>> Tokenizer::encodeBatch(
    const std::vector<std::string>& texts,
    size_t max_length,
    bool add_bos,
    bool add_eos) const {
    
    if (!processor) return {};
    
    std::vector<std::vector<int>> batch;
    
    for (const auto& text : texts) {
        std::vector<int> ids = encode(text);
        
        // Add special tokens
        if (add_bos) {
            ids.insert(ids.begin(), bos_id);
        }
        if (add_eos && ids.size() < max_length) {
            ids.push_back(eos_id);
        }
        
        // Truncate or pad
        if (ids.size() > max_length) {
            ids.resize(max_length);
        } else {
            while (ids.size() < max_length) {
                ids.push_back(pad_id);
            }
        }
        
        batch.push_back(ids);
    }
    
    return batch;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    if (!processor) return "";
    
    std::string text;
    processor->Decode(ids, &text);
    return text;
}

std::vector<std::string> Tokenizer::decodeBatch(
    const std::vector<std::vector<int>>& batch_ids) const {
    
    std::vector<std::string> texts;
    for (const auto& ids : batch_ids) {
        texts.push_back(decode(ids));
    }
    return texts;
}

std::string Tokenizer::idToToken(int id) const {
    if (!processor) return "";
    return processor->IdToPiece(id);
}

int Tokenizer::tokenToId(const std::string& token) const {
    if (!processor) return unk_id;
    return processor->PieceToId(token);
}

bool Tokenizer::isLoaded() const {
    return processor != nullptr;
}

void Tokenizer::printInfo() const {
    if (!processor) {
        std::cout << "No SentencePiece model loaded\n";
        return;
    }
    
    std::cout << "========== Tokenizer Info ==========\n";
    std::cout << "Vocabulary Size: " << vocab_size << "\n";
    std::cout << "PAD ID: " << pad_id << " (" << idToToken(pad_id) << ")\n";
    std::cout << "UNK ID: " << unk_id << " (" << idToToken(unk_id) << ")\n";
    std::cout << "BOS ID: " << bos_id << " (" << idToToken(bos_id) << ")\n";
    std::cout << "EOS ID: " << eos_id << " (" << idToToken(eos_id) << ")\n";
    std::cout << "====================================\n";
}

#else
// Fallback implementation without SentencePiece

Tokenizer::Tokenizer(const std::string& model_path) {
    processor = nullptr;
    std::cerr << "SentencePiece not available. Please use SimpleTokenizer instead.\n";
    vocab_size = 0;
    pad_id = 0;
    unk_id = 1;
    bos_id = 2;
    eos_id = 3;
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string& text) const {
    return {};
}

std::vector<std::vector<int>> Tokenizer::encodeBatch(
    const std::vector<std::string>& texts, size_t max_length,
    bool add_bos, bool add_eos) const {
    return {};
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    return "";
}

std::vector<std::string> Tokenizer::decodeBatch(
    const std::vector<std::vector<int>>& batch_ids) const {
    return {};
}

std::string Tokenizer::idToToken(int id) const {
    return "";
}

int Tokenizer::tokenToId(const std::string& token) const {
    return unk_id;
}

bool Tokenizer::isLoaded() const {
    return false;
}

void Tokenizer::printInfo() const {
    std::cout << "SentencePiece not available\n";
}

#endif

// ============ SimpleTokenizer ============

SimpleTokenizer::SimpleTokenizer(const std::string& vocab_file, const std::string& type) {
    // Set special token IDs
    pad_id = 0;
    unk_id = 1;
    bos_id = 2;
    eos_id = 3;
    
    // Initialize vocabulary with special tokens
    vocab["<pad>"] = pad_id;
    vocab["<unk>"] = unk_id;
    vocab["<bos>"] = bos_id;
    vocab["<eos>"] = eos_id;
    
    inv_vocab[pad_id] = "<pad>";
    inv_vocab[unk_id] = "<unk>";
    inv_vocab[bos_id] = "<bos>";
    inv_vocab[eos_id] = "<eos>";
    
    vocab_size = 4;
    
    // Set tokenization type
    if (type == "char") {
        this->type = TokenizationType::CHAR;
    } else if (type == "bpe") {
        this->type = TokenizationType::BPE;
    } else {
        this->type = TokenizationType::WORD;
    }
    
    // Load vocabulary if file exists
    if (!vocab_file.empty()) {
        loadVocab(vocab_file);
    }
}

void SimpleTokenizer::buildVocab(const std::string& corpus_path,
                                size_t target_vocab_size,
                                size_t min_freq) {
    std::ifstream file(corpus_path);
    if (!file.is_open()) {
        std::cerr << "Could not open corpus file: " << corpus_path << std::endl;
        return;
    }
    
    // Count word/character frequencies
    std::map<std::string, size_t> freq;
    std::string line;
    
    while (std::getline(file, line)) {
        if (type == TokenizationType::CHAR) {
            for (char c : line) {
                std::string token(1, c);
                freq[token]++;
            }
        } else {
            // Word tokenization (whitespace split)
            std::istringstream iss(line);
            std::string word;
            while (iss >> word) {
                freq[word]++;
            }
        }
    }
    
    file.close();
    
    // Sort by frequency
    std::vector<std::pair<std::string, size_t>> sorted_freq(freq.begin(), freq.end());
    std::sort(sorted_freq.begin(), sorted_freq.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Build vocabulary
    int current_id = vocab_size;
    for (const auto& [token, count] : sorted_freq) {
        if (count < min_freq) break;
        if (vocab.size() >= target_vocab_size) break;
        
        if (vocab.find(token) == vocab.end()) {
            vocab[token] = current_id;
            inv_vocab[current_id] = token;
            current_id++;
        }
    }
    
    vocab_size = vocab.size();
    std::cout << "Built vocabulary with " << vocab_size << " tokens\n";
}

void SimpleTokenizer::buildVocabFromText(const std::string& text) {
    // Reset vocab to just special tokens
    vocab.clear();
    inv_vocab.clear();
    
    vocab["<pad>"] = pad_id;
    vocab["<unk>"] = unk_id;
    vocab["<bos>"] = bos_id;
    vocab["<eos>"] = eos_id;
    
    inv_vocab[pad_id] = "<pad>";
    inv_vocab[unk_id] = "<unk>";
    inv_vocab[bos_id] = "<bos>";
    inv_vocab[eos_id] = "<eos>";
    
    int current_id = 4;
    
    if (type == TokenizationType::CHAR) {
        // Build character vocabulary
        std::set<char> unique_chars(text.begin(), text.end());
        for (char c : unique_chars) {
            std::string token(1, c);
            if (vocab.find(token) == vocab.end()) {
                vocab[token] = current_id;
                inv_vocab[current_id] = token;
                current_id++;
            }
        }
    } else {
        // Build word vocabulary
        std::istringstream iss(text);
        std::string word;
        std::set<std::string> unique_words;
        while (iss >> word) {
            unique_words.insert(word);
        }
        for (const auto& word : unique_words) {
            if (vocab.find(word) == vocab.end()) {
                vocab[word] = current_id;
                inv_vocab[current_id] = word;
                current_id++;
            }
        }
    }
    
    vocab_size = vocab.size();
    std::cout << "Built vocabulary with " << vocab_size << " tokens from text\n";
}

void SimpleTokenizer::setVocab(const std::unordered_map<std::string, int>& new_vocab) {
    vocab = new_vocab;
    inv_vocab.clear();
    
    // Build inverse vocabulary
    for (const auto& [token, id] : vocab) {
        inv_vocab[id] = token;
    }
    
    vocab_size = vocab.size();
}

bool SimpleTokenizer::saveVocab(const std::string& output_path) const {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Could not open file for writing: " << output_path << std::endl;
        return false;
    }
    
    // Write vocabulary (sorted by ID)
    for (size_t i = 0; i < vocab_size; i++) {
        auto it = inv_vocab.find(i);
        if (it != inv_vocab.end()) {
            file << it->second << "\n";
        }
    }
    
    file.close();
    std::cout << "Saved vocabulary to " << output_path << std::endl;
    return true;
}

bool SimpleTokenizer::loadVocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Could not open vocabulary file: " << vocab_path << std::endl;
        return false;
    }
    
    vocab.clear();
    inv_vocab.clear();
    
    // Re-add special tokens
    vocab["<pad>"] = pad_id;
    vocab["<unk>"] = unk_id;
    vocab["<bos>"] = bos_id;
    vocab["<eos>"] = eos_id;
    
    inv_vocab[pad_id] = "<pad>";
    inv_vocab[unk_id] = "<unk>";
    inv_vocab[bos_id] = "<bos>";
    inv_vocab[eos_id] = "<eos>";
    
    int current_id = 4;
    std::string token;
    
    while (std::getline(file, token)) {
        if (!token.empty() && vocab.find(token) == vocab.end()) {
            vocab[token] = current_id;
            inv_vocab[current_id] = token;
            current_id++;
        }
    }
    
    file.close();
    vocab_size = vocab.size();
    
    std::cout << "Loaded vocabulary with " << vocab_size << " tokens from " << vocab_path << std::endl;
    return true;
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    
    if (type == TokenizationType::CHAR) {
        // Character-level tokenization
        for (char c : text) {
            std::string token(1, c);
            auto it = vocab.find(token);
            ids.push_back(it != vocab.end() ? it->second : unk_id);
        }
    } else {
        // Word-level tokenization
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            auto it = vocab.find(word);
            ids.push_back(it != vocab.end() ? it->second : unk_id);
        }
    }
    
    return ids;
}

std::vector<std::string> SimpleTokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    
    if (type == TokenizationType::CHAR) {
        // Character-level tokenization
        for (char c : text) {
            tokens.push_back(std::string(1, c));
        }
    } else {
        // Word-level tokenization
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

std::vector<int> SimpleTokenizer::encode(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    for (const auto& token : tokens) {
        auto it = vocab.find(token);
        ids.push_back(it != vocab.end() ? it->second : unk_id);
    }
    return ids;
}

std::string SimpleTokenizer::decode(const std::vector<int>& ids) const {
    std::string text;
    
    for (size_t i = 0; i < ids.size(); i++) {
        int id = ids[i];
        
        // Skip special tokens in decoding
        if (id == pad_id || id == bos_id || id == eos_id) {
            continue;
        }
        
        auto it = inv_vocab.find(id);
        if (it != inv_vocab.end()) {
            if (type == TokenizationType::CHAR) {
                text += it->second;
            } else {
                if (!text.empty()) text += " ";
                text += it->second;
            }
        }
    }
    
    return text;
}

std::vector<std::vector<int>> SimpleTokenizer::encodeBatch(
    const std::vector<std::string>& texts,
    size_t max_length,
    bool add_bos,
    bool add_eos) const {
    
    std::vector<std::vector<int>> batch;
    
    for (const auto& text : texts) {
        std::vector<int> ids = encode(text);
        
        // Add special tokens
        if (add_bos) {
            ids.insert(ids.begin(), bos_id);
        }
        if (add_eos && ids.size() < max_length) {
            ids.push_back(eos_id);
        }
        
        // Truncate or pad
        if (ids.size() > max_length) {
            ids.resize(max_length);
        } else {
            while (ids.size() < max_length) {
                ids.push_back(pad_id);
            }
        }
        
        batch.push_back(ids);
    }
    
    return batch;
}

std::vector<std::string> SimpleTokenizer::decodeBatch(
    const std::vector<std::vector<int>>& batch_ids) const {
    
    std::vector<std::string> texts;
    for (const auto& ids : batch_ids) {
        texts.push_back(decode(ids));
    }
    return texts;
}

void SimpleTokenizer::printInfo() const {
    std::cout << "========== SimpleTokenizer Info ==========\n";
    std::cout << "Type: ";
    switch (type) {
        case TokenizationType::WORD: std::cout << "Word-level\n"; break;
        case TokenizationType::CHAR: std::cout << "Character-level\n"; break;
        case TokenizationType::BPE: std::cout << "BPE\n"; break;
    }
    std::cout << "Vocabulary Size: " << vocab_size << "\n";
    std::cout << "PAD ID: " << pad_id << "\n";
    std::cout << "UNK ID: " << unk_id << "\n";
    std::cout << "BOS ID: " << bos_id << "\n";
    std::cout << "EOS ID: " << eos_id << "\n";
    std::cout << "==========================================\n";
}
