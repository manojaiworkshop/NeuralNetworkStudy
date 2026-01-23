#include "../include/nn/transformer/transformer.h"
#include "../include/nn/transformer/tokenizer.h"
#include "../include/nn/transformer/model_saver.h"
#include "../include/nn/matrix.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iomanip>

using json = nlohmann::json;

/**
 * @brief Interactive Chatbot for Intent and Slot Detection
 * 
 * This example loads a trained IntentSlotTransformer model and provides
 * an interactive interface for users to input queries and see:
 *   - Detected intent (what the user wants)
 *   - Extracted entities (names, locations, dates, etc.)
 *   - Slot labels for each token
 * 
 * Usage:
 *   ./intent_slot_chatbot
 *   > book a flight from boston to new york
 *   Intent: book_flight
 *   Entities:
 *     • boston [from_city]
 *     • new york [to_city]
 */

// ============ Simplified Model Loader (Inference Only) ============

class IntentSlotChatbot {
private:
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::unique_ptr<TransformerEncoder> encoder;
    
    Matrix W_intent;
    Matrix b_intent;
    Matrix W_slot;
    Matrix b_slot;
    
    size_t d_model;
    size_t vocab_size;
    size_t num_intents;
    size_t num_slots;
    
    // Label mappings
    std::unordered_map<int, std::string> id_to_intent;
    std::unordered_map<int, std::string> id_to_slot;
    
    // Tokenizer
    SimpleTokenizer tokenizer;
    
public:
    IntentSlotChatbot(const std::string& model_dir) {
        std::cout << "Loading model from: " << model_dir << "\n\n";
        
        // Load config
        json config = ModelSaver::loadConfig(model_dir);
        vocab_size = config["vocab_size"];
        d_model = config["d_model"];
        size_t num_layers = config["num_layers"];
        size_t num_heads = config["num_heads"];
        size_t d_ff = config["d_ff"];
        size_t max_seq_len = config["max_seq_len"];
        num_intents = config["num_intents"];
        num_slots = config["num_slots"];
        double dropout = config["dropout"];
        
        std::cout << "✓ Config loaded:\n";
        std::cout << "  - vocab_size: " << vocab_size << "\n";
        std::cout << "  - d_model: " << d_model << "\n";
        std::cout << "  - num_layers: " << num_layers << "\n";
        std::cout << "  - num_heads: " << num_heads << "\n";
        std::cout << "  - num_intents: " << num_intents << "\n";
        std::cout << "  - num_slots: " << num_slots << "\n\n";
        
        // Initialize architecture
        token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
        positional_encoding = std::make_unique<PositionalEncoding>(d_model, max_seq_len);
        encoder = std::make_unique<TransformerEncoder>(num_layers, d_model, num_heads, d_ff, dropout);
        
        // Load weights from model.bin
        std::ifstream weights_file(model_dir + "/model.bin", std::ios::binary);
        if (!weights_file.is_open()) {
            throw std::runtime_error("Cannot open model.bin");
        }
        
        // Load token embedding weights
        Matrix emb_weights = ModelSaver::loadMatrix(weights_file);
        token_embedding->setWeights(emb_weights);
        
        // Load encoder weights (COMPLETE IMPLEMENTATION)
        encoder->loadWeights(weights_file);
        
        // Load intent head
        W_intent = ModelSaver::loadMatrix(weights_file);
        b_intent = ModelSaver::loadMatrix(weights_file);
        
        // Load slot head
        W_slot = ModelSaver::loadMatrix(weights_file);
        b_slot = ModelSaver::loadMatrix(weights_file);
        
        weights_file.close();
        std::cout << "✓ Model weights loaded\n\n";
        
        // Load vocabulary
        auto [vocab, pad_id, unk_id, bos_id, eos_id] = ModelSaver::loadVocab(model_dir);
        
        // Initialize tokenizer with special tokens
        tokenizer = SimpleTokenizer(vocab_size, pad_id, unk_id, bos_id, eos_id);
        
        // Set vocabulary
        tokenizer.setVocab(vocab);
        std::cout << "✓ Vocabulary loaded (" << vocab.size() << " tokens)\n\n";
        
        // Load labels
        std::unordered_map<std::string, int> intent_to_id, slot_to_id;
        std::tie(intent_to_id, id_to_intent, slot_to_id, id_to_slot) = 
            ModelSaver::loadLabels(model_dir);
        
        std::cout << "✓ Labels loaded:\n";
        std::cout << "  Intents: ";
        for (const auto& [id, intent] : id_to_intent) {
            std::cout << intent << " ";
        }
        std::cout << "\n  Slots: ";
        for (const auto& [id, slot] : id_to_slot) {
            if (slot != "O") std::cout << slot << " ";
        }
        std::cout << "\n\n";
    }
    
    // Predict intent and slots for a query
    void processQuery(const std::string& query) {
        // Tokenize
        auto tokens = tokenizer.tokenize(query);
        auto token_ids = tokenizer.encode(tokens);
        
        // Forward pass
        Matrix embeddings = token_embedding->forward(token_ids);
        embeddings = positional_encoding->forward(embeddings, token_ids.size());
        Matrix encoder_output = encoder->forward(embeddings, nullptr, false);
        
        // Intent prediction (from first token)
        Matrix cls_repr(1, d_model);
        for (size_t j = 0; j < d_model; j++) {
            cls_repr.set(0, j, encoder_output.get(0, j));
        }
        
        Matrix intent_logits = cls_repr * W_intent;
        for (size_t j = 0; j < num_intents; j++) {
            intent_logits.set(0, j, intent_logits.get(0, j) + b_intent.get(0, j));
        }
        
        int pred_intent = 0;
        double max_intent_score = intent_logits.get(0, 0);
        for (size_t j = 1; j < num_intents; j++) {
            if (intent_logits.get(0, j) > max_intent_score) {
                max_intent_score = intent_logits.get(0, j);
                pred_intent = j;
            }
        }
        
        // Slot prediction (per token)
        Matrix slot_logits = encoder_output * W_slot;
        for (size_t i = 0; i < token_ids.size(); i++) {
            for (size_t j = 0; j < num_slots; j++) {
                slot_logits.set(i, j, slot_logits.get(i, j) + b_slot.get(0, j));
            }
        }
        
        std::vector<int> pred_slots;
        for (size_t i = 0; i < token_ids.size(); i++) {
            int pred_slot = 0;
            double max_slot_score = slot_logits.get(i, 0);
            for (size_t j = 1; j < num_slots; j++) {
                if (slot_logits.get(i, j) > max_slot_score) {
                    max_slot_score = slot_logits.get(i, j);
                    pred_slot = j;
                }
            }
            pred_slots.push_back(pred_slot);
        }
        
        // Display results
        displayResults(query, tokens, pred_intent, pred_slots, intent_logits);
    }
    
    void displayResults(const std::string& query, 
                       const std::vector<std::string>& tokens,
                       int intent_id,
                       const std::vector<int>& slot_ids,
                       const Matrix& intent_logits) {
        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Query Analysis                                          ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
        
        std::cout << "📝 Query: \"" << query << "\"\n\n";
        
        // Intent with confidence
        std::string intent = id_to_intent[intent_id];
        std::cout << "🎯 Intent: " << intent << "\n";
        
        // Show intent probabilities (softmax)
        std::cout << "   Confidence:\n";
        double max_logit = intent_logits.get(0, 0);
        for (size_t j = 1; j < num_intents; j++) {
            max_logit = std::max(max_logit, intent_logits.get(0, j));
        }
        
        double sum_exp = 0.0;
        std::vector<double> probs(num_intents);
        for (size_t j = 0; j < num_intents; j++) {
            probs[j] = std::exp(intent_logits.get(0, j) - max_logit);
            sum_exp += probs[j];
        }
        
        for (size_t j = 0; j < num_intents; j++) {
            probs[j] /= sum_exp;
            if (probs[j] > 0.01) {  // Show only > 1%
                std::cout << "     - " << std::setw(15) << std::left << id_to_intent[j]
                         << ": " << std::fixed << std::setprecision(1) 
                         << (probs[j] * 100) << "%\n";
            }
        }
        
        // Extract entities from slot labels
        std::vector<std::pair<std::string, std::string>> entities;
        std::string current_entity = "";
        std::string current_type = "";
        
        for (size_t i = 0; i < tokens.size() && i < slot_ids.size(); i++) {
            std::string slot_tag = id_to_slot[slot_ids[i]];
            
            if (slot_tag[0] == 'B') {
                // Save previous entity
                if (!current_entity.empty()) {
                    entities.push_back({current_entity, current_type});
                }
                // Start new entity
                current_type = slot_tag.substr(2);  // Remove "B-"
                current_entity = tokens[i];
            } else if (slot_tag[0] == 'I' && !current_entity.empty()) {
                // Continue entity
                current_entity += " " + tokens[i];
            } else {
                // Outside or new B tag
                if (!current_entity.empty()) {
                    entities.push_back({current_entity, current_type});
                    current_entity = "";
                    current_type = "";
                }
            }
        }
        
        // Add last entity
        if (!current_entity.empty()) {
            entities.push_back({current_entity, current_type});
        }
        
        // Display entities
        std::cout << "\n🏷️  Entities:\n";
        if (entities.empty()) {
            std::cout << "   None detected\n";
        } else {
            for (const auto& [entity, type] : entities) {
                std::cout << "   • " << std::setw(20) << std::left << entity 
                         << " [" << type << "]\n";
            }
        }
        
        // Display token-level slots
        std::cout << "\n📋 Token Analysis:\n";
        std::cout << "   " << std::setw(15) << std::left << "Token" 
                 << " | " << "Slot Label\n";
        std::cout << "   " << std::string(15, '-') << "-+-" << std::string(20, '-') << "\n";
        
        for (size_t i = 0; i < tokens.size() && i < slot_ids.size(); i++) {
            std::string slot_tag = id_to_slot[slot_ids[i]];
            std::cout << "   " << std::setw(15) << std::left << tokens[i]
                     << " | " << slot_tag << "\n";
        }
        
        std::cout << "\n════════════════════════════════════════════════════════════\n";
    }
    
    void run() {
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║     Intent & Slot Detection Chatbot                     ║\n";
        std::cout << "║     Interactive Query Understanding                      ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
        
        std::cout << "Type your queries below (or 'quit' to exit):\n\n";
        std::cout << "Example queries:\n";
        std::cout << "  - book a flight from boston to chicago\n";
        std::cout << "  - cancel my flight to london tomorrow\n";
        std::cout << "  - what's the fare from delhi to mumbai\n";
        std::cout << "  - show me weather in paris\n\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        std::string query;
        while (true) {
            std::cout << "You: ";
            std::getline(std::cin, query);
            
            // Trim whitespace
            query.erase(0, query.find_first_not_of(" \t\n\r"));
            query.erase(query.find_last_not_of(" \t\n\r") + 1);
            
            if (query.empty()) continue;
            
            if (query == "quit" || query == "exit" || query == "q") {
                std::cout << "\n👋 Goodbye! Thanks for chatting!\n";
                break;
            }
            
            // Convert to lowercase for better matching
            std::transform(query.begin(), query.end(), query.begin(), ::tolower);
            
            // Process query
            processQuery(query);
        }
    }
};

// ============ Main ============

int main() {
    try {
        std::string model_dir = "../saved_models/intent_slot_model";
        
        IntentSlotChatbot chatbot(model_dir);
        chatbot.run();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << "\n";
        std::cerr << "\nMake sure you've trained the model first:\n";
        std::cerr << "  ./intent_slot_detection_example\n";
        return 1;
    }
    
    return 0;
}
