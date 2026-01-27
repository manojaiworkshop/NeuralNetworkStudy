/**
 * @file joint_nlu_inference.cpp
 * @brief C++ inference for Joint NLU model using LibTorch
 * 
 * Load trained FinBERT model and perform GPU inference for:
 * - Intent classification
 * - Slot detection  
 * - Entity recognition
 */

#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>
#include <sstream>

using json = nlohmann::json;

// ANSI colors
#define RESET "\033[0m"
#define BOLD "\033[1m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RED "\033[31m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"

// ============================================================================
// BERT TOKENIZER (Using HuggingFace via Python subprocess)
// ============================================================================

class BertTokenizer {
private:
    FILE* tokenizer_process;
    int max_length;

public:
    BertTokenizer(int max_len = 64) : max_length(max_len) {
        // Start Python tokenizer subprocess
        tokenizer_process = popen("python3 tokenizer_server.py", "w");
        if (!tokenizer_process) {
            throw std::runtime_error("Failed to start tokenizer subprocess");
        }
    }
    
    ~BertTokenizer() {
        if (tokenizer_process) {
            pclose(tokenizer_process);
        }
    }
    
    struct TokenizerOutput {
        std::vector<int> input_ids;
        std::vector<int> attention_mask;
        std::vector<std::string> tokens;
    };
    
    TokenizerOutput encode(const std::string& text) {
        // Create JSON request
        json request;
        request["text"] = text;
        request["max_length"] = max_length;
        
        // Send to Python tokenizer
        std::string request_str = request.dump() + "\n";
        fprintf(tokenizer_process, "%s", request_str.c_str());
        fflush(tokenizer_process);
        
        // Read response (this is a simplified version - production needs proper IPC)
        // For now, we'll use a workaround with file-based communication
        
        // Fallback: Use Python directly via system call
        std::string cmd = "python3 -c \"from transformers import AutoTokenizer; "
                         "t = AutoTokenizer.from_pretrained('ProsusAI/finbert'); "
                         "e = t('" + text + "', max_length=" + std::to_string(max_length) + 
                         ", padding='max_length', truncation=True); "
                         "import json; "
                         "print(json.dumps({'input_ids': e['input_ids'], 'attention_mask': e['attention_mask']}))\"";
        
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to tokenize text");
        }
        
        char buffer[4096];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        
        // Parse JSON response
        TokenizerOutput output;
        try {
            json response = json::parse(result);
            output.input_ids = response["input_ids"].get<std::vector<int>>();
            output.attention_mask = response["attention_mask"].get<std::vector<int>>();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse tokenizer output: " + std::string(e.what()));
        }
        
        return output;
    }
};

// ============================================================================
// JOINT NLU MODEL INFERENCE
// ============================================================================

class JointNLUInference {
private:
    torch::jit::script::Module model;
    torch::Device device;
    BertTokenizer tokenizer;
    
    // Vocabularies
    std::map<int, std::string> id2intent;
    std::map<int, std::string> id2slot;
    std::map<int, std::string> id2entity;
    int max_length;
    int num_intents;
    int num_slots;
    int num_entities;

public:
    JointNLUInference(const std::string& model_path, const std::string& vocab_path)
        : device(torch::kCUDA) {
        
        std::cout << CYAN << "🤖 Loading Joint NLU Model..." << RESET << "\n";
        
        // Check CUDA availability
        if (!torch::cuda::is_available()) {
            std::cerr << RED << "❌ CUDA not available! Using CPU instead." << RESET << "\n";
            device = torch::Device(torch::kCPU);
        } else {
            std::cout << GREEN << "✅ CUDA available - using GPU" << RESET << "\n";
            std::cout << "   GPU Count: " << torch::cuda::device_count() << "\n";
        }
        
        // Load vocabularies
        loadVocabularies(vocab_path);
        
        // Load model
        try {
            model = torch::jit::load(model_path, device);
            model.eval();
            std::cout << GREEN << "✅ Model loaded successfully" << RESET << "\n";
            std::cout << "   Device: " << (device.is_cuda() ? "GPU" : "CPU") << "\n\n";
        } catch (const c10::Error& e) {
            std::cerr << RED << "❌ Error loading model: " << e.what() << RESET << "\n";
            throw;
        }
    }
    
    void loadVocabularies(const std::string& vocab_path) {
        std::ifstream file(vocab_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vocabulary file: " + vocab_path);
        }
        
        json vocab_data;
        file >> vocab_data;
        
        // Load intent mapping
        for (const auto& item : vocab_data["id_to_intent"].items()) {
            int id = std::stoi(item.key());
            id2intent[id] = item.value();
        }
        
        // Load slot mapping
        for (const auto& item : vocab_data["id_to_slot"].items()) {
            int id = std::stoi(item.key());
            id2slot[id] = item.value();
        }
        
        // Load entity mapping
        for (const auto& item : vocab_data["id_to_entity"].items()) {
            int id = std::stoi(item.key());
            id2entity[id] = item.value();
        }
        
        max_length = vocab_data["max_length"];
        num_intents = vocab_data["num_intents"];
        num_slots = vocab_data["num_slots"];
        num_entities = vocab_data["num_entities"];
        
        std::cout << YELLOW << "📚 Vocabularies loaded:" << RESET << "\n";
        std::cout << "   Intents: " << num_intents << "\n";
        std::cout << "   Slots: " << num_slots << "\n";
        std::cout << "   Entities: " << num_entities << "\n";
    }
    
    struct NLUResult {
        std::string intent;
        float intent_confidence;
        std::vector<std::string> tokens;
        std::vector<std::string> slot_predictions;
        std::vector<std::string> entity_predictions;
        std::vector<float> slot_confidences;
        std::vector<float> entity_confidences;
    };
    
    NLUResult analyze(const std::string& text) {
        NLUResult result;
        
        // Tokenize input using HuggingFace BERT tokenizer
        std::cout << YELLOW << "⏳ Tokenizing with BERT..." << RESET << std::flush;
        auto tokenizer_output = tokenizer.encode(text);
        std::cout << " ✓\n";
        
        // Convert to tensors
        torch::Tensor input_ids_tensor = torch::tensor(tokenizer_output.input_ids)
            .unsqueeze(0)  // Add batch dimension
            .to(device);
        
        torch::Tensor attention_mask_tensor = torch::tensor(tokenizer_output.attention_mask)
            .unsqueeze(0)
            .to(device);
        
        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids_tensor);
        inputs.push_back(attention_mask_tensor);
        
        auto outputs = model.forward(inputs).toTuple();
        
        // Extract outputs: (intent_logits, slot_logits, entity_logits)
        torch::Tensor intent_logits = outputs->elements()[0].toTensor();
        torch::Tensor slot_logits = outputs->elements()[1].toTensor();
        torch::Tensor entity_logits = outputs->elements()[2].toTensor();
        
        // Get predictions
        auto intent_pred = intent_logits.argmax(1).item<int>();
        auto intent_probs = torch::softmax(intent_logits, 1);
        result.intent_confidence = intent_probs[0][intent_pred].item<float>();
        result.intent = id2intent[intent_pred];
        
        // Slot predictions
        auto slot_preds = slot_logits.argmax(2).squeeze(0);
        auto slot_probs = torch::softmax(slot_logits, 2).squeeze(0);
        
        // Entity predictions
        auto entity_preds = entity_logits.argmax(2).squeeze(0);
        auto entity_probs = torch::softmax(entity_logits, 2).squeeze(0);
        
        // We need to use Python tokenizer's output to get the actual word positions
        // For now, we'll use the BERT tokens and filter by attention mask
        std::istringstream iss(text);
        std::string word;
        std::vector<std::string> words;
        while (iss >> word) {
            words.push_back(word);
        }
        
        // Map BERT tokens back to words (simplified - proper implementation needs token offsets)
        int word_idx = 0;
        for (int token_idx = 1; token_idx < max_length - 1 && word_idx < words.size(); token_idx++) {
            if (tokenizer_output.attention_mask[token_idx] == 1) {
                // Check if this is the start of a word (not a subword)
                int slot_id = slot_preds[token_idx].item<int>();
                int entity_id = entity_preds[token_idx].item<int>();
                
                std::string slot_label = id2slot[slot_id];
                std::string entity_label = id2entity[entity_id];
                
                // Only add if it's a B- tag (beginning of entity/slot) or single word
                if (slot_label.substr(0, 2) == "B-" || slot_label == "O") {
                    if (word_idx < words.size()) {
                        result.tokens.push_back(words[word_idx]);
                        result.slot_predictions.push_back(slot_label);
                        result.entity_predictions.push_back(entity_label);
                        result.slot_confidences.push_back(slot_probs[token_idx][slot_id].item<float>());
                        result.entity_confidences.push_back(entity_probs[token_idx][entity_id].item<float>());
                        word_idx++;
                    }
                }
                token_idx++;
            }
        }
        
        return result;
    }
    
    void printResult(const NLUResult& result, const std::string& query) {
        std::cout << BOLD << CYAN << "\n🔍 NLU Analysis" << RESET << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Query: \"" << YELLOW << query << RESET << "\"\n\n";
        
        // Intent
        std::cout << BOLD << "🎯 Intent: " << RESET;
        std::cout << GREEN << result.intent << RESET;
        std::cout << " (confidence: " << std::fixed << std::setprecision(3) 
                  << result.intent_confidence << ")\n\n";
        
        // Slots and Entities
        std::cout << BOLD << "📝 Token Analysis:" << RESET << "\n";
        for (size_t i = 0; i < result.tokens.size(); i++) {
            std::cout << "  ";
            
            // Token
            std::cout << "\"" << result.tokens[i] << "\"";
            
            // Slot
            if (result.slot_predictions[i] != "O") {
                std::cout << " → " << MAGENTA << "Slot:" << result.slot_predictions[i] 
                          << " (" << std::setprecision(2) << result.slot_confidences[i] << ")" << RESET;
            }
            
            // Entity
            if (result.entity_predictions[i] != "O" && 
                result.entity_predictions[i] != "aircraft_code") {
                std::cout << ", " << BLUE << "Entity:" << result.entity_predictions[i]
                          << " (" << std::setprecision(2) << result.entity_confidences[i] << ")" << RESET;
            }
            
            if (result.slot_predictions[i] == "O" && 
                result.entity_predictions[i] == "O") {
                std::cout << " → O";
            }
            
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

// ============================================================================
// MAIN - INTERACTIVE CHAT
// ============================================================================

int main(int argc, char** argv) {
    std::cout << BOLD << CYAN;
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          Joint NLU Inference (C++ + LibTorch + GPU)          ║\n";
    std::cout << "║          Intent + Slot + Entity Detection                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << RESET;
    std::cout << "\n";
    
    try {
        // Initialize model
        JointNLUInference nlu("joint_nlu_model_traced.pt", "joint_nlu_vocab.json");
        
        std::cout << BOLD << YELLOW << "💬 Interactive Chat Mode" << RESET << "\n";
        std::cout << "Enter flight queries (or 'quit' to exit)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        
        // Sample queries
        std::cout << CYAN << "💡 Sample queries:" << RESET << "\n";
        std::cout << "  1. I want to fly from Boston to Denver on Monday\n";
        std::cout << "  2. Show me American Airlines flights to Chicago\n";
        std::cout << "  3. What's the cheapest flight from New York to LA?\n";
        std::cout << "  4. Book a ticket from Seattle to Miami tomorrow\n\n";
        
        // Interactive loop
        std::string query;
        while (true) {
            std::cout << BOLD << GREEN << "🗣️  Enter query: " << RESET;
            std::getline(std::cin, query);
            
            if (query.empty()) continue;
            
            // Check for quit commands
            std::string lower_query = query;
            std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
            if (lower_query == "quit" || lower_query == "exit" || lower_query == "q") {
                std::cout << "\n👋 Goodbye!\n\n";
                break;
            }
            
            // Perform inference
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = nlu.analyze(query);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Display results
            nlu.printResult(result, query);
            std::cout << YELLOW << "⚡ Inference time: " << duration.count() << "ms" << RESET << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << RED << "❌ Error: " << e.what() << RESET << "\n";
        std::cerr << "\nMake sure you've run:\n";
        std::cerr << "  1. python joint_nlu_finbert.py (train model)\n";
        std::cerr << "  2. python export_model_torchscript.py (export model)\n";
        return 1;
    }
    
    return 0;
}
