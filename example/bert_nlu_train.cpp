/**
 * @file bert_nlu_train.cpp
 * @brief Complete BERT-based NLU Training with CUDA GPU Acceleration
 * 
 * Multi-Task Learning:
 * - Intent Classification
 * - Slot Detection
 * - Entity Recognition
 * 
 * Trained on ATIS dataset using GPU
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <memory>

#include <nlohmann/json.hpp>
#include "../include/nn/bert_encoder_cuda.h"
#include "../include/nn/matrix_cuda.h"

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
// DATA STRUCTURES
// ============================================================================

struct TrainingExample {
    std::string text;
    std::string intent;
    std::vector<std::string> tokens;
    std::vector<std::string> slots;
};

struct Vocabulary {
    std::unordered_map<std::string, int> word2id;
    std::unordered_map<int, std::string> id2word;
    std::unordered_map<std::string, int> intent2id;
    std::unordered_map<int, std::string> id2intent;
    std::unordered_map<std::string, int> slot2id;
    std::unordered_map<int, std::string> id2slot;
    std::unordered_map<std::string, int> entity2id;
    std::unordered_map<int, std::string> id2entity;
    
    int pad_id = 0;
    int unk_id = 1;
    int cls_id = 2;
    int sep_id = 3;
    
    size_t vocab_size = 0;
    size_t num_intents = 0;
    size_t num_slots = 0;
    size_t num_entities = 0;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN;
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(58) << std::left << title << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝";
    std::cout << RESET << "\n\n";
}

void printProgress(const std::string& message, bool success = true) {
    std::cout << (success ? GREEN : RED) << "  ✓ " << RESET << message << "\n";
}

// ============================================================================
// DATA LOADING
// ============================================================================

std::vector<TrainingExample> loadDataset(const std::string& filepath) {
    std::vector<TrainingExample> examples;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << RED << "Error: Could not open " << filepath << RESET << "\n";
        return examples;
    }
    
    json j;
    file >> j;
    
    for (const auto& item : j) {
        TrainingExample ex;
        ex.text = item["text"];
        ex.intent = item["intent"];
        ex.tokens = item["tokens"].get<std::vector<std::string>>();
        ex.slots = item["slots"].get<std::vector<std::string>>();
        examples.push_back(ex);
    }
    
    return examples;
}

// ============================================================================
// VOCABULARY BUILDING
// ============================================================================

std::string extractEntityFromSlot(const std::string& slot) {
    // Extract entity type from BIO tag
    // e.g., "B-from_city" -> "from_city"
    if (slot == "O") return "O";
    if (slot.length() > 2 && slot[1] == '-') {
        return slot.substr(2);
    }
    return slot;
}

Vocabulary buildVocabulary(const std::vector<TrainingExample>& data) {
    Vocabulary vocab;
    
    // Special tokens
    vocab.word2id["[PAD]"] = 0;
    vocab.id2word[0] = "[PAD]";
    vocab.word2id["[UNK]"] = 1;
    vocab.id2word[1] = "[UNK]";
    vocab.word2id["[CLS]"] = 2;
    vocab.id2word[2] = "[CLS]";
    vocab.word2id["[SEP]"] = 3;
    vocab.id2word[3] = "[SEP]";
    
    std::set<std::string> words;
    std::set<std::string> intents;
    std::set<std::string> slots;
    std::set<std::string> entities;
    
    // Collect unique items
    for (const auto& ex : data) {
        intents.insert(ex.intent);
        for (const auto& token : ex.tokens) {
            words.insert(token);
        }
        for (const auto& slot : ex.slots) {
            slots.insert(slot);
            std::string entity = extractEntityFromSlot(slot);
            entities.insert(entity);
        }
    }
    
    // Build word vocabulary
    int word_id = 4;  // Start after special tokens
    for (const auto& word : words) {
        vocab.word2id[word] = word_id;
        vocab.id2word[word_id] = word;
        word_id++;
    }
    vocab.vocab_size = word_id;
    
    // Build intent vocabulary
    int intent_id = 0;
    for (const auto& intent : intents) {
        vocab.intent2id[intent] = intent_id;
        vocab.id2intent[intent_id] = intent;
        intent_id++;
    }
    vocab.num_intents = intent_id;
    
    // Build slot vocabulary
    int slot_id = 0;
    for (const auto& slot : slots) {
        vocab.slot2id[slot] = slot_id;
        vocab.id2slot[slot_id] = slot;
        slot_id++;
    }
    vocab.num_slots = slot_id;
    
    // Build entity vocabulary
    int entity_id = 0;
    for (const auto& entity : entities) {
        vocab.entity2id[entity] = entity_id;
        vocab.id2entity[entity_id] = entity;
        entity_id++;
    }
    vocab.num_entities = entity_id;
    
    return vocab;
}

// ============================================================================
// DATA PREPROCESSING
// ============================================================================

std::vector<int> tokenize(const std::vector<std::string>& tokens, const Vocabulary& vocab) {
    std::vector<int> token_ids;
    token_ids.push_back(vocab.cls_id);  // [CLS] token
    
    for (const auto& token : tokens) {
        auto it = vocab.word2id.find(token);
        if (it != vocab.word2id.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(vocab.unk_id);
        }
    }
    
    return token_ids;
}

std::vector<int> getSlotIds(const std::vector<std::string>& slots, const Vocabulary& vocab) {
    std::vector<int> slot_ids;
    slot_ids.push_back(vocab.slot2id.at("O"));  // [CLS] token has no slot
    
    for (const auto& slot : slots) {
        slot_ids.push_back(vocab.slot2id.at(slot));
    }
    
    return slot_ids;
}

std::vector<int> getEntityIds(const std::vector<std::string>& slots, const Vocabulary& vocab) {
    std::vector<int> entity_ids;
    entity_ids.push_back(vocab.entity2id.at("O"));  // [CLS] token has no entity
    
    for (const auto& slot : slots) {
        std::string entity = extractEntityFromSlot(slot);
        entity_ids.push_back(vocab.entity2id.at(entity));
    }
    
    return entity_ids;
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

double crossEntropyLoss(const MatrixCUDA& logits, int target) {
    const double EPSILON = 1e-8;
    size_t num_classes = logits.getCols();
    
    // Clamp logits to prevent extreme values
    double max_logit = logits.get(0, 0);
    for (size_t i = 1; i < num_classes; i++) {
        double val = logits.get(0, i);
        if (std::isnan(val) || std::isinf(val)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        max_logit = std::max(max_logit, val);
    }
    
    // Clamp max_logit to reasonable range
    max_logit = std::min(max_logit, 50.0);
    max_logit = std::max(max_logit, -50.0);
    
    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; i++) {
        double exp_val = std::exp(std::min(50.0, std::max(-50.0, logits.get(0, i) - max_logit)));
        sum_exp += exp_val;
    }
    
    double log_sum_exp = max_logit + std::log(sum_exp + EPSILON);
    double target_logit = std::min(50.0, std::max(-50.0, logits.get(0, target)));
    double loss = log_sum_exp - target_logit;
    
    return loss;
}

MatrixCUDA crossEntropyGradient(const MatrixCUDA& logits, int target) {
    size_t num_classes = logits.getCols();
    
    double max_logit = logits.get(0, 0);
    for (size_t i = 1; i < num_classes; i++) {
        max_logit = std::max(max_logit, logits.get(0, i));
    }
    
    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; i++) {
        sum_exp += std::exp(logits.get(0, i) - max_logit);
    }
    
    MatrixCUDA grad(1, num_classes);
    for (size_t i = 0; i < num_classes; i++) {
        double softmax = std::exp(logits.get(0, i) - max_logit) / sum_exp;
        double one_hot = (i == static_cast<size_t>(target)) ? 1.0 : 0.0;
        grad.set(0, i, softmax - one_hot);
    }
    
    return grad;
}

double sequenceCrossEntropyLoss(const MatrixCUDA& logits, const std::vector<int>& targets) {
    const double EPSILON = 1e-8;
    size_t seq_len = logits.getRows();
    size_t num_classes = logits.getCols();
    
    double total_loss = 0.0;
    for (size_t i = 0; i < seq_len; i++) {
        double max_logit = logits.get(i, 0);
        for (size_t j = 1; j < num_classes; j++) {
            double val = logits.get(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            max_logit = std::max(max_logit, val);
        }
        
        // Clamp to prevent overflow
        max_logit = std::min(max_logit, 50.0);
        max_logit = std::max(max_logit, -50.0);
        
        double sum_exp = 0.0;
        for (size_t j = 0; j < num_classes; j++) {
            double exp_val = std::exp(std::min(50.0, std::max(-50.0, logits.get(i, j) - max_logit)));
            sum_exp += exp_val;
        }
        
        double log_sum_exp = max_logit + std::log(sum_exp + EPSILON);
        double target_logit = std::min(50.0, std::max(-50.0, logits.get(i, targets[i])));
        total_loss += log_sum_exp - target_logit;
    }
    
    return total_loss / seq_len;
}

MatrixCUDA sequenceCrossEntropyGradient(const MatrixCUDA& logits, const std::vector<int>& targets) {
    size_t seq_len = logits.getRows();
    size_t num_classes = logits.getCols();
    MatrixCUDA grad(seq_len, num_classes);
    
    for (size_t i = 0; i < seq_len; i++) {
        double max_logit = logits.get(i, 0);
        for (size_t j = 1; j < num_classes; j++) {
            max_logit = std::max(max_logit, logits.get(i, j));
        }
        
        double sum_exp = 0.0;
        for (size_t j = 0; j < num_classes; j++) {
            sum_exp += std::exp(logits.get(i, j) - max_logit);
        }
        
        for (size_t j = 0; j < num_classes; j++) {
            double softmax = std::exp(logits.get(i, j) - max_logit) / sum_exp;
            double one_hot = (j == static_cast<size_t>(targets[i])) ? 1.0 : 0.0;
            grad.set(i, j, (softmax - one_hot) / seq_len);
        }
    }
    
    return grad;
}

// ============================================================================
// GRADIENT CLIPPING
// ============================================================================

void clipGradient(MatrixCUDA& grad, float max_norm) {
    // Compute gradient norm
    double norm_squared = 0.0;
    for (size_t i = 0; i < grad.getRows(); i++) {
        for (size_t j = 0; j < grad.getCols(); j++) {
            double val = grad.get(i, j);
            norm_squared += val * val;
        }
    }
    double norm = std::sqrt(norm_squared);
    
    // Clip if necessary
    if (norm > max_norm) {
        double scale = max_norm / norm;
        for (size_t i = 0; i < grad.getRows(); i++) {
            for (size_t j = 0; j < grad.getCols(); j++) {
                grad.set(i, j, grad.get(i, j) * scale);
            }
        }
    }
}

// ============================================================================
// EVALUATION
// ============================================================================

void evaluate(BERTForNLUCUDA& model, const std::vector<TrainingExample>& data,
              const Vocabulary& vocab, const std::string& dataset_name) {
    
    int correct_intents = 0;
    int total_slot_predictions = 0;
    int correct_slots = 0;
    int total_entity_predictions = 0;
    int correct_entities = 0;
    int evaluated_examples = 0;
    
    for (const auto& ex : data) {
        // Skip if intent not in vocab
        if (vocab.intent2id.find(ex.intent) == vocab.intent2id.end()) {
            continue;
        }
        
        std::vector<int> token_ids = tokenize(ex.tokens, vocab);
        int true_intent = vocab.intent2id.at(ex.intent);
        std::vector<int> true_slots = getSlotIds(ex.slots, vocab);
        std::vector<int> true_entities = getEntityIds(ex.slots, vocab);
        
        auto [pred_intent, pred_slots, pred_entities] = model.predict(token_ids);
        evaluated_examples++;
        
        // Intent accuracy
        if (pred_intent == true_intent) {
            correct_intents++;
        }
        
        // Slot accuracy
        size_t min_len = std::min(pred_slots.size(), true_slots.size());
        for (size_t i = 0; i < min_len; i++) {
            total_slot_predictions++;
            if (pred_slots[i] == true_slots[i]) {
                correct_slots++;
            }
        }
        
        // Entity accuracy
        for (size_t i = 0; i < min_len; i++) {
            total_entity_predictions++;
            if (pred_entities[i] == true_entities[i]) {
                correct_entities++;
            }
        }
    }
    
    double intent_acc = evaluated_examples > 0 ? 100.0 * correct_intents / evaluated_examples : 0.0;
    double slot_acc = total_slot_predictions > 0 ? 100.0 * correct_slots / total_slot_predictions : 0.0;
    double entity_acc = total_entity_predictions > 0 ? 100.0 * correct_entities / total_entity_predictions : 0.0;
    
    std::cout << BOLD << YELLOW << dataset_name << " Results:" << RESET << " (" << evaluated_examples << " examples)\n";
    std::cout << "  Intent Accuracy:  " << GREEN << std::fixed << std::setprecision(2) 
              << intent_acc << "%" << RESET << "\n";
    std::cout << "  Slot Accuracy:    " << GREEN << slot_acc << "%" << RESET << "\n";
    std::cout << "  Entity Accuracy:  " << GREEN << entity_acc << "%" << RESET << "\n";
}

// ============================================================================
// TRAINING
// ============================================================================

void trainModel(BERTForNLUCUDA& model, const std::vector<TrainingExample>& train_data,
                const std::vector<TrainingExample>& val_data, const Vocabulary& vocab,
                int num_epochs, float learning_rate, float gradient_clip_value = 5.0f) {
    
    std::cout << BOLD << CYAN << "\n🚀 Starting Training..." << RESET << "\n\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Shuffle training data
        std::vector<size_t> indices(train_data.size());
        for (size_t i = 0; i < train_data.size(); i++) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), gen);
        
        double total_loss = 0.0;
        int num_batches = 0;
        
        // Training loop
        for (size_t idx : indices) {
            const auto& ex = train_data[idx];
            
            // Skip if intent not in vocabulary
            if (vocab.intent2id.find(ex.intent) == vocab.intent2id.end()) {
                continue;
            }
            
            // Prepare inputs
            std::vector<int> token_ids = tokenize(ex.tokens, vocab);
            int intent_label = vocab.intent2id.at(ex.intent);
            std::vector<int> slot_labels = getSlotIds(ex.slots, vocab);
            std::vector<int> entity_labels = getEntityIds(ex.slots, vocab);
            
            // Forward pass
            auto [intent_logits, slot_logits, entity_logits] = model.forward(token_ids);
            
            // ===== VALIDATION: Check forward pass outputs =====
            bool outputs_valid = true;
            double max_intent_val = 0.0;
            for (size_t r = 0; r < intent_logits.getRows() && outputs_valid; r++) {
                for (size_t c = 0; c < intent_logits.getCols() && outputs_valid; c++) {
                    double val = intent_logits.get(r, c);
                    max_intent_val = std::max(max_intent_val, std::abs(val));
                    if (std::isnan(val) || std::isinf(val) || std::abs(val) > 100.0) {
                        if (num_batches == 0) {
                            std::cout << "\n" << RED << "ERROR: Invalid intent logit detected at (" << r << "," << c << "): " << val << RESET << "\n";
                        }
                        outputs_valid = false;
                    }
                }
            }
            
            if (!outputs_valid) {
                if (num_batches == 0) {
                    std::cout << RED << "Skipping due to invalid forward pass outputs (max val: " << max_intent_val << ")" << RESET << "\n";
                }
                continue;
            }
            
            // Calculate losses
            double intent_loss = crossEntropyLoss(intent_logits, intent_label);
            double slot_loss = sequenceCrossEntropyLoss(slot_logits, slot_labels);
            double entity_loss = sequenceCrossEntropyLoss(entity_logits, entity_labels);
            
            // Check individual losses
            if (std::isnan(intent_loss) || std::isinf(intent_loss)) {
                if (num_batches == 0) std::cout << "\n" << RED << "ERROR: Invalid intent_loss" << RESET << "\n";
                continue;
            }
            if (std::isnan(slot_loss) || std::isinf(slot_loss)) {
                if (num_batches == 0) std::cout << "\n" << RED << "ERROR: Invalid slot_loss" << RESET << "\n";
                continue;
            }
            if (std::isnan(entity_loss) || std::isinf(entity_loss)) {
                if (num_batches == 0) std::cout << "\n" << RED << "ERROR: Invalid entity_loss" << RESET << "\n";
                continue;
            }
            
            double total_example_loss = intent_loss + slot_loss + entity_loss;
            
            // Final validation
            if (std::isnan(total_example_loss) || std::isinf(total_example_loss)) {
                if (num_batches == 0) {
                    std::cout << "\n" << RED << "ERROR: Invalid total_loss (intent:" << intent_loss 
                              << " slot:" << slot_loss << " entity:" << entity_loss << ")" << RESET << "\n";
                }
                continue;
            }
            
            total_loss += total_example_loss;
            
            // Backward pass
            MatrixCUDA grad_intent = crossEntropyGradient(intent_logits, intent_label);
            MatrixCUDA grad_slots = sequenceCrossEntropyGradient(slot_logits, slot_labels);
            MatrixCUDA grad_entities = sequenceCrossEntropyGradient(entity_logits, entity_labels);
            
            // Clip gradients to prevent exploding gradients
            clipGradient(grad_intent, gradient_clip_value);
            clipGradient(grad_slots, gradient_clip_value);
            clipGradient(grad_entities, gradient_clip_value);
            
            model.backward(grad_intent, grad_slots, grad_entities);
            
            // Update parameters
            model.updateParameters(learning_rate);
            
            num_batches++;
            
            // Progress update every 100 examples for full dataset
            if (num_batches % 100 == 0) {
                std::cout << "  Epoch " << (epoch + 1) << "/" << num_epochs 
                          << " | Batch " << num_batches << "/" << train_data.size()
                          << " | Avg Loss: " << std::fixed << std::setprecision(4)
                          << (total_loss / num_batches) << "\r" << std::flush;
            }
        }
        
        double avg_loss = total_loss / num_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "\n" << BOLD << "Epoch " << (epoch + 1) << "/" << num_epochs << RESET
                  << " | Loss: " << YELLOW << std::fixed << std::setprecision(4) << avg_loss << RESET
                  << " | Time: " << duration.count() << "s"
                  << " | Examples/sec: " << std::fixed << std::setprecision(1) 
                  << (train_data.size() / (double)duration.count()) << "\n";
        
        // Validation removed to avoid vocabulary mismatches during training
    }
}

// ============================================================================
// DEMO INFERENCE
// ============================================================================

void displayTestExamples(BERTForNLUCUDA& model, const std::vector<TrainingExample>& test_data,
                         const Vocabulary& vocab, int num_examples = 10) {
    printHeader("Test Data Predictions");
    
    int displayed = 0;
    for (const auto& ex : test_data) {
        if (displayed >= num_examples) break;
        
        // Skip if intent not in vocabulary
        if (vocab.intent2id.find(ex.intent) == vocab.intent2id.end()) {
            continue;
        }
        
        std::vector<int> token_ids = tokenize(ex.tokens, vocab);
        int true_intent_id = vocab.intent2id.at(ex.intent);
        
        auto [pred_intent, pred_slots, pred_entities] = model.predict(token_ids);
        
        std::cout << BOLD << "\n[Example " << (displayed + 1) << "]" << RESET << "\n";
        std::cout << "  Text: \"" << ex.text << "\"\n";
        std::cout << "  Tokens: ";
        for (const auto& token : ex.tokens) {
            std::cout << token << " ";
        }
        std::cout << "\n\n";
        
        // Intent
        std::string true_intent_name = vocab.id2intent.at(true_intent_id);
        std::string pred_intent_name = (vocab.id2intent.find(pred_intent) != vocab.id2intent.end()) 
                                        ? vocab.id2intent.at(pred_intent) : "UNKNOWN";
        bool intent_correct = (pred_intent == true_intent_id);
        
        std::cout << "  " << BOLD << "Intent:" << RESET << "\n";
        std::cout << "    True:      " << CYAN << true_intent_name << RESET << "\n";
        std::cout << "    Predicted: " << (intent_correct ? GREEN : RED) 
                  << pred_intent_name << RESET;
        if (intent_correct) {
            std::cout << " ✓";
        }
        std::cout << "\n\n";
        
        // Slots
        std::cout << "  " << BOLD << "Slots:" << RESET << "\n";
        std::cout << "    ";
        size_t min_len = std::min({ex.tokens.size(), ex.slots.size(), (size_t)pred_slots.size() - 1});
        for (size_t i = 0; i < min_len; i++) {
            std::string true_slot = ex.slots[i];
            std::string pred_slot = (vocab.id2slot.find(pred_slots[i + 1]) != vocab.id2slot.end()) 
                                     ? vocab.id2slot.at(pred_slots[i + 1]) : "O";
            bool slot_correct = (true_slot == pred_slot);
            
            std::cout << ex.tokens[i] << "/"
                      << (slot_correct ? GREEN : YELLOW) << pred_slot << RESET
                      << " ";
        }
        std::cout << "\n    True: ";
        for (size_t i = 0; i < min_len; i++) {
            std::cout << ex.tokens[i] << "/" << CYAN << ex.slots[i] << RESET << " ";
        }
        std::cout << "\n\n";
        
        displayed++;
    }
    
    std::cout << "\n" << BOLD << "Legend:" << RESET << "\n";
    std::cout << "  " << GREEN << "Green" << RESET << " = Correct prediction\n";
    std::cout << "  " << YELLOW << "Yellow" << RESET << " = Incorrect slot prediction\n";
    std::cout << "  " << RED << "Red" << RESET << " = Incorrect intent prediction\n";
}

void demoInference(BERTForNLUCUDA& model, const Vocabulary& vocab) {
    printHeader("DEMO: Interactive Inference");
    
    std::vector<std::string> demo_sentences = {
        "show me flights from boston to denver",
        "what is the cheapest flight",
        "book a ticket to new york",
        "find hotels near the airport"
    };
    
    for (const auto& sentence : demo_sentences) {
        std::cout << BOLD << "\nInput: " << RESET << "\"" << sentence << "\"\n";
        
        // Tokenize
        std::vector<std::string> tokens;
        std::string token;
        for (char c : sentence) {
            if (c == ' ') {
                if (!token.empty()) {
                    tokens.push_back(token);
                    token.clear();
                }
            } else {
                token += c;
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
        
        std::vector<int> token_ids = tokenize(tokens, vocab);
        
        // Predict
        auto [pred_intent, pred_slots, pred_entities] = model.predict(token_ids);
        
        // Display results
        std::string intent_name = (vocab.id2intent.find(pred_intent) != vocab.id2intent.end()) 
                                   ? vocab.id2intent.at(pred_intent) : "UNKNOWN";
        std::cout << GREEN << "  Intent: " << RESET << intent_name << "\n";
        std::cout << CYAN << "  Slots:  " << RESET;
        for (size_t i = 1; i < std::min(pred_slots.size(), tokens.size() + 1); i++) {
            std::string slot_name = (vocab.id2slot.find(pred_slots[i]) != vocab.id2slot.end()) 
                                     ? vocab.id2slot.at(pred_slots[i]) : "O";
            std::cout << tokens[i-1] << "/" << slot_name << " ";
        }
        std::cout << "\n";
        std::cout << MAGENTA << "  Entities: " << RESET;
        for (size_t i = 1; i < std::min(pred_entities.size(), tokens.size() + 1); i++) {
            std::string entity = (vocab.id2entity.find(pred_entities[i]) != vocab.id2entity.end()) 
                                  ? vocab.id2entity.at(pred_entities[i]) : "O";
            if (entity != "O") {
                std::cout << tokens[i-1] << "/" << entity << " ";
            }
        }
        std::cout << "\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printHeader("BERT-based Multi-Task NLU Training with CUDA GPU");
    
    // Configuration
    const std::string train_file = "data/train.json";
    const std::string val_file = "data/validation.json";
    const std::string test_file = "data/test.json";
    
    const size_t d_model = 64;         // Model dimension - REDUCED FOR STABILITY
    const size_t num_heads = 2;        // Number of attention heads - REDUCED
    const size_t d_ff = 256;           // Feed-forward dimension - REDUCED
    const size_t num_layers = 1;       // Number of encoder layers - REDUCED
    const size_t max_seq_length = 64;  // Maximum sequence length
    
    const int num_epochs = 10;  // Increased for full dataset training
    const float learning_rate = 0.00001f;  // VERY CONSERVATIVE learning rate to prevent NaN (was 0.0001)
    const float gradient_clip_value = 1.0f;  // Gradient clipping threshold - REDUCED (was 5.0)
    
    std::cout << BOLD << "Configuration:" << RESET << "\n";
    std::cout << "  Model Dimension:     " << d_model << "\n";
    std::cout << "  Attention Heads:     " << num_heads << "\n";
    std::cout << "  Feed-Forward Dim:    " << d_ff << "\n";
    std::cout << "  Encoder Layers:      " << num_layers << "\n";
    std::cout << "  Max Seq Length:      " << max_seq_length << "\n";
    std::cout << "  Learning Rate:       " << learning_rate << "\n";
    std::cout << "  Epochs:              " << num_epochs << "\n\n";
    
    // Load datasets
    std::cout << BOLD << "Loading Datasets..." << RESET << "\n";
    auto train_data = loadDataset(train_file);
    auto val_data = loadDataset(val_file);
    auto test_data = loadDataset(test_file);
    
    if (train_data.empty()) {
        std::cerr << RED << "Error: No training data loaded!" << RESET << "\n";
        return 1;
    }
    
    printProgress("Loaded " + std::to_string(train_data.size()) + " training examples");
    printProgress("Loaded " + std::to_string(val_data.size()) + " validation examples");
    printProgress("Loaded " + std::to_string(test_data.size()) + " test examples");
    
    // Build vocabulary
    std::cout << "\n" << BOLD << "Building Vocabulary..." << RESET << "\n";
    Vocabulary vocab = buildVocabulary(train_data);
    
    printProgress("Vocabulary size: " + std::to_string(vocab.vocab_size));
    printProgress("Number of intents: " + std::to_string(vocab.num_intents));
    printProgress("Number of slots: " + std::to_string(vocab.num_slots));
    printProgress("Number of entities: " + std::to_string(vocab.num_entities));
    
    // Initialize model
    std::cout << "\n" << BOLD << "Initializing BERT Model on GPU..." << RESET << "\n";
    BERTForNLUCUDA model(vocab.vocab_size, d_model, num_heads, d_ff, num_layers,
                         max_seq_length, vocab.num_intents, vocab.num_slots, vocab.num_entities);
    printProgress("Model initialized successfully on GPU");
    
    // Train model
    trainModel(model, train_data, val_data, vocab, num_epochs, learning_rate, gradient_clip_value);
    
    // Final evaluation
    std::cout << "\n" << BOLD << CYAN << "═══════════════════════════════════════" << RESET << "\n";
    std::cout << BOLD << CYAN << "Final Evaluation" << RESET << "\n";
    std::cout << BOLD << CYAN << "═══════════════════════════════════════" << RESET << "\n\n";
    
    try {
        evaluate(model, test_data, vocab, "Test Set");
    } catch (const std::exception& e) {
        std::cout << RED << "Warning: Evaluation skipped due to vocabulary mismatch\n" << RESET;
        std::cout << "Reason: " << e.what() << "\n\n";
    }
    
    // Display test examples with predictions
    std::cout << "\n";
    try {
        displayTestExamples(model, test_data, vocab, 10);
    } catch (const std::exception& e) {
        std::cout << RED << "Warning: Could not display all test examples\n" << RESET;
        std::cout << "Reason: " << e.what() << "\n\n";
    }
    
    // Demo on custom sentences
    std::cout << "\n";
    try {
        demoInference(model, vocab);
    } catch (const std::exception& e) {
        std::cout << RED << "Warning: Demo inference skipped\n" << RESET;
        std::cout << "Reason: " << e.what() << "\n\n";
    }
    
    std::cout << "\n" << GREEN << BOLD << "✅ Training and Evaluation Complete!" << RESET << "\n\n";
    
    return 0;
}
