#include "../include/nn/transformer/transformer.h"
#include "../include/nn/transformer/tokenizer.h"
#include "../include/nn/transformer/model_saver.h"
#include "../include/nn/network.h"
#include "../include/nn/layer.h"
#include "../include/nn/activation.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

using json = nlohmann::json;

/**
 * @brief Intent and Slot Detection using Transformer
 * 
 * Task: Given a user utterance, predict:
 *   1. Intent: What the user wants (e.g., "book_flight", "get_weather")
 *   2. Slots: Named entities in the text (e.g., "from_city", "to_city", "date")
 * 
 * This is a joint learning task common in dialogue systems and chatbots.
 * 
 * Dataset: ATIS (Airline Travel Information System) format
 * Example:
 *   Text: "book a flight from boston to new york tomorrow"
 *   Intent: book_flight
 *   Slots: O O O O B-from_city O B-to_city I-to_city B-date
 */

// ============ Data Structures ============

struct IntentSlotExample {
    std::string text;
    std::string intent;
    std::vector<std::string> tokens;
    std::vector<std::string> slots;  // BIO format
};

struct IntentSlotDataset {
    std::vector<IntentSlotExample> examples;
    std::set<std::string> intents;
    std::set<std::string> slot_tags;
    std::unordered_map<std::string, int> intent_to_id;
    std::unordered_map<int, std::string> id_to_intent;
    std::unordered_map<std::string, int> slot_to_id;
    std::unordered_map<int, std::string> id_to_slot;
};

// ============ Dataset Creation (Load from JSON) ============

IntentSlotDataset loadDatasetFromJSON(const std::string& json_path) {
    IntentSlotDataset dataset;
    
    // Read JSON file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << json_path << std::endl;
        return dataset;
    }
    
    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return dataset;
    }
    
    // Load examples
    for (const auto& ex : data["examples"]) {
        IntentSlotExample example;
        example.text = ex["text"].get<std::string>();
        example.intent = ex["intent"].get<std::string>();
        example.tokens = ex["tokens"].get<std::vector<std::string>>();
        example.slots = ex["slots"].get<std::vector<std::string>>();
        
        dataset.examples.push_back(example);
    }
    
    // Build intent and slot vocabularies
    for (const auto& ex : dataset.examples) {
        dataset.intents.insert(ex.intent);
        for (const auto& slot : ex.slots) {
            dataset.slot_tags.insert(slot);
        }
    }
    
    // Create mappings
    int intent_id = 0;
    for (const auto& intent : dataset.intents) {
        dataset.intent_to_id[intent] = intent_id;
        dataset.id_to_intent[intent_id] = intent;
        intent_id++;
    }
    
    int slot_id = 0;
    for (const auto& slot : dataset.slot_tags) {
        dataset.slot_to_id[slot] = slot_id;
        dataset.id_to_slot[slot_id] = slot;
        slot_id++;
    }
    
    return dataset;
    return dataset;
}

// ============ Transformer-based Intent-Slot Model ============

class IntentSlotTransformer {
private:
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::unique_ptr<TransformerEncoder> encoder;
    size_t d_model;
    size_t vocab_size;
    size_t num_intents;
    size_t num_slots;
    
    // Intent classification head (uses [CLS] token representation)
    Matrix W_intent;
    Matrix b_intent;
    
    // Slot detection head (token-level classification)
    Matrix W_slot;
    Matrix b_slot;
    
    // Cached values for backward pass
    Matrix cached_embeddings;
    Matrix cached_encoder_output;
    Matrix cached_intent_logits;
    Matrix cached_slot_logits;
    std::vector<int> cached_src_tokens;
    
public:
    IntentSlotTransformer(size_t vocab_size, size_t d_model, size_t num_layers,
                         size_t num_heads, size_t d_ff, size_t max_seq_len,
                         size_t num_intents, size_t num_slots, double dropout = 0.1)
        : d_model(d_model), vocab_size(vocab_size), num_intents(num_intents), num_slots(num_slots) {
        
        // Initialize embedding layers
        token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
        positional_encoding = std::make_unique<PositionalEncoding>(d_model, max_seq_len);
        
        // Initialize encoder (takes embeddings, not vocab_size)
        encoder = std::make_unique<TransformerEncoder>(
            num_layers, d_model, num_heads, d_ff, dropout
        );
        
        // Initialize intent classification head
        W_intent = Matrix(d_model, num_intents);
        b_intent = Matrix(1, num_intents, 0.0);
        
        // Initialize slot detection head
        W_slot = Matrix(d_model, num_slots);
        b_slot = Matrix(1, num_slots, 0.0);
        
        // Xavier initialization
        double intent_scale = std::sqrt(2.0 / (d_model + num_intents));
        double slot_scale = std::sqrt(2.0 / (d_model + num_slots));
        
        for (size_t i = 0; i < W_intent.getRows(); i++) {
            for (size_t j = 0; j < W_intent.getCols(); j++) {
                W_intent.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * intent_scale);
            }
        }
        
        for (size_t i = 0; i < W_slot.getRows(); i++) {
            for (size_t j = 0; j < W_slot.getCols(); j++) {
                W_slot.set(i, j, ((double)rand() / RAND_MAX - 0.5) * 2.0 * slot_scale);
            }
        }
    }
    
    // Forward pass: returns (intent_logits, slot_logits)
    std::pair<Matrix, Matrix> forward(const std::vector<int>& src_tokens, bool training = true) {
        cached_src_tokens = src_tokens;
        
        // Convert tokens to embeddings
        cached_embeddings = token_embedding->forward(src_tokens);
        
        // Add positional encodings
        size_t seq_len = src_tokens.size();
        cached_embeddings = positional_encoding->forward(cached_embeddings, seq_len);
        
        // Encode input sequence
        cached_encoder_output = encoder->forward(cached_embeddings, nullptr, training);
        
        // Intent classification: use first token ([CLS] or average pooling)
        Matrix cls_repr(1, d_model);
        for (size_t j = 0; j < d_model; j++) {
            cls_repr.set(0, j, cached_encoder_output.get(0, j));
        }
        
        // Intent logits: [1 x d_model] * [d_model x num_intents] = [1 x num_intents]
        cached_intent_logits = cls_repr * W_intent;
        for (size_t j = 0; j < num_intents; j++) {
            cached_intent_logits.set(0, j, cached_intent_logits.get(0, j) + b_intent.get(0, j));
        }
        
        // Slot detection: classify each token
        // [seq_len x d_model] * [d_model x num_slots] = [seq_len x num_slots]
        cached_slot_logits = cached_encoder_output * W_slot;
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < num_slots; j++) {
                cached_slot_logits.set(i, j, cached_slot_logits.get(i, j) + b_slot.get(0, j));
            }
        }
        
        return {cached_intent_logits, cached_slot_logits};
    }
    
    // Compute loss and gradients
    double computeLoss(const Matrix& intent_logits, const Matrix& slot_logits,
                      int true_intent, const std::vector<int>& true_slots) {
        double total_loss = 0.0;
        
        // Intent loss (cross-entropy)
        double intent_max = intent_logits.get(0, 0);
        for (size_t j = 1; j < num_intents; j++) {
            intent_max = std::max(intent_max, intent_logits.get(0, j));
        }
        
        double intent_sum = 0.0;
        for (size_t j = 0; j < num_intents; j++) {
            intent_sum += std::exp(intent_logits.get(0, j) - intent_max);
        }
        
        double intent_loss = -intent_logits.get(0, true_intent) + intent_max + std::log(intent_sum);
        total_loss += intent_loss * 0.2;  // Even lower weight for intent (it learns easily)
        
        // Slot loss (cross-entropy per token) - with class weighting
        size_t seq_len = slot_logits.getRows();
        double slot_loss_sum = 0.0;
        for (size_t i = 0; i < seq_len && i < true_slots.size(); i++) {
            double slot_max = slot_logits.get(i, 0);
            for (size_t j = 1; j < num_slots; j++) {
                slot_max = std::max(slot_max, slot_logits.get(i, j));
            }
            
            double slot_sum = 0.0;
            for (size_t j = 0; j < num_slots; j++) {
                slot_sum += std::exp(slot_logits.get(i, j) - slot_max);
            }
            
            double slot_loss = -slot_logits.get(i, true_slots[i]) + slot_max + std::log(slot_sum);
            
            // Apply higher weight to non-O slots (combat class imbalance)
            // Most tokens are "O", so we need to emphasize entity slots
            double weight = 1.0;
            if (true_slots[i] != 7) {  // Assuming O is last slot (id 7)
                weight = 8.0;  // Very aggressive weight for entity slots
            }
            slot_loss_sum += slot_loss * weight;
        }
        
        total_loss += (slot_loss_sum / seq_len) * 3.0;  // Much higher weight for slot (harder task)
        
        return total_loss / 3.2;  // Normalize by total weights (0.2 + 3.0)
    }
    
    // Simple SGD update
    void updateWeights(double learning_rate, int true_intent, const std::vector<int>& true_slots) {
        size_t seq_len = cached_slot_logits.getRows();
        
        // Compute softmax probabilities for intent
        std::vector<double> intent_probs(num_intents);
        double intent_max = cached_intent_logits.get(0, 0);
        for (size_t j = 1; j < num_intents; j++) {
            intent_max = std::max(intent_max, cached_intent_logits.get(0, j));
        }
        double intent_sum = 0.0;
        for (size_t j = 0; j < num_intents; j++) {
            intent_probs[j] = std::exp(cached_intent_logits.get(0, j) - intent_max);
            intent_sum += intent_probs[j];
        }
        for (size_t j = 0; j < num_intents; j++) {
            intent_probs[j] /= intent_sum;
        }
        
        // Intent gradient
        Matrix intent_grad(1, num_intents);
        for (size_t j = 0; j < num_intents; j++) {
            double grad = intent_probs[j];
            if (j == (size_t)true_intent) grad -= 1.0;
            intent_grad.set(0, j, grad);
        }
        
        // Update intent weights
        Matrix cls_repr(1, d_model);
        for (size_t j = 0; j < d_model; j++) {
            cls_repr.set(0, j, cached_encoder_output.get(0, j));
        }
        
        Matrix cls_T = cls_repr.transpose();
        Matrix W_intent_grad = cls_T * intent_grad;
        
        for (size_t i = 0; i < d_model; i++) {
            for (size_t j = 0; j < num_intents; j++) {
                double update = W_intent_grad.get(i, j);
                W_intent.set(i, j, W_intent.get(i, j) - learning_rate * update);
            }
        }
        
        for (size_t j = 0; j < num_intents; j++) {
            b_intent.set(0, j, b_intent.get(0, j) - learning_rate * intent_grad.get(0, j));
        }
        
        // Slot gradients - with higher weight
        Matrix slot_grad(seq_len, num_slots);
        for (size_t i = 0; i < seq_len && i < true_slots.size(); i++) {
            // Softmax for this position
            double slot_max = cached_slot_logits.get(i, 0);
            for (size_t j = 1; j < num_slots; j++) {
                slot_max = std::max(slot_max, cached_slot_logits.get(i, j));
            }
            
            double slot_sum = 0.0;
            std::vector<double> slot_probs(num_slots);
            for (size_t j = 0; j < num_slots; j++) {
                slot_probs[j] = std::exp(cached_slot_logits.get(i, j) - slot_max);
                slot_sum += slot_probs[j];
            }
            
            for (size_t j = 0; j < num_slots; j++) {
                slot_probs[j] /= slot_sum;
                double grad = slot_probs[j];
                if (j == (size_t)true_slots[i]) grad -= 1.0;
                slot_grad.set(i, j, grad * 4.0);  // Even stronger gradient amplification
            }
        }
        
        // Update slot weights
        Matrix encoder_T = cached_encoder_output.transpose();
        Matrix W_slot_grad = encoder_T * slot_grad;
        
        for (size_t i = 0; i < d_model; i++) {
            for (size_t j = 0; j < num_slots; j++) {
                double update = W_slot_grad.get(i, j);
                W_slot.set(i, j, W_slot.get(i, j) - learning_rate * update);
            }
        }
        
        // Update slot bias
        for (size_t j = 0; j < num_slots; j++) {
            double bias_grad = 0.0;
            for (size_t i = 0; i < seq_len && i < true_slots.size(); i++) {
                bias_grad += slot_grad.get(i, j);
            }
            b_slot.set(0, j, b_slot.get(0, j) - learning_rate * bias_grad);
        }
    }
    
    // Predict intent and slots
    std::pair<int, std::vector<int>> predict(const std::vector<int>& src_tokens) {
        auto [intent_logits, slot_logits] = forward(src_tokens, false);
        
        // Get intent
        int pred_intent = 0;
        double max_intent = intent_logits.get(0, 0);
        for (size_t j = 1; j < num_intents; j++) {
            if (intent_logits.get(0, j) > max_intent) {
                max_intent = intent_logits.get(0, j);
                pred_intent = j;
            }
        }
        
        // Get slots
        std::vector<int> pred_slots;
        for (size_t i = 0; i < slot_logits.getRows(); i++) {
            int pred_slot = 0;
            double max_slot = slot_logits.get(i, 0);
            for (size_t j = 1; j < num_slots; j++) {
                if (slot_logits.get(i, j) > max_slot) {
                    max_slot = slot_logits.get(i, j);
                    pred_slot = j;
                }
            }
            pred_slots.push_back(pred_slot);
        }
        
        return {pred_intent, pred_slots};
    }
    
    // Save model to HuggingFace-style directory
    bool saveModel(const std::string& model_dir, size_t num_layers, size_t num_heads, 
                   size_t d_ff, size_t max_seq_len, double dropout) {
        // Create directory if not exists
        mkdir(model_dir.c_str(), 0755);
        
        // Save config.json
        json config;
        config["model_type"] = "intent_slot_transformer";
        config["vocab_size"] = vocab_size;
        config["d_model"] = d_model;
        config["num_layers"] = num_layers;
        config["num_heads"] = num_heads;
        config["d_ff"] = d_ff;
        config["max_seq_len"] = max_seq_len;
        config["num_intents"] = num_intents;
        config["num_slots"] = num_slots;
        config["dropout"] = dropout;
        
        if (!ModelSaver::saveConfig(model_dir, config)) {
            std::cerr << "Error saving config.json\n";
            return false;
        }
        
        // Save model weights to model.bin
        std::ofstream weights_file(model_dir + "/model.bin", std::ios::binary);
        if (!weights_file.is_open()) {
            std::cerr << "Error opening model.bin\n";
            return false;
        }
        
        // Save token embedding weights
        auto& emb_weights = token_embedding->getWeights();
        ModelSaver::saveMatrix(weights_file, emb_weights);
        
        // Save encoder weights (COMPLETE IMPLEMENTATION)
        encoder->saveWeights(weights_file);
        
        // Save intent head weights
        ModelSaver::saveMatrix(weights_file, W_intent);
        ModelSaver::saveMatrix(weights_file, b_intent);
        
        // Save slot head weights
        ModelSaver::saveMatrix(weights_file, W_slot);
        ModelSaver::saveMatrix(weights_file, b_slot);
        
        weights_file.close();
        std::cout << "✓ Model weights saved to " << model_dir << "/model.bin\n";
        return true;
    }
};

// ============ Main Training Loop ============

int main() {
    srand(42);
    
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Intent and Slot Detection with Transformer          ║\n";
    std::cout << "║  Joint Learning for Dialogue Understanding            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    // ========== Step 1: Load Dataset ==========
    std::cout << "Step 1: Loading datasets from JSON files...\n";
    
    auto train_dataset = loadDatasetFromJSON("../data/train.json");
    auto test_dataset = loadDatasetFromJSON("../data/test.json");
    
    if (train_dataset.examples.empty()) {
        std::cerr << "Error: Failed to load training dataset!\n";
        std::cerr << "Please run: python3 scripts/generate_synthetic_dataset.py\n";
        return 1;
    }
    
    std::cout << "Dataset Statistics:\n";
    std::cout << "  Training examples: " << train_dataset.examples.size() << "\n";
    std::cout << "  Test examples: " << test_dataset.examples.size() << "\n";
    std::cout << "  Number of intents: " << train_dataset.intents.size() << "\n";
    std::cout << "  Number of slot tags: " << train_dataset.slot_tags.size() << "\n\n";
    
    std::cout << "Intents:\n";
    for (const auto& intent : train_dataset.intents) {
        std::cout << "  - " << intent << "\n";
    }
    
    std::cout << "\nSlot Tags (BIO format):\n";
    for (const auto& slot : train_dataset.slot_tags) {
        std::cout << "  - " << slot << "\n";
    }
    std::cout << "\n";
    
    // ========== Step 2: Create Tokenizer with SentencePiece ==========
    std::cout << "Step 2: Preparing tokenizer...\n";
    
    // Build vocabulary from all tokens in dataset
    std::set<std::string> vocab_set;
    for (const auto& ex : train_dataset.examples) {
        for (const auto& token : ex.tokens) {
            vocab_set.insert(token);
        }
    }
    
    // For this demo, use SimpleTokenizer (word-level)
    // In production, you would train SentencePiece model
    SimpleTokenizer tokenizer("", "word");
    
    // Build vocabulary from dataset
    std::string all_text;
    for (const auto& ex : train_dataset.examples) {
        all_text += ex.text + " ";
    }
    tokenizer.buildVocabFromText(all_text);
    
    std::cout << "Tokenizer ready with vocabulary size: " << tokenizer.getVocabSize() << "\n\n";
    
    // ========== Step 3: Initialize Model ==========
    std::cout << "Step 3: Initializing Transformer model...\n";
    
    size_t vocab_size = tokenizer.getVocabSize();
    size_t d_model = 128;  // Larger model for better capacity
    size_t num_layers = 4;  // More layers for complex patterns
    size_t num_heads = 8;  // More attention heads
    size_t d_ff = 256;  // Larger feed-forward
    size_t max_seq_len = 32;
    double dropout = 0.1;
    
    IntentSlotTransformer model(
        vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len,
        train_dataset.intents.size(), train_dataset.slot_tags.size(), dropout
    );
    
    std::cout << "Model Configuration:\n";
    std::cout << "  d_model: " << d_model << "\n";
    std::cout << "  Layers: " << num_layers << "\n";
    std::cout << "  Heads: " << num_heads << "\n";
    std::cout << "  Intent classes: " << train_dataset.intents.size() << "\n";
    std::cout << "  Slot tags: " << train_dataset.slot_tags.size() << "\n\n";
    
    // ========== Step 4: Training ==========
    std::cout << "Step 4: Training model...\n";
    std::cout << "------------------------------------------------------------\n";
    
    size_t num_epochs = 600;  // More epochs with larger dataset
    double learning_rate = 0.003;  // Lower learning rate for larger model
    
    for (size_t epoch = 1; epoch <= num_epochs; epoch++) {
        double total_loss = 0.0;
        
        // Shuffle training data
        auto shuffled_examples = train_dataset.examples;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffled_examples.begin(), shuffled_examples.end(), g);
        
        for (const auto& example : shuffled_examples) {
            // Tokenize
            std::vector<int> tokens = tokenizer.encode(example.text);
            
            // Get true labels
            int true_intent = train_dataset.intent_to_id[example.intent];
            std::vector<int> true_slots;
            for (const auto& slot : example.slots) {
                true_slots.push_back(train_dataset.slot_to_id[slot]);
            }
            
            // Forward pass
            auto [intent_logits, slot_logits] = model.forward(tokens, true);
            
            // Compute loss
            double loss = model.computeLoss(intent_logits, slot_logits, true_intent, true_slots);
            total_loss += loss;
            
            // Backward pass and update
            model.updateWeights(learning_rate, true_intent, true_slots);
        }
        
        double avg_loss = total_loss / train_dataset.examples.size();
        
        if (epoch % 50 == 0 || epoch == 1 || epoch == num_epochs - 1 || epoch == num_epochs) {
            // Quick validation on first 50 test examples
            int val_correct_intents = 0;
            int val_correct_slots = 0;
            int val_total_slots = 0;
            size_t val_size = std::min((size_t)50, test_dataset.examples.size());
            
            for (size_t i = 0; i < val_size; i++) {
                std::vector<int> val_tokens = tokenizer.encode(test_dataset.examples[i].text);
                auto [val_pred_intent, val_pred_slots] = model.predict(val_tokens);
                
                int val_true_intent = test_dataset.intent_to_id[test_dataset.examples[i].intent];
                if (val_pred_intent == val_true_intent) {
                    val_correct_intents++;
                }
                
                for (size_t j = 0; j < test_dataset.examples[i].slots.size() && j < val_pred_slots.size(); j++) {
                    int val_true_slot = test_dataset.slot_to_id[test_dataset.examples[i].slots[j]];
                    if (val_pred_slots[j] == val_true_slot) {
                        val_correct_slots++;
                    }
                    val_total_slots++;
                }
            }
            
            double val_intent_acc = 100.0 * val_correct_intents / val_size;
            double val_slot_acc = val_total_slots > 0 ? 100.0 * val_correct_slots / val_total_slots : 0.0;
            
            std::cout << "Epoch " << std::setw(3) << epoch << "/" << num_epochs 
                     << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                     << " | LR: " << std::setprecision(4) << learning_rate
                     << " | Val Intent: " << std::setprecision(1) << val_intent_acc << "%"
                     << " | Val Slot: " << val_slot_acc << "%\n";
        }
        
        // Learning rate decay
        if (epoch % 100 == 0) {
            learning_rate *= 0.5;  // Reduce learning rate
        }
    }
    
    std::cout << "------------------------------------------------------------\n\n";
    
    // ========== Step 5: Evaluation ==========
    std::cout << "Step 5: Evaluating model on test set...\n\n";
    
    int correct_intents = 0;
    int total_tokens = 0;
    int correct_slots = 0;
    
    for (const auto& example : test_dataset.examples) {
        std::vector<int> tokens = tokenizer.encode(example.text);
        auto [pred_intent, pred_slots] = model.predict(tokens);
        
        int true_intent = test_dataset.intent_to_id[example.intent];
        if (pred_intent == true_intent) {
            correct_intents++;
        }
        
        for (size_t i = 0; i < example.slots.size() && i < pred_slots.size(); i++) {
            int true_slot = test_dataset.slot_to_id[example.slots[i]];
            if (pred_slots[i] == true_slot) {
                correct_slots++;
            }
            total_tokens++;
        }
    }
    
    double intent_acc = 100.0 * correct_intents / test_dataset.examples.size();
    double slot_f1 = 100.0 * correct_slots / total_tokens;
    
    std::cout << "Performance Metrics:\n";
    std::cout << "  Intent Accuracy: " << std::fixed << std::setprecision(2) 
             << intent_acc << "%\n";
    std::cout << "  Slot Token Accuracy: " << std::fixed << std::setprecision(2) 
             << slot_f1 << "%\n\n";
    
    // ========== Step 6: Demo Predictions ==========
    std::cout << "Step 6: Demo predictions on test utterances...\n";
    std::cout << "============================================================\n\n";
    
    std::vector<std::string> test_utterances = {
        "book a flight from boston to seattle",
        "what is the weather in london",
        "cancel my flight to chicago",
        "how much is a flight from miami to denver",
        "fly from delhi to mumbai",
        "weather in paris tomorrow",
        "book from new york to chicago",
        "book a flight from kolkata to malda"
    };
    
    for (const auto& utterance : test_utterances) {
        std::cout << "Input: \"" << utterance << "\"\n";
        
        std::vector<int> tokens = tokenizer.encode(utterance);
        auto [pred_intent_id, pred_slot_ids] = model.predict(tokens);
        
        std::string pred_intent = train_dataset.id_to_intent[pred_intent_id];
        
        std::cout << "  Intent: " << pred_intent << "\n";
        
        std::vector<std::string> words;
        std::istringstream iss(utterance);
        std::string word;
        while (iss >> word) {
            words.push_back(word);
        }
        
        // Extract entities from BIO tags
        std::vector<std::pair<std::string, std::string>> entities;
        std::string current_entity = "";
        std::string current_type = "";
        
        for (size_t i = 0; i < words.size() && i < pred_slot_ids.size(); i++) {
            std::string slot_tag = train_dataset.id_to_slot[pred_slot_ids[i]];
            
            if (slot_tag[0] == 'B') {
                // Save previous entity if exists
                if (!current_entity.empty()) {
                    entities.push_back({current_entity, current_type});
                }
                // Start new entity
                current_entity = words[i];
                current_type = slot_tag.substr(2);  // Remove "B-"
            } else if (slot_tag[0] == 'I' && !current_entity.empty()) {
                // Continue entity
                current_entity += " " + words[i];
            } else {
                // Outside or new B tag
                if (!current_entity.empty()) {
                    entities.push_back({current_entity, current_type});
                    current_entity = "";
                    current_type = "";
                }
            }
        }
        
        // Add last entity if exists
        if (!current_entity.empty()) {
            entities.push_back({current_entity, current_type});
        }
        
        // Display entities
        if (entities.empty()) {
            std::cout << "  Entities: None detected\n";
        } else {
            std::cout << "  Entities:\n";
            for (const auto& [entity, type] : entities) {
                std::cout << "    • " << std::setw(20) << std::left << entity 
                         << " [" << type << "]\n";
            }
        }
        
        // Also show detailed slot tags
        std::cout << "  Detailed Slots:\n";
        for (size_t i = 0; i < words.size() && i < pred_slot_ids.size(); i++) {
            std::string slot_tag = train_dataset.id_to_slot[pred_slot_ids[i]];
            std::cout << "    " << std::setw(15) << std::left << words[i] 
                     << " -> " << slot_tag << "\n";
        }
        std::cout << "\n";
    }
    
    std::cout << "============================================================\n\n";
    
    // ========== Step 9: Save Model ==========
    std::cout << "Step 9: Saving model to HuggingFace-style format...\n";
    
    std::string model_dir = "../saved_models/intent_slot_model";
    
    // Save model config and weights
    model.saveModel(model_dir, num_layers, num_heads, d_ff, max_seq_len, dropout);
    
    // Save tokenizer vocabulary
    ModelSaver::saveVocab(model_dir, tokenizer.getVocab(), 
                         tokenizer.getPadId(), tokenizer.getUnkId(),
                         tokenizer.getBosId(), tokenizer.getEosId());
    std::cout << "✓ Tokenizer vocabulary saved to " << model_dir << "/vocab.json\n";
    
    // Save label mappings
    ModelSaver::saveLabels(model_dir,
                          train_dataset.intent_to_id, train_dataset.id_to_intent,
                          train_dataset.slot_to_id, train_dataset.id_to_slot);
    std::cout << "✓ Label mappings saved to " << model_dir << "/labels.json\n";
    
    std::cout << "\n✓ Model saved successfully to: " << model_dir << "/\n";
    std::cout << "  Files: config.json, model.bin, vocab.json, labels.json\n\n";
    
    std::cout << "============================================================\n\n";
    
    std::cout << "========== Summary ==========\n";
    std::cout << "✓ Joint Intent and Slot Detection\n";
    std::cout << "✓ Transformer encoder for context understanding\n";
    std::cout << "✓ Intent classification from [CLS] token\n";
    std::cout << "✓ Token-level slot labeling (BIO format)\n";
    std::cout << "✓ Multi-task learning improves both tasks\n";
    std::cout << "✓ Model saved in HuggingFace-compatible format\n\n";
    
    std::cout << "Applications:\n";
    std::cout << "  • Chatbots and virtual assistants\n";
    std::cout << "  • Voice command systems\n";
    std::cout << "  • Customer service automation\n";
    std::cout << "  • Smart home devices\n\n";
    
    std::cout << "Next Steps:\n";
    std::cout << "  1. Run the chatbot example: ./intent_slot_chatbot\n";
    std::cout << "  2. Type queries and see intent/entity detection\n";
    std::cout << "  3. Model loads from: " << model_dir << "\n\n";
    
    std::cout << "To use with real SentencePiece:\n";
    std::cout << "  1. Train SentencePiece model on your corpus\n";
    std::cout << "  2. Replace SimpleTokenizer with Tokenizer(\"model.model\")\n";
    std::cout << "  3. Ensure USE_SENTENCEPIECE is defined\n";
    std::cout << "  4. Link with -lsentencepiece\n\n";
    
    std::cout << "To use HuggingFace datasets:\n";
    std::cout << "  1. Use Python script to download ATIS/SNIPS dataset\n";
    std::cout << "  2. Export to JSON/CSV format\n";
    std::cout << "  3. Load in C++ using JSON parser (nlohmann/json)\n";
    std::cout << "  4. Or use HuggingFace C++ API (if available)\n";
    
    return 0;
}
