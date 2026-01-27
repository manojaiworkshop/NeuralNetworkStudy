/**
 * @file intent_cuda_train_working.cpp
 * @brief Working CUDA Intent Detection Training with ATIS Dataset
 * 
 * This version properly uses existing CUDA infrastructure:
 * - Uses NeuralNetworkCUDA for proper GPU acceleration
 * - Loads ATIS dataset in BIO format
 * - Trains on GPU with batch processing
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
#include <chrono>
#include <iomanip>

#include <nlohmann/json.hpp>

// Use existing CUDA network infrastructure
#include "../include/nn/network_cuda.h"
#include "../include/nn/layer_cuda.h"
#include "../include/nn/activation_cuda.h"
#include "../include/nn/loss_cuda.h"
#include "../include/nn/optimizer_cuda.h"

using json = nlohmann::json;

// ANSI colors
#define RESET "\033[0m"
#define BOLD "\033[1m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define RED "\033[31m"
#define BLUE "\033[34m"

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Example {
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
    
    int pad_id = 0;
    int unk_id = 1;
    size_t vocab_size = 0;
    size_t num_intents = 0;
    size_t num_slots = 0;
};

// ============================================================================
// DATASET LOADING
// ============================================================================

std::vector<Example> loadDataset(const std::string& filepath) {
    std::vector<Example> examples;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << RED << "Error: Cannot open " << filepath << RESET << "\n";
        return examples;
    }
    
    json data;
    try {
        file >> data;
        for (const auto& item : data) {
            Example ex;
            ex.text = item["text"];
            ex.intent = item["intent"];
            ex.tokens = item["tokens"].get<std::vector<std::string>>();
            ex.slots = item["slots"].get<std::vector<std::string>>();
            examples.push_back(ex);
        }
    } catch (const std::exception& e) {
        std::cerr << RED << "Error parsing JSON: " << e.what() << RESET << "\n";
    }
    
    return examples;
}

Vocabulary buildVocabulary(const std::vector<Example>& data) {
    Vocabulary vocab;
    
    // Special tokens
    vocab.word2id["<PAD>"] = 0;
    vocab.word2id["<UNK>"] = 1;
    vocab.id2word[0] = "<PAD>";
    vocab.id2word[1] = "<UNK>";
    
    std::set<std::string> words, intents, slots;
    
    // Collect all unique items
    for (const auto& ex : data) {
        for (const auto& token : ex.tokens) {
            words.insert(token);
        }
        intents.insert(ex.intent);
        for (const auto& slot : ex.slots) {
            slots.insert(slot);
        }
    }
    
    // Build word vocabulary
    int word_id = 2;
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
    
    return vocab;
}

void printDatasetStats(const std::vector<Example>& data, const std::string& name) {
    std::cout << BOLD << CYAN << name << " Dataset:" << RESET << "\n";
    std::cout << "  Examples: " << data.size() << "\n";
    
    if (!data.empty()) {
        // Calculate average sequence length
        double avg_len = 0;
        for (const auto& ex : data) {
            avg_len += ex.tokens.size();
        }
        avg_len /= data.size();
        
        std::cout << "  Avg sequence length: " << std::fixed << std::setprecision(1) << avg_len << "\n";
        
        // Show example
        const auto& ex = data[0];
        std::cout << "  Example: \"" << ex.text << "\"\n";
        std::cout << "    Intent: " << ex.intent << "\n";
        std::cout << "    Tokens: ";
        for (size_t i = 0; i < std::min(ex.tokens.size(), (size_t)5); i++) {
            std::cout << ex.tokens[i] << " ";
        }
        if (ex.tokens.size() > 5) std::cout << "...";
        std::cout << "\n";
        std::cout << "    Slots:  ";
        for (size_t i = 0; i < std::min(ex.slots.size(), (size_t)5); i++) {
            std::cout << ex.slots[i] << " ";
        }
        if (ex.slots.size() > 5) std::cout << "...";
        std::cout << "\n";
    }
    std::cout << "\n";
}

void printVocabStats(const Vocabulary& vocab) {
    std::cout << BOLD << YELLOW << "Vocabulary Statistics:" << RESET << "\n";
    std::cout << "  Words: " << vocab.vocab_size << "\n";
    std::cout << "  Intents: " << vocab.num_intents << " (";
    for (size_t i = 0; i < vocab.num_intents; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << vocab.id2intent.at(i);
    }
    std::cout << ")\n";
    std::cout << "  Slots: " << vocab.num_slots << " (";
    for (size_t i = 0; i < std::min((size_t)5, vocab.num_slots); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << vocab.id2slot.at(i);
    }
    if (vocab.num_slots > 5) std::cout << ", ...";
    std::cout << ")\n\n";
}

// ============================================================================
// DATA PREPROCESSING FOR NEURAL NETWORK
// ============================================================================

std::pair<MatrixCUDA, MatrixCUDA> prepareTrainingData(
    const std::vector<Example>& examples, 
    const Vocabulary& vocab,
    size_t max_seq_len = 32) {
    
    size_t num_examples = examples.size();
    
    // Prepare input matrix (examples × max_seq_len)
    MatrixCUDA X(num_examples, max_seq_len);
    
    // Prepare intent labels (examples × num_intents) - one-hot encoded  
    MatrixCUDA y_intent(num_examples, vocab.num_intents);
    
    // Process each example
    for (size_t i = 0; i < num_examples; i++) {
        const auto& ex = examples[i];
        
        // Convert tokens to IDs and pad/truncate
        for (size_t j = 0; j < max_seq_len; j++) {
            int token_id = vocab.pad_id;
            if (j < ex.tokens.size()) {
                auto it = vocab.word2id.find(ex.tokens[j]);
                token_id = (it != vocab.word2id.end()) ? it->second : vocab.unk_id;
            }
            X.set(i, j, static_cast<double>(token_id));
        }
        
        // Set intent label (one-hot encoding)
        int intent_id = vocab.intent2id.at(ex.intent);
        for (size_t j = 0; j < vocab.num_intents; j++) {
            y_intent.set(i, j, (j == intent_id) ? 1.0 : 0.0);
        }
    }
    
    // Move to GPU
    X.toGPU();
    y_intent.toGPU();
    
    return {X, y_intent};
}

// ============================================================================
// TRAINING FUNCTION
// ============================================================================

void trainModel(const std::vector<Example>& train_data,
               const std::vector<Example>& val_data, 
               const Vocabulary& vocab) {
    
    std::cout << BOLD << CYAN << "Building GPU Neural Network..." << RESET << "\n";
    
    // Network configuration
    const size_t max_seq_len = 32;
    const size_t embedding_dim = 64;
    const size_t hidden_dim = 128;
    
    // Create CUDA neural network
    NeuralNetworkCUDA network;
    
    // Add embedding layer (simulate with dense layer)
    network.addLayer(new DenseLayerCUDA(max_seq_len, embedding_dim, new ReLUCUDA()));
    
    // Add hidden layers
    network.addLayer(new DenseLayerCUDA(embedding_dim, hidden_dim, new ReLUCUDA()));
    network.addLayer(new DenseLayerCUDA(hidden_dim, hidden_dim, new ReLUCUDA()));
    
    // Add intent classification head
    network.addLayer(new DenseLayerCUDA(hidden_dim, vocab.num_intents, nullptr));
    
    // Set loss function
    network.setLoss(new CategoricalCrossEntropyLossCUDA());
    
    // Set optimizer  
    network.setOptimizer(new SGD_CUDA(0.001));
    
    std::cout << GREEN << "✓ Network created on GPU" << RESET << "\n";
    network.summary();
    
    // Prepare data
    std::cout << BOLD << YELLOW << "Preparing training data..." << RESET << "\n";
    auto [X_train, y_train] = prepareTrainingData(train_data, vocab, max_seq_len);
    auto [X_val, y_val] = prepareTrainingData(val_data, vocab, max_seq_len);
    
    std::cout << GREEN << "✓ Data prepared and moved to GPU" << RESET << "\n";
    std::cout << "  Training input shape: " << X_train.getRows() << " × " << X_train.getCols() << "\n";
    std::cout << "  Validation input shape: " << X_val.getRows() << " × " << X_val.getCols() << "\n\n";
    
    // Training parameters
    const int epochs = 50;
    const int batch_size = 16;
    const bool verbose = true;
    
    std::cout << BOLD << CYAN << "Starting GPU Training..." << RESET << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Learning rate: 0.001\n\n";
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Train the network
    network.trainWithValidation(
        X_train, y_train,
        X_val, y_val, 
        epochs, batch_size, 
        0.001,  // learning rate
        verbose
    );
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n" << GREEN << BOLD << "✓ Training completed in " << duration.count() << " seconds!" << RESET << "\n\n";
    
    // Evaluate final performance
    std::cout << BOLD << YELLOW << "Final Evaluation:" << RESET << "\n";
    
    // Compute final accuracy on validation set
    MatrixCUDA val_predictions = network.forward(X_val);
    
    // Move predictions to CPU for evaluation
    val_predictions.toCPU();
    y_val.toCPU();
    
    int correct = 0;
    for (size_t i = 0; i < X_val.getRows(); i++) {
        // Find predicted class (argmax)
        int pred_class = 0;
        double max_prob = val_predictions.get(i, 0);
        for (size_t j = 1; j < vocab.num_intents; j++) {
            if (val_predictions.get(i, j) > max_prob) {
                max_prob = val_predictions.get(i, j);
                pred_class = j;
            }
        }
        
        // Find true class (argmax of one-hot)
        int true_class = 0;
        double max_true = y_val.get(i, 0);
        for (size_t j = 1; j < vocab.num_intents; j++) {
            if (y_val.get(i, j) > max_true) {
                max_true = y_val.get(i, j);
                true_class = j;
            }
        }
        
        if (pred_class == true_class) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / X_val.getRows() * 100.0;
    std::cout << "  Validation Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << "\n";
    std::cout << "  Correct predictions: " << correct << "/" << X_val.getRows() << "\n\n";
    
    // Show some example predictions
    std::cout << BOLD << BLUE << "Example Predictions:" << RESET << "\n";
    for (size_t i = 0; i < std::min((size_t)5, X_val.getRows()); i++) {
        int pred_class = 0;
        double max_prob = val_predictions.get(i, 0);
        for (size_t j = 1; j < vocab.num_intents; j++) {
            if (val_predictions.get(i, j) > max_prob) {
                max_prob = val_predictions.get(i, j);
                pred_class = j;
            }
        }
        
        // Find true class from one-hot
        int true_class = 0;
        for (size_t j = 0; j < vocab.num_intents; j++) {
            if (y_val.get(i, j) > 0.5) {
                true_class = j;
                break;
            }
        }
        
        std::cout << "  \"" << val_data[i].text << "\"\n";
        std::cout << "    True: " << vocab.id2intent.at(true_class);
        std::cout << " | Predicted: " << vocab.id2intent.at(pred_class);
        std::cout << " | Confidence: " << std::setprecision(3) << max_prob;
        std::cout << (pred_class == true_class ? GREEN " ✓" : RED " ✗") << RESET << "\n\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    std::cout << BOLD << CYAN;
    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          CUDA Intent Detection Training (ATIS)            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n" << RESET;
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << RED << "No CUDA devices found!" << RESET << "\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\nGPU Information:\n";
    std::cout << "  Device: " << prop.name << "\n";
    std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n\n";
    
    // Load ATIS dataset
    std::cout << BOLD << YELLOW << "Loading ATIS Dataset..." << RESET << "\n\n";
    
    auto train_data = loadDataset("data/train.json");
    auto val_data = loadDataset("data/validation.json");
    auto test_data = loadDataset("data/test.json");
    
    if (train_data.empty()) {
        std::cerr << RED << "No training data loaded! Make sure data files exist." << RESET << "\n";
        return 1;
    }
    
    printDatasetStats(train_data, "Training");
    printDatasetStats(val_data, "Validation");
    printDatasetStats(test_data, "Test");
    
    // Build vocabulary from training data
    Vocabulary vocab = buildVocabulary(train_data);
    printVocabStats(vocab);
    
    // Train the model
    try {
        trainModel(train_data, val_data, vocab);
        
        std::cout << GREEN << BOLD << "🎉 Training completed successfully!\n" << RESET;
        std::cout << "Model trained on " << train_data.size() << " examples using GPU acceleration.\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Training failed: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}