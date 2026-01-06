#include "nn/network.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>

// Constructor
NeuralNetwork::NeuralNetwork() : use_optimizer(false) {}

// Add layer
void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(std::unique_ptr<Layer>(layer));
}

// Set loss function
void NeuralNetwork::setLoss(Loss* loss) {
    loss_function = std::unique_ptr<Loss>(loss);
}

// Set optimizer
void NeuralNetwork::setOptimizer(Optimizer* opt) {
    optimizer = std::unique_ptr<Optimizer>(opt);
    use_optimizer = true;
}

// Forward pass
Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    Matrix output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

// Backward pass
void NeuralNetwork::backward(const Matrix& loss_gradient) {
    Matrix gradient = loss_gradient;
    
    // Backpropagate through layers in reverse order
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }
}

// Update parameters
void NeuralNetwork::updateParameters(double learning_rate) {
    if (use_optimizer && optimizer) {
        // Use optimizer to update parameters
        for (size_t i = 0; i < layers.size(); ++i) {
            DenseLayer* dense = dynamic_cast<DenseLayer*>(layers[i].get());
            if (dense) {
                // Update weights
                std::string weight_id = "layer" + std::to_string(i) + "_weights";
                Matrix new_weights = optimizer->update(
                    dense->getWeights(), 
                    dense->getWeightGradients(), 
                    weight_id
                );
                dense->setWeights(new_weights);
                
                // Update biases
                std::string bias_id = "layer" + std::to_string(i) + "_biases";
                Matrix new_biases = optimizer->update(
                    dense->getBiases(), 
                    dense->getBiasGradients(), 
                    bias_id
                );
                dense->setBiases(new_biases);
                
                // Reset gradients
                dense->resetGradients();
            }
        }
    } else {
        // Use simple gradient descent
        for (auto& layer : layers) {
            layer->updateParameters(learning_rate);
        }
    }
}

// Create mini-batches
std::vector<std::pair<Matrix, Matrix>> NeuralNetwork::createBatches(
    const Matrix& X, const Matrix& y, int batch_size) {
    
    std::vector<std::pair<Matrix, Matrix>> batches;
    size_t num_samples = X.getRows();
    
    for (size_t start = 0; start < num_samples; start += batch_size) {
        size_t end = std::min(start + static_cast<size_t>(batch_size), num_samples);
        size_t current_batch_size = end - start;
        
        // Create batch matrices
        Matrix X_batch(current_batch_size, X.getCols());
        Matrix y_batch(current_batch_size, y.getCols());
        
        // Copy data
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < X.getCols(); ++j) {
                X_batch.set(i, j, X.get(start + i, j));
            }
            for (size_t j = 0; j < y.getCols(); ++j) {
                y_batch.set(i, j, y.get(start + i, j));
            }
        }
        
        batches.push_back({X_batch, y_batch});
    }
    
    return batches;
}

// Shuffle data
void NeuralNetwork::shuffleData(Matrix& X, Matrix& y) {
    size_t num_samples = X.getRows();
    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Create shuffled copies
    Matrix X_shuffled(X.getRows(), X.getCols());
    Matrix y_shuffled(y.getRows(), y.getCols());
    
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < X.getCols(); ++j) {
            X_shuffled.set(i, j, X.get(indices[i], j));
        }
        for (size_t j = 0; j < y.getCols(); ++j) {
            y_shuffled.set(i, j, y.get(indices[i], j));
        }
    }
    
    X = X_shuffled;
    y = y_shuffled;
}

// Train
void NeuralNetwork::train(const Matrix& X_train, const Matrix& y_train, 
                         int epochs, int batch_size, 
                         double learning_rate, bool verbose) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data each epoch
        Matrix X_shuffled = X_train;
        Matrix y_shuffled = y_train;
        shuffleData(X_shuffled, y_shuffled);
        
        // Create batches
        auto batches = createBatches(X_shuffled, y_shuffled, batch_size);
        
        double total_loss = 0.0;
        
        // Train on each batch
        for (const auto& batch : batches) {
            const Matrix& X_batch = batch.first;
            const Matrix& y_batch = batch.second;
            
            // Forward pass
            Matrix predictions = forward(X_batch);
            
            // Compute loss
            double batch_loss = loss_function->calculate(predictions, y_batch);
            total_loss += batch_loss * X_batch.getRows();
            
            // Backward pass
            Matrix loss_grad = loss_function->gradient(predictions, y_batch);
            backward(loss_grad);
            
            // Update parameters
            updateParameters(learning_rate);
        }
        
        // Average loss
        double avg_loss = total_loss / X_train.getRows();
        
        // Print progress
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << epochs 
                     << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                     << std::endl;
        }
    }
}

// Train with validation
void NeuralNetwork::trainWithValidation(const Matrix& X_train, const Matrix& y_train,
                                       const Matrix& X_val, const Matrix& y_val,
                                       int epochs, int batch_size,
                                       double learning_rate, bool verbose) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data
        Matrix X_shuffled = X_train;
        Matrix y_shuffled = y_train;
        shuffleData(X_shuffled, y_shuffled);
        
        // Create batches
        auto batches = createBatches(X_shuffled, y_shuffled, batch_size);
        
        double total_train_loss = 0.0;
        
        // Train on each batch
        for (const auto& batch : batches) {
            const Matrix& X_batch = batch.first;
            const Matrix& y_batch = batch.second;
            
            // Forward pass
            Matrix predictions = forward(X_batch);
            
            // Compute loss
            double batch_loss = loss_function->calculate(predictions, y_batch);
            total_train_loss += batch_loss * X_batch.getRows();
            
            // Backward pass
            Matrix loss_grad = loss_function->gradient(predictions, y_batch);
            backward(loss_grad);
            
            // Update parameters
            updateParameters(learning_rate);
        }
        
        // Average training loss
        double avg_train_loss = total_train_loss / X_train.getRows();
        
        // Compute validation loss
        double val_loss = evaluate(X_val, y_val);
        
        // Print progress
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << epochs 
                     << " - Train Loss: " << std::fixed << std::setprecision(6) << avg_train_loss
                     << " - Val Loss: " << val_loss
                     << std::endl;
        }
    }
}

// Predict
Matrix NeuralNetwork::predict(const Matrix& input) {
    return forward(input);
}

// Evaluate
double NeuralNetwork::evaluate(const Matrix& X_test, const Matrix& y_test) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    Matrix predictions = forward(X_test);
    return loss_function->calculate(predictions, y_test);
}

// Calculate accuracy
double NeuralNetwork::accuracy(const Matrix& X, const Matrix& y) {
    Matrix predictions = forward(X);
    
    int correct = 0;
    for (size_t i = 0; i < X.getRows(); ++i) {
        // Find predicted class (argmax)
        size_t pred_class = 0;
        double max_pred = predictions.get(i, 0);
        for (size_t j = 1; j < predictions.getCols(); ++j) {
            if (predictions.get(i, j) > max_pred) {
                max_pred = predictions.get(i, j);
                pred_class = j;
            }
        }
        
        // Find true class (argmax)
        size_t true_class = 0;
        double max_true = y.get(i, 0);
        for (size_t j = 1; j < y.getCols(); ++j) {
            if (y.get(i, j) > max_true) {
                max_true = y.get(i, j);
                true_class = j;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / X.getRows() * 100.0;
}

// Get layer
Layer* NeuralNetwork::getLayer(size_t index) {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers[index].get();
}

// Print summary
void NeuralNetwork::summary() const {
    std::cout << "\n========== Neural Network Summary ==========\n";
    std::cout << "Total Layers: " << layers.size() << "\n";
    std::cout << "Optimizer: " << (optimizer ? optimizer->getName() : "None (Gradient Descent)") << "\n";
    std::cout << "Loss Function: " << (loss_function ? loss_function->getName() : "Not set") << "\n";
    std::cout << "\n--- Layer Details ---\n";
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        std::cout << "Layer " << i + 1 << ": " << layer->getName()
                 << " (" << layer->getInputSize() << " -> " << layer->getOutputSize() << ")\n";
    }
    
    std::cout << "============================================\n\n";
}

// Reset optimizer
void NeuralNetwork::resetOptimizer() {
    if (optimizer) {
        optimizer->reset();
    }
}

// Visualize network architecture
void NeuralNetwork::visualizeNetwork(bool show_weights, const std::string& filename) const {
    std::ostream* out = &std::cout;
    std::ofstream file;
    
    if (!filename.empty()) {
        file.open(filename);
        if (file.is_open()) {
            out = &file;
        }
    }
    
    *out << "\n╔══════════════════════════════════════════════════════════════╗\n";
    *out << "║          NEURAL NETWORK ARCHITECTURE VISUALIZATION          ║\n";
    *out << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    *out << "Network Configuration:\n";
    *out << "  • Total Layers: " << layers.size() << "\n";
    *out << "  • Optimizer: " << (optimizer ? optimizer->getName() : "Gradient Descent") << "\n";
    *out << "  • Loss Function: " << (loss_function ? loss_function->getName() : "Not set") << "\n\n";
    
    // Calculate total parameters
    int total_params = 0;
    for (const auto& layer : layers) {
        total_params += layer->getParameterCount();
    }
    *out << "  • Total Parameters: " << total_params << "\n\n";
    
    *out << "┌────────────────────────────────────────────────────────────┐\n";
    *out << "│                    NETWORK STRUCTURE                       │\n";
    *out << "└────────────────────────────────────────────────────────────┘\n\n";
    
    // Draw input layer
    *out << "     INPUT LAYER\n";
    *out << "         │\n";
    *out << "         │ (" << (layers.empty() ? 0 : layers[0]->getInputSize()) << " features)\n";
    *out << "         ▼\n";
    
    // Draw each layer
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        
        *out << "   ┌─────────────┐\n";
        *out << "   │  LAYER " << std::setw(2) << (i + 1) << "  │\n";
        *out << "   └─────────────┘\n";
        *out << "   " << layer->getName() << "\n";
        *out << "   " << layer->getInputSize() << " → " << layer->getOutputSize() << " neurons\n";
        *out << "   Params: " << layer->getParameterCount() << "\n";
        
        if (show_weights) {
            // Get weight statistics
            const Matrix& weights = layer->getWeights();
            const Matrix& biases = layer->getBiases();
            
            double w_min = 1e9, w_max = -1e9, w_sum = 0;
            int w_count = 0;
            
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    double val = weights.get(r, c);
                    w_min = std::min(w_min, val);
                    w_max = std::max(w_max, val);
                    w_sum += val;
                    w_count++;
                }
            }
            
            double w_mean = w_sum / w_count;
            
            // Calculate std dev
            double variance = 0;
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    double diff = weights.get(r, c) - w_mean;
                    variance += diff * diff;
                }
            }
            double w_std = std::sqrt(variance / w_count);
            
            *out << std::fixed << std::setprecision(4);
            *out << "   Weights: min=" << w_min << ", max=" << w_max 
                 << ", mean=" << w_mean << ", std=" << w_std << "\n";
            
            // Bias statistics
            double b_sum = 0;
            for (size_t j = 0; j < biases.getCols(); ++j) {
                b_sum += biases.get(0, j);
            }
            double b_mean = b_sum / biases.getCols();
            *out << "   Biases: mean=" << b_mean << "\n";
        }
        
        if (i < layers.size() - 1) {
            *out << "         │\n";
            *out << "         ▼\n";
        }
    }
    
    // Output layer
    *out << "\n     OUTPUT LAYER\n";
    *out << "   (" << (layers.empty() ? 0 : layers.back()->getOutputSize()) << " outputs)\n\n";
    
    *out << "═══════════════════════════════════════════════════════════════\n\n";
    
    if (file.is_open()) {
        file.close();
        std::cout << "Network visualization saved to: " << filename << "\n";
    }
}

// Display trained weights
void NeuralNetwork::displayWeights(bool detailed) const {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    TRAINED WEIGHTS DISPLAY                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        
        std::cout << "┌─ Layer " << (i + 1) << ": " << layer->getName() << " ─┐\n";
        
        const Matrix& weights = layer->getWeights();
        const Matrix& biases = layer->getBiases();
        
        std::cout << "  Shape: (" << weights.getRows() << " × " << weights.getCols() << ")\n";
        
        if (detailed && weights.getRows() * weights.getCols() <= 100) {
            std::cout << "\n  Weight Matrix:\n";
            std::cout << std::fixed << std::setprecision(4);
            
            for (size_t r = 0; r < weights.getRows(); ++r) {
                std::cout << "  [ ";
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    std::cout << std::setw(8) << weights.get(r, c) << " ";
                }
                std::cout << "]\n";
            }
            
            std::cout << "\n  Bias Vector:\n  [ ";
            for (size_t r = 0; r < biases.getRows(); ++r) {
                std::cout << std::setw(8) << biases.get(r, 0) << " ";
            }
            std::cout << "]\n";
        } else {
            // Just show statistics
            double w_min = 1e9, w_max = -1e9, w_sum = 0;
            int w_count = 0;
            
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    double val = weights.get(r, c);
                    w_min = std::min(w_min, val);
                    w_max = std::max(w_max, val);
                    w_sum += val;
                    w_count++;
                }
            }
            
            double w_mean = w_sum / w_count;
            double variance = 0;
            
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    double diff = weights.get(r, c) - w_mean;
                    variance += diff * diff;
                }
            }
            double w_std = std::sqrt(variance / w_count);
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  Weight Statistics:\n";
            std::cout << "    Min:    " << w_min << "\n";
            std::cout << "    Max:    " << w_max << "\n";
            std::cout << "    Mean:   " << w_mean << "\n";
            std::cout << "    Std:    " << w_std << "\n";
            
            // Sample some weights
            if (weights.getRows() > 0 && weights.getCols() > 0) {
                std::cout << "  Sample weights: [ ";
                int samples = std::min(5, static_cast<int>(weights.getCols()));
                for (int s = 0; s < samples; ++s) {
                    std::cout << weights.get(0, s) << " ";
                }
                if (weights.getCols() > 5) std::cout << "...";
                std::cout << "]\n";
            }
        }
        
        std::cout << "\n";
    }
    
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
}

// Save model to file
bool NeuralNetwork::saveModel(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return false;
    }
    
    try {
        // Write header
        file << "NNMODEL\n";
        file << "VERSION 1.0\n";
        
        // Write network metadata
        file << "LAYERS " << layers.size() << "\n";
        
        // Write each layer
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            
            file << "LAYER " << i << "\n";
            file << "NAME " << layer->getName() << "\n";
            file << "INPUT_SIZE " << layer->getInputSize() << "\n";
            file << "OUTPUT_SIZE " << layer->getOutputSize() << "\n";
            
            // Save weights
            const Matrix& weights = layer->getWeights();
            file << "WEIGHTS " << weights.getRows() << " " << weights.getCols() << "\n";
            
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    file << weights.get(r, c);
                    if (c < weights.getCols() - 1) file << " ";
                }
                file << "\n";
            }
            
            // Save biases
            const Matrix& biases = layer->getBiases();
            file << "BIASES " << biases.getRows() << "\n";
            for (size_t r = 0; r < biases.getRows(); ++r) {
                file << biases.get(r, 0);
                if (r < biases.getRows() - 1) file << " ";
            }
            file << "\n";
        }
        
        file << "END\n";
        file.close();
        
        std::cout << "✓ Model saved successfully to: " << filename << "\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << "\n";
        file.close();
        return false;
    }
}

// Load model from file
bool NeuralNetwork::loadModel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << "\n";
        return false;
    }
    
    try {
        std::string line;
        
        // Read and verify header
        std::getline(file, line);
        if (line != "NNMODEL") {
            std::cerr << "Error: Invalid model file format\n";
            return false;
        }
        
        std::getline(file, line); // VERSION
        
        // Read number of layers
        std::getline(file, line);
        std::istringstream iss(line);
        std::string token;
        int num_layers;
        iss >> token >> num_layers; // "LAYERS" num_layers
        
        // Clear existing layers
        layers.clear();
        
        // Read each layer
        for (int i = 0; i < num_layers; ++i) {
            std::getline(file, line); // "LAYER i"
            
            std::getline(file, line); // "NAME ..."
            std::string layer_name = line.substr(5); // Skip "NAME "
            
            std::getline(file, line); // "INPUT_SIZE n"
            iss.clear();
            iss.str(line);
            size_t input_size;
            iss >> token >> input_size;
            
            std::getline(file, line); // "OUTPUT_SIZE n"
            iss.clear();
            iss.str(line);
            size_t output_size;
            iss >> token >> output_size;
            
            // Create layer based on name
            Activation* act = nullptr;
            if (layer_name.find("ReLU") != std::string::npos) {
                act = new ReLU();
            } else if (layer_name.find("Sigmoid") != std::string::npos) {
                act = new Sigmoid();
            } else if (layer_name.find("Tanh") != std::string::npos) {
                act = new Tanh();
            } else {
                act = new Linear();
            }
            
            DenseLayer* layer = new DenseLayer(input_size, output_size, act);
            
            // Read weights
            std::getline(file, line); // "WEIGHTS rows cols"
            iss.clear();
            iss.str(line);
            size_t w_rows, w_cols;
            iss >> token >> w_rows >> w_cols;
            
            Matrix weights(w_rows, w_cols);
            for (size_t r = 0; r < w_rows; ++r) {
                std::getline(file, line);
                iss.clear();
                iss.str(line);
                for (size_t c = 0; c < w_cols; ++c) {
                    double val;
                    iss >> val;
                    weights.set(r, c, val);
                }
            }
            
            // Read biases
            std::getline(file, line); // "BIASES n"
            iss.clear();
            iss.str(line);
            size_t b_size;
            iss >> token >> b_size;
            
            std::getline(file, line);
            iss.clear();
            iss.str(line);
            Matrix biases(b_size, 1);
            for (size_t r = 0; r < b_size; ++r) {
                double val;
                iss >> val;
                biases.set(r, 0, val);
            }
            
            // Set weights and biases
            layer->setWeights(weights);
            layer->setBiases(biases);
            
            // Add layer to network
            addLayer(layer);
        }
        
        file.close();
        
        std::cout << "✓ Model loaded successfully from: " << filename << "\n";
        std::cout << "  Loaded " << layers.size() << " layers\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        file.close();
        return false;
    }
}

// Export to DOT format for GraphViz
void NeuralNetwork::exportToDot(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return;
    }
    
    file << "digraph NeuralNetwork {\n";
    file << "  rankdir=LR;\n";
    file << "  node [shape=circle, style=filled, fontsize=10];\n";
    file << "  edge [fontsize=9, decorate=false];\n";
    file << "  graph [splines=curved, nodesep=1.0, ranksep=3.0];\n\n";
    
    // Input layer
    file << "  subgraph cluster_input {\n";
    file << "    label=\"Input Layer\";\n";
    file << "    style=filled;\n";
    file << "    color=lightgrey;\n";
    file << "    node [fillcolor=\"#ADD8E6\"];\n";
    
    int input_size = layers.empty() ? 0 : layers[0]->getInputSize();
    int show_inputs = std::min(input_size, 10);
    
    for (int i = 0; i < show_inputs; ++i) {
        file << "    input" << i << " [label=\"x" << i << "\"];\n";
    }
    if (input_size > 10) {
        file << "    input_more [label=\"...\", shape=plaintext];\n";
    }
    file << "  }\n\n";
    
    // Hidden and output layers
    for (size_t l = 0; l < layers.size(); ++l) {
        const auto& layer = layers[l];
        bool isOutputLayer = (l == layers.size() - 1);
        
        file << "  subgraph cluster_layer" << l << " {\n";
        
        // Label differently for output layer
        if (isOutputLayer) {
            file << "    label=\"Output Layer\\n";
        } else {
            file << "    label=\"" << layer->getName() << " Layer\\n";
        }
        
        // Add layer info
        file << "(" << layer->getInputSize() << "→" << layer->getOutputSize() << ")";
        file << "\\nParams: " << layer->getParameterCount() << "\";\n";
        file << "    style=filled;\n";
        
        if (isOutputLayer) {
            file << "    color=\"#90EE90\";\n";
            file << "    node [fillcolor=\"#98FB98\"];\n";
        } else {
            file << "    color=\"#FFFACD\";\n";
            file << "    node [fillcolor=\"#FFFFE0\"];\n";
        }
        
        int neurons = std::min(static_cast<int>(layer->getOutputSize()), 10);
        for (int i = 0; i < neurons; ++i) {
            // Add bias value to neuron label
            const Matrix& biases = layer->getBiases();
            if (biases.getRows() > 0 && i < biases.getRows()) {
                file << "    layer" << l << "_" << i 
                     << " [label=\"b=" << std::fixed << std::setprecision(2) 
                     << biases.get(i, 0) << "\"];\n";
            } else {
                file << "    layer" << l << "_" << i << " [label=\"\"];\n";
            }
        }
        if (layer->getOutputSize() > 10) {
            file << "    layer" << l << "_more [label=\"...\", shape=plaintext];\n";
        }
        file << "  }\n\n";
    }
    
    // Connections with ALL weights shown
    // Input to first layer
    if (!layers.empty()) {
        const Matrix& weights = layers[0]->getWeights();
        int input_nodes = std::min(input_size, 10);
        int first_layer_nodes = std::min(static_cast<int>(layers[0]->getOutputSize()), 10);
        
        // Show ALL connections
        for (int i = 0; i < input_nodes; ++i) {
            for (int j = 0; j < first_layer_nodes; ++j) {
                double weight = weights.get(j, i);
                
                // Color edge based on weight magnitude
                std::string color = "black";
                double penwidth = 1.0;
                
                if (std::abs(weight) > 1.0) {
                    color = "red";
                    penwidth = 2.0;
                } else if (std::abs(weight) > 0.5) {
                    color = "orange";
                    penwidth = 1.5;
                } else if (std::abs(weight) < 0.1) {
                    color = "gray";
                    penwidth = 0.5;
                }
                
                // Use curved splines and better label positioning
                file << "  input" << i << " -> layer0_" << j 
                     << " [label=<  <FONT POINT-SIZE=\"9\">" << std::fixed << std::setprecision(2) << weight << "</FONT>  >"
                     << ", color=\"" << color << "\""
                     << ", penwidth=" << penwidth 
                     << ", headlabel=\"\""
                     << ", taillabel=\"\"];\n";
            }
        }
    }
    
    // Between layers - show ALL connections
    for (size_t l = 0; l < layers.size() - 1; ++l) {
        const Matrix& weights = layers[l + 1]->getWeights();
        int curr_nodes = std::min(static_cast<int>(layers[l]->getOutputSize()), 10);
        int next_nodes = std::min(static_cast<int>(layers[l + 1]->getOutputSize()), 10);
        
        // Show ALL connections with weight labels
        for (int i = 0; i < curr_nodes; ++i) {
            for (int j = 0; j < next_nodes; ++j) {
                double weight = weights.get(j, i);
                
                // Color edge based on weight magnitude
                std::string color = "black";
                double penwidth = 1.0;
                
                if (std::abs(weight) > 1.0) {
                    color = "red";
                    penwidth = 2.0;
                } else if (std::abs(weight) > 0.5) {
                    color = "orange";
                    penwidth = 1.5;
                } else if (std::abs(weight) < 0.1) {
                    color = "gray";
                    penwidth = 0.5;
                }
                
                file << "  layer" << l << "_" << i << " -> layer" << (l + 1) << "_" << j 
                     << " [label=<  <FONT POINT-SIZE=\"9\">" << std::fixed << std::setprecision(2) << weight << "</FONT>  >"
                     << ", color=\"" << color << "\""
                     << ", penwidth=" << penwidth
                     << ", headlabel=\"\""
                     << ", taillabel=\"\"];\n";
            }
        }
    }
    
    // Add legend
    file << "\n  // Legend\n";
    file << "  subgraph cluster_legend {\n";
    file << "    label=\"Weight Legend\";\n";
    file << "    style=filled;\n";
    file << "    color=white;\n";
    file << "    node [shape=plaintext];\n";
    file << "    legend [label=<\n";
    file << "      <TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\" CELLPADDING=\"4\">\n";
    file << "        <TR><TD><FONT COLOR=\"red\">Red</FONT></TD><TD>|w| &gt; 1.0</TD></TR>\n";
    file << "        <TR><TD><FONT COLOR=\"orange\">Orange</FONT></TD><TD>|w| &gt; 0.5</TD></TR>\n";
    file << "        <TR><TD>Black</TD><TD>0.1 ≤ |w| ≤ 0.5</TD></TR>\n";
    file << "        <TR><TD><FONT COLOR=\"gray\">Gray</FONT></TD><TD>|w| &lt; 0.1</TD></TR>\n";
    file << "      </TABLE>\n";
    file << "    >];\n";
    file << "  }\n";
    
    file << "}\n";
    file.close();
    
    std::cout << "✓ Network graph exported to: " << filename << "\n";
    std::cout << "  Generate PNG: dot -Tpng -Gdpi=150 " << filename << " -o network.png\n";
    std::cout << "  High quality: dot -Tpng -Gdpi=300 " << filename << " -o network_hq.png\n";
    std::cout << "  Cleaner layout: sfdp -Tpng -Goverlap=prism " << filename << " -o network_sfdp.png\n";
    std::cout << "  ALL weights are shown on edges with color-coding:\n";
    std::cout << "    • Red: Strong weights (|w| > 1.0)\n";
    std::cout << "    • Orange: Medium weights (|w| > 0.5)\n";
    std::cout << "    • Black: Normal weights (0.1 ≤ |w| ≤ 0.5)\n";
    std::cout << "    • Gray: Weak weights (|w| < 0.1)\n";
}
