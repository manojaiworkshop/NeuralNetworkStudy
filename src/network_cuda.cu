/**
 * @file network_cuda.cu
 * @brief GPU-accelerated Neural Network implementation
 */

#include "nn/network_cuda.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// ============================================================================
// Constructor
// ============================================================================

NeuralNetworkCUDA::NeuralNetworkCUDA() : use_optimizer(false) {
    // Initialize empty network
    // Layers will be added via addLayer()
}

// ============================================================================
// Network Building Methods
// ============================================================================

void NeuralNetworkCUDA::addLayer(LayerCUDA* layer) {
    layers.push_back(std::unique_ptr<LayerCUDA>(layer));
}

void NeuralNetworkCUDA::setLoss(LossCUDA* loss) {
    loss_function = std::unique_ptr<LossCUDA>(loss);
}

void NeuralNetworkCUDA::setOptimizer(Optimizer_CUDA* opt) {
    optimizer = std::unique_ptr<Optimizer_CUDA>(opt);
    use_optimizer = true;
}

// ============================================================================
// Forward and Backward Propagation (GPU)
// ============================================================================

MatrixCUDA NeuralNetworkCUDA::forward(const MatrixCUDA& input) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    // Forward propagation through all layers
    // All computation happens on GPU
    MatrixCUDA output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetworkCUDA::backward(const MatrixCUDA& loss_gradient) {
    // Backpropagate through layers in reverse order
    // All computation happens on GPU
    MatrixCUDA gradient = loss_gradient;
    
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }
}

void NeuralNetworkCUDA::updateParameters(double learning_rate) {
    if (use_optimizer && optimizer) {
        // Use optimizer to update parameters (GPU)
        for (size_t i = 0; i < layers.size(); ++i) {
            DenseLayerCUDA* dense = dynamic_cast<DenseLayerCUDA*>(layers[i].get());
            if (dense) {
                // Update weights
                std::string weight_id = "layer" + std::to_string(i) + "_weights";
                MatrixCUDA new_weights = optimizer->update(
                    dense->getWeights(), 
                    dense->getWeightGradients(), 
                    weight_id
                );
                dense->setWeights(new_weights);
                
                // Update biases
                std::string bias_id = "layer" + std::to_string(i) + "_biases";
                MatrixCUDA new_biases = optimizer->update(
                    dense->getBiases(), 
                    dense->getBiasGradients(), 
                    bias_id
                );
                dense->setBiases(new_biases);
            }
        }
    } else {
        // Use simple gradient descent (GPU)
        for (auto& layer : layers) {
            layer->updateParameters(learning_rate);
        }
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

std::vector<std::pair<MatrixCUDA, MatrixCUDA>> NeuralNetworkCUDA::createBatches(
    const MatrixCUDA& X, const MatrixCUDA& y, int batch_size) {
    
    std::vector<std::pair<MatrixCUDA, MatrixCUDA>> batches;
    size_t num_samples = X.getRows();
    
    for (size_t start = 0; start < num_samples; start += batch_size) {
        size_t end = std::min(start + static_cast<size_t>(batch_size), num_samples);
        size_t current_batch_size = end - start;
        
        // Create batch matrices on GPU
        MatrixCUDA X_batch(current_batch_size, X.getCols());
        MatrixCUDA y_batch(current_batch_size, y.getCols());
        
        // Copy data (this happens on GPU using CUDA memcpy)
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

void NeuralNetworkCUDA::shuffleData(MatrixCUDA& X, MatrixCUDA& y) {
    // Transfer to CPU for shuffling (small overhead)
    X.toCPU();
    y.toCPU();
    
    size_t num_samples = X.getRows();
    std::vector<size_t> indices(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Create shuffled copies
    MatrixCUDA X_shuffled(X.getRows(), X.getCols());
    MatrixCUDA y_shuffled(y.getRows(), y.getCols());
    
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
    
    // Transfer back to GPU
    X.toGPU();
    y.toGPU();
}

// ============================================================================
// Training Methods
// ============================================================================

void NeuralNetworkCUDA::train(const MatrixCUDA& X_train, const MatrixCUDA& y_train, 
                              int epochs, int batch_size, 
                              double learning_rate, bool verbose) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    // Ensure data is on GPU
    MatrixCUDA X_gpu = X_train;
    MatrixCUDA y_gpu = y_train;
    X_gpu.toGPU();
    y_gpu.toGPU();
    
    size_t num_samples = X_gpu.getRows();
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;
    
    if (verbose) {
        std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║           GPU-ACCELERATED TRAINING STARTED            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
        std::cout << "Training Configuration:\n";
        std::cout << "  • Samples: " << num_samples << "\n";
        std::cout << "  • Batch size: " << batch_size << "\n";
        std::cout << "  • Batches per epoch: " << num_batches << "\n";
        std::cout << "  • Epochs: " << epochs << "\n";
        std::cout << "  • Learning rate: " << learning_rate << "\n";
        std::cout << "  • Optimizer: " << (use_optimizer ? "Custom" : "SGD") << "\n\n";
        std::cout << "Epoch | Loss      | Accuracy  | Time (ms)\n";
        std::cout << "------+-----------+-----------+----------\n";
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Shuffle data
        shuffleData(X_gpu, y_gpu);
        
        // Create batches
        auto batches = createBatches(X_gpu, y_gpu, batch_size);
        
        double epoch_loss = 0.0;
        
        // Train on each batch
        for (auto& [X_batch, y_batch] : batches) {
            // Forward pass (GPU)
            MatrixCUDA predictions = forward(X_batch);
            
            // Compute loss (GPU)
            double batch_loss = loss_function->calculateCUDA(predictions, y_batch);
            epoch_loss += batch_loss;
            
            // Backward pass (GPU)
            MatrixCUDA loss_grad = loss_function->gradientCUDA(predictions, y_batch);
            backward(loss_grad);
            
            // Update parameters (GPU)
            updateParameters(learning_rate);
        }
        
        epoch_loss /= num_batches;
        train_losses.push_back(epoch_loss);
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        // Calculate accuracy
        double acc = accuracy(X_gpu, y_gpu);
        train_accuracies.push_back(acc);
        
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6) << std::setw(9) << epoch_loss << " | "
                      << std::fixed << std::setprecision(2) << std::setw(8) << acc << "% | "
                      << std::setw(8) << duration.count() << "\n";
        }
    }
    
    if (verbose) {
        std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
        std::cout << "║           GPU TRAINING COMPLETED ✓                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
    }
}

void NeuralNetworkCUDA::trainWithValidation(const MatrixCUDA& X_train, const MatrixCUDA& y_train,
                                            const MatrixCUDA& X_val, const MatrixCUDA& y_val,
                                            int epochs, int batch_size,
                                            double learning_rate, bool verbose) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    // Ensure data is on GPU
    MatrixCUDA X_train_gpu = X_train;
    MatrixCUDA y_train_gpu = y_train;
    MatrixCUDA X_val_gpu = X_val;
    MatrixCUDA y_val_gpu = y_val;
    
    X_train_gpu.toGPU();
    y_train_gpu.toGPU();
    X_val_gpu.toGPU();
    y_val_gpu.toGPU();
    
    size_t num_samples = X_train_gpu.getRows();
    size_t num_batches = (num_samples + batch_size - 1) / batch_size;
    
    if (verbose) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║      GPU-ACCELERATED TRAINING WITH VALIDATION STARTED             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
        std::cout << "Epoch | Train Loss | Train Acc | Val Loss  | Val Acc  | Time (ms)\n";
        std::cout << "------+------------+-----------+-----------+----------+-----------\n";
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Training phase
        shuffleData(X_train_gpu, y_train_gpu);
        auto batches = createBatches(X_train_gpu, y_train_gpu, batch_size);
        
        double epoch_loss = 0.0;
        for (auto& [X_batch, y_batch] : batches) {
            MatrixCUDA predictions = forward(X_batch);
            double batch_loss = loss_function->calculateCUDA(predictions, y_batch);
            epoch_loss += batch_loss;
            
            MatrixCUDA loss_grad = loss_function->gradientCUDA(predictions, y_batch);
            backward(loss_grad);
            updateParameters(learning_rate);
        }
        
        epoch_loss /= num_batches;
        train_losses.push_back(epoch_loss);
        
        // Validation phase
        double val_loss = evaluate(X_val_gpu, y_val_gpu);
        val_losses.push_back(val_loss);
        
        double train_acc = accuracy(X_train_gpu, y_train_gpu);
        double val_acc = accuracy(X_val_gpu, y_val_gpu);
        train_accuracies.push_back(train_acc);
        val_accuracies.push_back(val_acc);
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6) << std::setw(10) << epoch_loss << " | "
                      << std::fixed << std::setprecision(2) << std::setw(9) << train_acc << "% | "
                      << std::fixed << std::setprecision(6) << std::setw(9) << val_loss << " | "
                      << std::fixed << std::setprecision(2) << std::setw(8) << val_acc << "% | "
                      << std::setw(9) << duration.count() << "\n";
        }
    }
    
    if (verbose) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         GPU TRAINING WITH VALIDATION COMPLETED ✓                   ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    }
}

// ============================================================================
// Prediction and Evaluation
// ============================================================================

MatrixCUDA NeuralNetworkCUDA::predict(const MatrixCUDA& input) {
    return forward(input);
}

double NeuralNetworkCUDA::evaluate(const MatrixCUDA& X_test, const MatrixCUDA& y_test) {
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    MatrixCUDA predictions = forward(X_test);
    return loss_function->calculateCUDA(predictions, y_test);
}

double NeuralNetworkCUDA::accuracy(const MatrixCUDA& X, const MatrixCUDA& y) {
    MatrixCUDA predictions = forward(X);
    
    // Transfer to CPU for accuracy calculation
    predictions.toCPU();
    MatrixCUDA y_cpu = y;
    y_cpu.toCPU();
    
    size_t correct = 0;
    size_t total = predictions.getRows();
    
    for (size_t i = 0; i < total; ++i) {
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
        double max_true = y_cpu.get(i, 0);
        for (size_t j = 1; j < y_cpu.getCols(); ++j) {
            if (y_cpu.get(i, j) > max_true) {
                max_true = y_cpu.get(i, j);
                true_class = j;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return (static_cast<double>(correct) / total) * 100.0;
}

// ============================================================================
// Utility Methods
// ============================================================================

LayerCUDA* NeuralNetworkCUDA::getLayer(size_t index) {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range");
    }
    return layers[index].get();
}

void NeuralNetworkCUDA::resetOptimizer() {
    if (optimizer) {
        optimizer->reset();
    }
}

void NeuralNetworkCUDA::summary() const {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                GPU NEURAL NETWORK SUMMARY                         ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Layer (type)                Output Shape         Param #     \n";
    std::cout << "==================================================================\n";
    
    size_t total_params = 0;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        size_t params = layer->getParameterCount();
        total_params += params;
        
        std::cout << std::setw(2) << i << ". " << std::setw(20) << std::left << layer->getName()
                  << " (" << layer->getInputSize() << " → " << layer->getOutputSize() << ")     "
                  << std::setw(12) << std::right << params << "\n";
    }
    
    std::cout << "==================================================================\n";
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "GPU Memory (approx): " << (total_params * 2 * sizeof(float)) / (1024.0 * 1024.0) 
              << " MB\n";
    std::cout << "  (weights + gradients)\n\n";
}

void NeuralNetworkCUDA::visualizeNetwork(bool show_weights, const std::string& filename) const {
    std::ostringstream oss;
    
    oss << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    oss << "║            GPU NEURAL NETWORK ARCHITECTURE                        ║\n";
    oss << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        
        oss << "Layer " << i << ": " << layer->getName() << "\n";
        oss << "  Input:  " << layer->getInputSize() << " neurons\n";
        oss << "  Output: " << layer->getOutputSize() << " neurons\n";
        oss << "  Params: " << layer->getParameterCount() << "\n";
        
        if (i < layers.size() - 1) {
            oss << "  ↓\n";
        }
    }
    
    oss << "\nTotal Layers: " << layers.size() << "\n";
    oss << "Total Parameters: " << getTotalParameters() << "\n";
    
    if (filename.empty()) {
        std::cout << oss.str();
    } else {
        std::ofstream file(filename);
        file << oss.str();
        std::cout << "Network visualization saved to " << filename << "\n";
    }
}

void NeuralNetworkCUDA::displayWeights(bool detailed) const {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  NETWORK WEIGHTS (FROM GPU)                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    for (size_t i = 0; i < layers.size(); ++i) {
        DenseLayerCUDA* dense = dynamic_cast<DenseLayerCUDA*>(layers[i].get());
        if (dense) {
            std::cout << "Layer " << i << " Weights:\n";
            
            MatrixCUDA weights = dense->getWeights();
            weights.toCPU();  // Transfer from GPU
            
            // Calculate statistics
            double sum = 0, min_val = weights.get(0,0), max_val = weights.get(0,0);
            for (size_t r = 0; r < weights.getRows(); ++r) {
                for (size_t c = 0; c < weights.getCols(); ++c) {
                    double val = weights.get(r, c);
                    sum += val;
                    min_val = std::min(min_val, val);
                    max_val = std::max(max_val, val);
                }
            }
            double mean = sum / (weights.getRows() * weights.getCols());
            
            std::cout << "  Shape: (" << weights.getRows() << " × " << weights.getCols() << ")\n";
            std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean << "\n";
            
            if (detailed && weights.getRows() <= 10 && weights.getCols() <= 10) {
                weights.print();
            }
            
            std::cout << "\n";
        }
    }
}

size_t NeuralNetworkCUDA::getTotalParameters() const {
    size_t total = 0;
    for (const auto& layer : layers) {
        total += layer->getParameterCount();
    }
    return total;
}

void NeuralNetworkCUDA::printGPUMemoryUsage() const {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    GPU MEMORY USAGE                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "Total GPU Memory: " << total_mem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Used: " << used_mem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Free: " << free_mem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Utilization: " << (used_mem * 100.0 / total_mem) << "%\n\n";
}

void NeuralNetworkCUDA::plotTrainingHistory() const {
    if (train_losses.empty()) {
        std::cout << "No training history available.\n";
        return;
    }
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  TRAINING HISTORY                                 ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";
    
    // ASCII plot of training loss
    std::cout << "Training Loss:\n";
    double max_loss = *std::max_element(train_losses.begin(), train_losses.end());
    double min_loss = *std::min_element(train_losses.begin(), train_losses.end());
    
    const int plot_height = 10;
    const int plot_width = 60;
    
    for (int h = plot_height; h >= 0; --h) {
        double threshold = min_loss + (max_loss - min_loss) * h / plot_height;
        std::cout << std::fixed << std::setprecision(4) << std::setw(8) << threshold << " │";
        
        for (size_t i = 0; i < std::min(train_losses.size(), (size_t)plot_width); ++i) {
            size_t idx = i * train_losses.size() / plot_width;
            if (std::abs(train_losses[idx] - threshold) < (max_loss - min_loss) / plot_height) {
                std::cout << "*";
            } else if (train_losses[idx] > threshold) {
                std::cout << " ";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }
    
    std::cout << "         └";
    for (int i = 0; i < plot_width; ++i) std::cout << "─";
    std::cout << ">\n";
    std::cout << "          0" << std::string(plot_width/2 - 5, ' ') << "Epoch" 
              << std::string(plot_width/2 - 5, ' ') << train_losses.size() << "\n\n";
    
    // Final metrics
    std::cout << "Final Training Loss: " << train_losses.back() << "\n";
    if (!train_accuracies.empty()) {
        std::cout << "Final Training Accuracy: " << train_accuracies.back() << "%\n";
    }
    if (!val_losses.empty()) {
        std::cout << "Final Validation Loss: " << val_losses.back() << "\n";
        std::cout << "Final Validation Accuracy: " << val_accuracies.back() << "%\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Save/Load Methods (Placeholder)
// ============================================================================

bool NeuralNetworkCUDA::saveModel(const std::string& filename) const {
    std::cout << "Model saving not yet implemented for CUDA networks.\n";
    std::cout << "Filename: " << filename << "\n";
    return false;
}

bool NeuralNetworkCUDA::loadModel(const std::string& filename) {
    std::cout << "Model loading not yet implemented for CUDA networks.\n";
    std::cout << "Filename: " << filename << "\n";
    return false;
}
