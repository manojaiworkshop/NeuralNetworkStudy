/**
 * @file network_complete_example.cpp
 * @brief Complete demonstration of NeuralNetwork class
 * 
 * This example shows:
 * 1. Building a network from scratch
 * 2. Training on XOR problem
 * 3. Making predictions
 * 4. Evaluating performance
 * 5. Network visualization
 * 6. Saving and loading models
 */

#include "../include/nn/network.h"
#include "../include/nn/layer.h"
#include "../include/nn/activation.h"
#include "../include/nn/loss.h"
#include "../include/nn/optimizer.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// ANSI colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(58) << std::left << title << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << RESET << "\n\n";
}

void printSubHeader(const std::string& title) {
    std::cout << "\n" << BOLD << YELLOW << "─── " << title << " ───" << RESET << "\n\n";
}

// Example 1: Building a simple network
void example1_BuildingNetwork() {
    printHeader("EXAMPLE 1: Building a Neural Network from Scratch");
    
    std::cout << "Step-by-step network construction:\n\n";
    
    // Step 1: Create network object
    std::cout << BOLD << "1. Create empty network container:" << RESET << "\n";
    NeuralNetwork network;
    std::cout << "   NeuralNetwork network;\n";
    std::cout << "   " << GREEN << "✓ Network object created" << RESET << "\n\n";
    
    // Step 2: Add layers
    std::cout << BOLD << "2. Add layers to build architecture:" << RESET << "\n\n";
    
    std::cout << "   Layer 1: Input → Hidden (2 → 4 neurons, ReLU)\n";
    network.addLayer(new DenseLayer(2, 4, new ReLU()));
    std::cout << "   network.addLayer(new DenseLayer(2, 4, new ReLU()));\n";
    std::cout << "   Parameters: (2×4) + 4 = 12\n";
    std::cout << "   " << GREEN << "✓ Layer 1 added" << RESET << "\n\n";
    
    std::cout << "   Layer 2: Hidden → Output (4 → 1 neuron, Sigmoid)\n";
    network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
    std::cout << "   network.addLayer(new DenseLayer(4, 1, new Sigmoid()));\n";
    std::cout << "   Parameters: (4×1) + 1 = 5\n";
    std::cout << "   " << GREEN << "✓ Layer 2 added" << RESET << "\n\n";
    
    // Step 3: Set loss function
    std::cout << BOLD << "3. Set loss function:" << RESET << "\n";
    network.setLoss(new MSELoss());
    std::cout << "   network.setLoss(new MSELoss());\n";
    std::cout << "   " << GREEN << "✓ Loss function set (Mean Squared Error)" << RESET << "\n\n";
    
    // Step 4: Set optimizer
    std::cout << BOLD << "4. Set optimizer (optional):" << RESET << "\n";
    network.setOptimizer(new SGD(0.1));
    std::cout << "   network.setOptimizer(new SGD(0.1));\n";
    std::cout << "   Learning rate: 0.1\n";
    std::cout << "   " << GREEN << "✓ Optimizer configured" << RESET << "\n\n";
    
    // Display summary
    std::cout << BOLD << "5. Network Summary:" << RESET << "\n\n";
    network.summary();
    
    // Visualize architecture
    std::cout << BOLD << "6. Architecture Visualization:" << RESET << "\n";
    network.visualizeNetwork(false);
}

// Example 2: Training on XOR problem
void example2_TrainingXOR() {
    printHeader("EXAMPLE 2: Training Neural Network (XOR Problem)");
    
    std::cout << "The XOR (exclusive OR) problem is a classic test for neural networks.\n";
    std::cout << "It cannot be solved by a single-layer network (linear classifier).\n\n";
    
    std::cout << BOLD << "XOR Truth Table:" << RESET << "\n";
    std::cout << "  ┌───┬───┬───────┐\n";
    std::cout << "  │ A │ B │ A XOR B│\n";
    std::cout << "  ├───┼───┼───────┤\n";
    std::cout << "  │ 0 │ 0 │   0   │\n";
    std::cout << "  │ 0 │ 1 │   1   │\n";
    std::cout << "  │ 1 │ 0 │   1   │\n";
    std::cout << "  │ 1 │ 1 │   0   │\n";
    std::cout << "  └───┴───┴───────┘\n\n";
    
    // Build network
    std::cout << BOLD << "Building network: 2 → 4 → 1" << RESET << "\n\n";
    NeuralNetwork network;
    network.addLayer(new DenseLayer(2, 4, new ReLU()));
    network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
    network.setLoss(new MSELoss());
    network.setOptimizer(new SGD(0.1));
    
    // Prepare data
    std::cout << BOLD << "Preparing training data:" << RESET << "\n\n";
    Matrix X_train(4, 2);
    X_train.set(0, 0, 0.0); X_train.set(0, 1, 0.0);
    X_train.set(1, 0, 0.0); X_train.set(1, 1, 1.0);
    X_train.set(2, 0, 1.0); X_train.set(2, 1, 0.0);
    X_train.set(3, 0, 1.0); X_train.set(3, 1, 1.0);
    
    Matrix y_train(4, 1);
    y_train.set(0, 0, 0.0);
    y_train.set(1, 0, 1.0);
    y_train.set(2, 0, 1.0);
    y_train.set(3, 0, 0.0);
    
    std::cout << "  Training samples: 4\n";
    std::cout << "  Input features: 2\n";
    std::cout << "  Output classes: 1 (binary)\n\n";
    
    // Train
    std::cout << BOLD << "Training network:" << RESET << "\n";
    std::cout << "  Epochs: 1000\n";
    std::cout << "  Batch size: 4 (full batch)\n";
    std::cout << "  Learning rate: 0.1\n\n";
    
    network.train(X_train, y_train, 1000, 4, 0.1, true);
    
    // Test predictions
    printSubHeader("Testing Trained Network");
    Matrix predictions = network.predict(X_train);
    
    std::cout << "Results:\n\n";
    std::cout << "  " << std::setw(10) << "Input" 
              << " │ " << std::setw(10) << "Target" 
              << " │ " << std::setw(10) << "Predicted" 
              << " │ " << "Result\n";
    std::cout << "  " << std::string(50, '─') << "\n";
    
    for (size_t i = 0; i < 4; i++) {
        std::cout << "  [" << X_train.get(i,0) << ", " << X_train.get(i,1) << "]   │ ";
        std::cout << "    " << y_train.get(i,0) << "     │ ";
        std::cout << "  " << std::fixed << std::setprecision(4) << predictions.get(i,0) << "   │ ";
        
        if (std::abs(predictions.get(i,0) - y_train.get(i,0)) < 0.1) {
            std::cout << GREEN << "  ✓ Correct" << RESET << "\n";
        } else {
            std::cout << RED << "  ✗ Wrong" << RESET << "\n";
        }
    }
    
    double final_loss = network.evaluate(X_train, y_train);
    std::cout << "\n  Final Loss: " << std::fixed << std::setprecision(6) << final_loss << "\n";
    
    if (final_loss < 0.01) {
        std::cout << "  " << GREEN << "✓ Network successfully learned XOR!" << RESET << "\n";
    }
}

// Example 3: Understanding forward and backward passes
void example3_ForwardBackward() {
    printHeader("EXAMPLE 3: Understanding Forward and Backward Passes");
    
    // Create simple network
    NeuralNetwork network;
    network.addLayer(new DenseLayer(2, 2, new ReLU()));
    network.addLayer(new DenseLayer(2, 1, new Sigmoid()));
    network.setLoss(new MSELoss());
    
    // Sample input
    Matrix input(1, 2);
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    
    Matrix target(1, 1);
    target.set(0, 0, 1.0);
    
    std::cout << BOLD << "Input:" << RESET << " [1.0, 2.0]\n";
    std::cout << BOLD << "Target:" << RESET << " [1.0]\n\n";
    
    // Forward pass
    printSubHeader("Forward Pass");
    std::cout << "Data flows through layers:\n\n";
    std::cout << "  Input [1.0, 2.0]\n";
    std::cout << "    ↓\n";
    std::cout << "  Layer 1: 2 → 2 (ReLU)\n";
    std::cout << "    Z₁ = X·W₁ᵀ + b₁\n";
    std::cout << "    A₁ = ReLU(Z₁)\n";
    std::cout << "    ↓\n";
    std::cout << "  Layer 2: 2 → 1 (Sigmoid)\n";
    std::cout << "    Z₂ = A₁·W₂ᵀ + b₂\n";
    std::cout << "    ŷ = Sigmoid(Z₂)\n";
    std::cout << "    ↓\n";
    
    Matrix prediction = network.forward(input);
    std::cout << "  Output: [" << std::fixed << std::setprecision(4) 
              << prediction.get(0,0) << "]\n\n";
    
    // Loss
    printSubHeader("Loss Calculation");
    double loss = network.evaluate(input, target);
    std::cout << "Loss = MSE(prediction, target)\n";
    std::cout << "     = (1.0 - " << prediction.get(0,0) << ")²\n";
    std::cout << "     = " << loss << "\n\n";
    
    // Backward pass
    printSubHeader("Backward Pass");
    std::cout << "Gradients flow backward:\n\n";
    std::cout << "  Loss gradient: ∂L/∂ŷ = -2(y - ŷ)\n";
    std::cout << "    ↓\n";
    std::cout << "  Layer 2 backward:\n";
    std::cout << "    Compute: ∂L/∂W₂, ∂L/∂b₂\n";
    std::cout << "    Pass back: ∂L/∂A₁\n";
    std::cout << "    ↓\n";
    std::cout << "  Layer 1 backward:\n";
    std::cout << "    Compute: ∂L/∂W₁, ∂L/∂b₁\n";
    std::cout << "    Pass back: ∂L/∂X\n";
    std::cout << "    ↓\n";
    std::cout << "  Gradients computed for all parameters!\n\n";
    
    // Show weight statistics before update
    std::cout << BOLD << "Before parameter update:" << RESET << "\n";
    network.displayWeights(false);
    
    // Update
    printSubHeader("Parameter Update");
    std::cout << "Applying gradient descent:\n";
    std::cout << "  W_new = W_old - learning_rate × ∂L/∂W\n";
    std::cout << "  b_new = b_old - learning_rate × ∂L/∂b\n\n";
    
    // Actually do backward and update
    MSELoss loss_fn;
    Matrix loss_grad = loss_fn.gradient(prediction, target);
    network.backward(loss_grad);
    network.updateParameters(0.1);
    
    std::cout << BOLD << "After parameter update:" << RESET << "\n";
    network.displayWeights(false);
    
    // Check if loss decreased
    Matrix new_prediction = network.forward(input);
    double new_loss = network.evaluate(input, target);
    
    std::cout << "\n" << BOLD << "Result:" << RESET << "\n";
    std::cout << "  Old loss: " << std::fixed << std::setprecision(6) << loss << "\n";
    std::cout << "  New loss: " << new_loss << "\n";
    
    if (new_loss < loss) {
        std::cout << "  " << GREEN << "✓ Loss decreased! Network is learning!" << RESET << "\n";
    } else {
        std::cout << "  " << YELLOW << "⚠ Loss increased (may happen with large learning rate)" << RESET << "\n";
    }
}

// Example 4: Batch training
void example4_BatchTraining() {
    printHeader("EXAMPLE 4: Understanding Mini-Batch Training");
    
    std::cout << "Mini-batch training splits data into small chunks for efficiency.\n\n";
    
    // Create larger dataset
    std::cout << BOLD << "Creating dataset: 100 samples" << RESET << "\n\n";
    
    Matrix X_train(100, 2);
    Matrix y_train(100, 1);
    
    // Simple pattern: y = (x₁ + x₂) > 1 ? 1 : 0
    for (size_t i = 0; i < 100; i++) {
        double x1 = (double)(rand() % 100) / 100.0;
        double x2 = (double)(rand() % 100) / 100.0;
        X_train.set(i, 0, x1);
        X_train.set(i, 1, x2);
        y_train.set(i, 0, (x1 + x2) > 1.0 ? 1.0 : 0.0);
    }
    
    // Build network
    NeuralNetwork network;
    network.addLayer(new DenseLayer(2, 8, new ReLU()));
    network.addLayer(new DenseLayer(8, 1, new Sigmoid()));
    network.setLoss(new BinaryCrossEntropyLoss());
    network.setOptimizer(new Adam(0.01));
    
    std::cout << BOLD << "Training with different batch sizes:" << RESET << "\n\n";
    
    // Compare batch sizes
    std::vector<int> batch_sizes = {1, 10, 32, 100};
    
    for (int batch_size : batch_sizes) {
        NeuralNetwork test_net;
        test_net.addLayer(new DenseLayer(2, 8, new ReLU()));
        test_net.addLayer(new DenseLayer(8, 1, new Sigmoid()));
        test_net.setLoss(new BinaryCrossEntropyLoss());
        test_net.setOptimizer(new Adam(0.01));
        
        std::cout << "  Batch size: " << std::setw(3) << batch_size;
        
        auto start = std::chrono::high_resolution_clock::now();
        test_net.train(X_train, y_train, 10, batch_size, 0.01, false);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double final_loss = test_net.evaluate(X_train, y_train);
        
        std::cout << " | Time: " << std::setw(4) << duration.count() << " ms";
        std::cout << " | Final Loss: " << std::fixed << std::setprecision(4) << final_loss << "\n";
    }
    
    std::cout << "\n" << BOLD << "Observations:" << RESET << "\n";
    std::cout << "  • Batch size = 1 (stochastic): Noisy, but fast updates\n";
    std::cout << "  • Batch size = 10-32 (mini-batch): Good balance\n";
    std::cout << "  • Batch size = 100 (full batch): Smooth, but slow\n\n";
    
    std::cout << "  " << GREEN << "✓ Mini-batch (32) is usually optimal!" << RESET << "\n";
}

// Example 5: Saving and loading models
void example5_SaveLoad() {
    printHeader("EXAMPLE 5: Saving and Loading Trained Models");
    
    // Train a model
    std::cout << BOLD << "1. Training a network:" << RESET << "\n\n";
    
    NeuralNetwork network;
    network.addLayer(new DenseLayer(2, 4, new ReLU()));
    network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
    network.setLoss(new MSELoss());
    
    Matrix X_train(4, 2);
    X_train.set(0, 0, 0.0); X_train.set(0, 1, 0.0);
    X_train.set(1, 0, 0.0); X_train.set(1, 1, 1.0);
    X_train.set(2, 0, 1.0); X_train.set(2, 1, 0.0);
    X_train.set(3, 0, 1.0); X_train.set(3, 1, 1.0);
    
    Matrix y_train(4, 1);
    y_train.set(0, 0, 0.0);
    y_train.set(1, 0, 1.0);
    y_train.set(2, 0, 1.0);
    y_train.set(3, 0, 0.0);
    
    network.train(X_train, y_train, 1000, 4, 0.1, false);
    
    Matrix pred_before = network.predict(X_train);
    std::cout << "  Prediction [0,0]: " << std::fixed << std::setprecision(4) 
              << pred_before.get(0,0) << "\n\n";
    
    // Save model
    std::cout << BOLD << "2. Saving model to file:" << RESET << "\n\n";
    bool saved = network.saveModel("xor_model.txt");
    
    if (saved) {
        std::cout << "\n" << BOLD << "3. Loading model from file:" << RESET << "\n\n";
        
        NeuralNetwork loaded_network;
        loaded_network.addLayer(new DenseLayer(2, 4, new ReLU()));
        loaded_network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
        loaded_network.setLoss(new MSELoss());
        
        bool loaded = loaded_network.loadModel("xor_model.txt");
        
        if (loaded) {
            Matrix pred_after = loaded_network.predict(X_train);
            std::cout << "\n" << BOLD << "4. Comparing predictions:" << RESET << "\n\n";
            
            std::cout << "  Original prediction [0,0]: " << pred_before.get(0,0) << "\n";
            std::cout << "  Loaded prediction [0,0]:   " << pred_after.get(0,0) << "\n\n";
            
            if (std::abs(pred_before.get(0,0) - pred_after.get(0,0)) < 1e-6) {
                std::cout << "  " << GREEN << "✓ Predictions match! Model saved/loaded correctly!" << RESET << "\n";
            }
        }
    }
}

int main() {
    std::cout << BOLD << GREEN << R"(
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║          NEURAL NETWORK CLASS - COMPLETE DEMONSTRATION             ║
║              Understanding How Networks Are Built                  ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
)" << RESET << "\n";
    
    try {
        example1_BuildingNetwork();
        example2_TrainingXOR();
        example3_ForwardBackward();
        example4_BatchTraining();
        example5_SaveLoad();
        
        printHeader("SUMMARY");
        
        std::cout << GREEN << "✓" << RESET << " All examples completed successfully!\n\n";
        
        std::cout << BOLD << "What you learned:" << RESET << "\n";
        std::cout << "  1. Building networks by adding layers\n";
        std::cout << "  2. Training networks with train() method\n";
        std::cout << "  3. Understanding forward/backward passes\n";
        std::cout << "  4. Mini-batch training strategies\n";
        std::cout << "  5. Saving and loading trained models\n\n";
        
        std::cout << BOLD << "Key Concepts:" << RESET << "\n";
        std::cout << "  • Network = container for layers + loss + optimizer\n";
        std::cout << "  • Forward pass = input → layers → output\n";
        std::cout << "  • Backward pass = loss → gradients → all layers\n";
        std::cout << "  • Training = repeated forward + backward + update\n";
        std::cout << "  • Mini-batches = efficient training on subsets\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
