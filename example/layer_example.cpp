/**
 * @file layer_example.cpp
 * @brief Comprehensive demonstration of neural network layers
 * 
 * This example shows:
 * 1. Creating and initializing layers
 * 2. Forward and backward passes
 * 3. Complete training loop
 * 4. Multi-layer networks
 * 5. Visualization of learning
 * 6. Different activation functions
 */

#include "../include/nn/layer.h"
#include "../include/nn/activation.h"
#include "../include/nn/loss.h"
#include "../include/nn/matrix.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <memory>

// ANSI color codes
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════" << RESET << "\n\n";
}

void printSubHeader(const std::string& title) {
    std::cout << "\n" << BOLD << YELLOW << "─── " << title << " ───" << RESET << "\n\n";
}

// Example 1: Basic layer creation and properties
void example1_LayerBasics() {
    printHeader("EXAMPLE 1: Layer Basics");
    
    std::cout << "Creating a Dense (Fully Connected) Layer\n\n";
    
    // Create layer: 10 inputs → 5 outputs
    DenseLayer layer(10, 5, new ReLU());
    
    std::cout << "Layer Properties:\n";
    std::cout << "  Name: " << layer.getName() << "\n";
    std::cout << "  Input size: " << layer.getInputSize() << "\n";
    std::cout << "  Output size: " << layer.getOutputSize() << "\n";
    std::cout << "  Parameters: " << layer.getParameterCount() << "\n";
    std::cout << "    = (input × output) + output\n";
    std::cout << "    = (10 × 5) + 5 = 55\n\n";
    
    // Show weight dimensions
    Matrix weights = layer.getWeights();
    Matrix biases = layer.getBiases();
    
    std::cout << "Weight matrix shape: " << weights.getRows() << " × " << weights.getCols() << "\n";
    std::cout << "  (output_size × input_size) = (5 × 10)\n";
    std::cout << "Bias vector shape: " << biases.getRows() << " × " << biases.getCols() << "\n";
    std::cout << "  (output_size × 1) = (5 × 1)\n\n";
    
    // Show some weights
    std::cout << "First 3 weights of neuron 1:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "  W[0," << i << "] = " << std::fixed << std::setprecision(4) 
                  << weights.get(0, i) << "\n";
    }
    
    std::cout << "\n" << GREEN << "✓ Layer created and initialized successfully!" << RESET << "\n";
}

// Example 2: Weight initialization strategies
void example2_WeightInitialization() {
    printHeader("EXAMPLE 2: Weight Initialization Strategies");
    
    std::vector<std::string> strategies = {"xavier", "he", "random", "zeros"};
    
    for (const auto& strategy : strategies) {
        printSubHeader("Strategy: " + strategy);
        
        DenseLayer layer(100, 50);  // 100 inputs → 50 outputs
        
        if (strategy == "zeros") {
            std::cout << YELLOW << "⚠ WARNING: Zeros initialization causes symmetry problem!" 
                      << RESET << "\n";
        }
        
        layer.initializeWeights(strategy);
        Matrix weights = layer.getWeights();
        
        // Compute statistics
        double sum = 0, sum_sq = 0, min_val = 1e10, max_val = -1e10;
        int count = weights.getRows() * weights.getCols();
        
        for (size_t i = 0; i < weights.getRows(); i++) {
            for (size_t j = 0; j < weights.getCols(); j++) {
                double val = weights.get(i, j);
                sum += val;
                sum_sq += val * val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
        
        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        double std_dev = std::sqrt(variance);
        
        std::cout << "Statistics for " << count << " weights:\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(6) << mean << "\n";
        std::cout << "  Std Dev: " << std_dev << "\n";
        std::cout << "  Min: " << min_val << "\n";
        std::cout << "  Max: " << max_val << "\n";
        std::cout << "  Variance: " << variance << "\n";
        
        // Expected variance
        if (strategy == "xavier") {
            double expected_var = 2.0 / (100 + 50);
            std::cout << "  Expected variance (Xavier): " << expected_var << "\n";
        } else if (strategy == "he") {
            double expected_var = 2.0 / 100;
            std::cout << "  Expected variance (He): " << expected_var << "\n";
        }
    }
}

// Example 3: Forward pass demonstration
void example3_ForwardPass() {
    printHeader("EXAMPLE 3: Forward Pass");
    
    std::cout << "Computing output from input\n\n";
    
    // Create small layer for easy visualization
    DenseLayer layer(3, 2, new ReLU());
    
    // Set known weights and biases
    Matrix W(2, 3);
    W.set(0, 0, 0.5); W.set(0, 1, 0.3); W.set(0, 2, 0.2);
    W.set(1, 0, 0.4); W.set(1, 1, 0.6); W.set(1, 2, 0.1);
    
    Matrix b(2, 1);
    b.set(0, 0, 0.1);
    b.set(1, 0, 0.2);
    
    layer.setWeights(W);
    layer.setBiases(b);
    
    std::cout << "Layer configuration:\n";
    std::cout << "  3 inputs → 2 outputs (ReLU)\n\n";
    
    std::cout << "Weights:\n";
    std::cout << "  Neuron 1: [" << W.get(0,0) << ", " << W.get(0,1) << ", " << W.get(0,2) << "]\n";
    std::cout << "  Neuron 2: [" << W.get(1,0) << ", " << W.get(1,1) << ", " << W.get(1,2) << "]\n\n";
    
    std::cout << "Biases:\n";
    std::cout << "  Neuron 1: " << b.get(0,0) << "\n";
    std::cout << "  Neuron 2: " << b.get(1,0) << "\n\n";
    
    // Create input
    Matrix input(1, 3);  // 1 sample, 3 features
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    input.set(0, 2, 3.0);
    
    std::cout << "Input: [" << input.get(0,0) << ", " << input.get(0,1) << ", " << input.get(0,2) << "]\n\n";
    
    std::cout << BOLD << "Step-by-step computation:" << RESET << "\n\n";
    
    // Manual computation
    double z1 = input.get(0,0)*W.get(0,0) + input.get(0,1)*W.get(0,1) + input.get(0,2)*W.get(0,2) + b.get(0,0);
    double z2 = input.get(0,0)*W.get(1,0) + input.get(0,1)*W.get(1,1) + input.get(0,2)*W.get(1,2) + b.get(1,0);
    
    std::cout << "1. Linear transformation (Z = X·W^T + b):\n";
    std::cout << "   Neuron 1: z1 = (1.0×0.5 + 2.0×0.3 + 3.0×0.2) + 0.1\n";
    std::cout << "              = (0.5 + 0.6 + 0.6) + 0.1\n";
    std::cout << "              = " << z1 << "\n\n";
    
    std::cout << "   Neuron 2: z2 = (1.0×0.4 + 2.0×0.6 + 3.0×0.1) + 0.2\n";
    std::cout << "              = (0.4 + 1.2 + 0.3) + 0.2\n";
    std::cout << "              = " << z2 << "\n\n";
    
    std::cout << "2. Activation (A = ReLU(Z)):\n";
    std::cout << "   a1 = ReLU(" << z1 << ") = " << std::max(0.0, z1) << "\n";
    std::cout << "   a2 = ReLU(" << z2 << ") = " << std::max(0.0, z2) << "\n\n";
    
    // Actual forward pass
    Matrix output = layer.forward(input);
    
    std::cout << BOLD << "Layer output: " << RESET 
              << "[" << output.get(0,0) << ", " << output.get(0,1) << "]\n";
    
    std::cout << "\n" << GREEN << "✓ Forward pass complete!" << RESET << "\n";
}

// Example 4: Backward pass demonstration
void example4_BackwardPass() {
    printHeader("EXAMPLE 4: Backward Pass (Gradient Computation)");
    
    std::cout << "Computing gradients for learning\n\n";
    
    // Create and setup layer
    DenseLayer layer(2, 2, new ReLU());
    
    Matrix W(2, 2);
    W.set(0, 0, 0.5); W.set(0, 1, 0.3);
    W.set(1, 0, 0.4); W.set(1, 1, 0.6);
    
    Matrix b(2, 1);
    b.set(0, 0, 0.1);
    b.set(1, 0, 0.2);
    
    layer.setWeights(W);
    layer.setBiases(b);
    
    std::cout << "Layer: 2 inputs → 2 outputs (ReLU)\n\n";
    
    // Forward pass first
    Matrix input(1, 2);
    input.set(0, 0, 1.0);
    input.set(0, 1, 2.0);
    
    std::cout << "Input: [" << input.get(0,0) << ", " << input.get(0,1) << "]\n";
    
    Matrix output = layer.forward(input);
    std::cout << "Output: [" << output.get(0,0) << ", " << output.get(0,1) << "]\n\n";
    
    // Simulate gradient from next layer
    Matrix output_gradient(1, 2);
    output_gradient.set(0, 0, 0.1);
    output_gradient.set(0, 1, 0.2);
    
    std::cout << "Gradient from next layer: [" << output_gradient.get(0,0) 
              << ", " << output_gradient.get(0,1) << "]\n\n";
    
    std::cout << BOLD << "Backward pass computes:" << RESET << "\n";
    std::cout << "  1. Weight gradients (∂L/∂W)\n";
    std::cout << "  2. Bias gradients (∂L/∂b)\n";
    std::cout << "  3. Input gradients (∂L/∂X) for previous layer\n\n";
    
    // Backward pass
    Matrix input_gradient = layer.backward(output_gradient);
    
    Matrix dW = layer.getWeightGradients();
    Matrix db = layer.getBiasGradients();
    
    std::cout << "Weight gradients (∂L/∂W):\n";
    std::cout << "  Neuron 1: [" << dW.get(0,0) << ", " << dW.get(0,1) << "]\n";
    std::cout << "  Neuron 2: [" << dW.get(1,0) << ", " << dW.get(1,1) << "]\n\n";
    
    std::cout << "Bias gradients (∂L/∂b):\n";
    std::cout << "  [" << db.get(0,0) << ", " << db.get(1,0) << "]\n\n";
    
    std::cout << "Input gradients (∂L/∂X) passed to previous layer:\n";
    std::cout << "  [" << input_gradient.get(0,0) << ", " << input_gradient.get(0,1) << "]\n";
    
    std::cout << "\n" << GREEN << "✓ Backward pass complete!" << RESET << "\n";
}

// Example 5: Parameter update demonstration
void example5_ParameterUpdate() {
    printHeader("EXAMPLE 5: Parameter Update (Learning)");
    
    std::cout << "Demonstrating gradient descent update\n\n";
    
    // Create layer
    DenseLayer layer(2, 2, nullptr);  // Linear (no activation)
    
    // Set initial weights
    Matrix W_initial(2, 2);
    W_initial.set(0, 0, 1.0); W_initial.set(0, 1, 2.0);
    W_initial.set(1, 0, 3.0); W_initial.set(1, 1, 4.0);
    
    Matrix b_initial(2, 1);
    b_initial.set(0, 0, 0.5);
    b_initial.set(1, 0, 1.0);
    
    layer.setWeights(W_initial);
    layer.setBiases(b_initial);
    
    std::cout << "Initial weights:\n";
    std::cout << "  [[" << W_initial.get(0,0) << ", " << W_initial.get(0,1) << "],\n";
    std::cout << "   [" << W_initial.get(1,0) << ", " << W_initial.get(1,1) << "]]\n\n";
    
    std::cout << "Initial biases:\n";
    std::cout << "  [" << b_initial.get(0,0) << ", " << b_initial.get(1,0) << "]\n\n";
    
    // Forward and backward
    Matrix input(1, 2);
    input.set(0, 0, 1.0);
    input.set(0, 1, 1.0);
    
    layer.forward(input);
    
    Matrix gradient(1, 2);
    gradient.set(0, 0, 0.1);
    gradient.set(0, 1, 0.2);
    
    layer.backward(gradient);
    
    Matrix dW = layer.getWeightGradients();
    Matrix db = layer.getBiasGradients();
    
    std::cout << "Computed gradients:\n";
    std::cout << "  Weight gradients:\n";
    std::cout << "    [[" << dW.get(0,0) << ", " << dW.get(0,1) << "],\n";
    std::cout << "     [" << dW.get(1,0) << ", " << dW.get(1,1) << "]]\n\n";
    
    std::cout << "  Bias gradients:\n";
    std::cout << "    [" << db.get(0,0) << ", " << db.get(1,0) << "]\n\n";
    
    // Update parameters
    double learning_rate = 0.1;
    std::cout << BOLD << "Updating with learning rate = " << learning_rate << RESET << "\n\n";
    
    std::cout << "Formula: θ_new = θ_old - learning_rate × gradient\n\n";
    
    layer.updateParameters(learning_rate);
    
    Matrix W_updated = layer.getWeights();
    Matrix b_updated = layer.getBiases();
    
    std::cout << "Updated weights:\n";
    std::cout << "  [[" << W_updated.get(0,0) << ", " << W_updated.get(0,1) << "],\n";
    std::cout << "   [" << W_updated.get(1,0) << ", " << W_updated.get(1,1) << "]]\n\n";
    
    std::cout << "Updated biases:\n";
    std::cout << "  [" << b_updated.get(0,0) << ", " << b_updated.get(1,0) << "]\n\n";
    
    // Show change
    std::cout << "Changes:\n";
    std::cout << "  ΔW[0,0] = " << W_updated.get(0,0) - W_initial.get(0,0) << "\n";
    std::cout << "  Δb[0] = " << b_updated.get(0,0) - b_initial.get(0,0) << "\n";
    
    std::cout << "\n" << GREEN << "✓ Parameters updated!" << RESET << "\n";
}

// Example 6: Complete training loop
void example6_TrainingLoop() {
    printHeader("EXAMPLE 6: Complete Training Loop");
    
    std::cout << "Training a simple network on XOR problem\n\n";
    
    // XOR dataset
    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;
    
    // [0, 0] → 0
    Matrix x1(1, 2); x1.set(0, 0, 0); x1.set(0, 1, 0);
    Matrix y1(1, 1); y1.set(0, 0, 0);
    inputs.push_back(x1); targets.push_back(y1);
    
    // [0, 1] → 1
    Matrix x2(1, 2); x2.set(0, 0, 0); x2.set(0, 1, 1);
    Matrix y2(1, 1); y2.set(0, 0, 1);
    inputs.push_back(x2); targets.push_back(y2);
    
    // [1, 0] → 1
    Matrix x3(1, 2); x3.set(0, 0, 1); x3.set(0, 1, 0);
    Matrix y3(1, 1); y3.set(0, 0, 1);
    inputs.push_back(x3); targets.push_back(y3);
    
    // [1, 1] → 0
    Matrix x4(1, 2); x4.set(0, 0, 1); x4.set(0, 1, 1);
    Matrix y4(1, 1); y4.set(0, 0, 0);
    inputs.push_back(x4); targets.push_back(y4);
    
    std::cout << "XOR Dataset:\n";
    std::cout << "  [0, 0] → 0\n";
    std::cout << "  [0, 1] → 1\n";
    std::cout << "  [1, 0] → 1\n";
    std::cout << "  [1, 1] → 0\n\n";
    
    // Create network: 2 → 4 → 1
    std::cout << "Network architecture:\n";
    std::cout << "  Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)\n\n";
    
    DenseLayer hidden(2, 4, new ReLU());
    DenseLayer output(4, 1, new Sigmoid());
    
    hidden.initializeWeights("xavier");
    output.initializeWeights("xavier");
    
    // Training
    double learning_rate = 0.1;
    int epochs = 1000;
    
    std::cout << "Training parameters:\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Epochs: " << epochs << "\n\n";
    
    std::cout << "Training progress:\n";
    
    MSELoss loss_fn;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        
        // Train on each sample
        for (size_t i = 0; i < inputs.size(); i++) {
            // Forward pass
            Matrix h = hidden.forward(inputs[i]);
            Matrix pred = output.forward(h);
            
            // Compute loss
            double loss = loss_fn.calculate(pred, targets[i]);
            total_loss += loss;
            
            // Backward pass
            Matrix loss_grad = loss_fn.gradient(pred, targets[i]);
            Matrix output_grad = output.backward(loss_grad);
            Matrix hidden_grad = hidden.backward(output_grad);
            
            // Update parameters
            output.updateParameters(learning_rate);
            hidden.updateParameters(learning_rate);
        }
        
        total_loss /= inputs.size();
        
        // Print progress
        if (epoch % 100 == 0) {
            std::cout << "  Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << total_loss;
            
            if (total_loss < 0.01) {
                std::cout << GREEN << " ✓ Converged!" << RESET;
            }
            std::cout << "\n";
        }
    }
    
    std::cout << "\n" << BOLD << "Testing trained network:" << RESET << "\n\n";
    
    for (size_t i = 0; i < inputs.size(); i++) {
        Matrix h = hidden.forward(inputs[i]);
        Matrix pred = output.forward(h);
        
        double predicted = pred.get(0, 0);
        double actual = targets[i].get(0, 0);
        
        std::cout << "  Input: [" << inputs[i].get(0,0) << ", " << inputs[i].get(0,1) << "] → ";
        std::cout << "Predicted: " << std::fixed << std::setprecision(4) << predicted;
        std::cout << " | Target: " << actual;
        
        if (std::abs(predicted - actual) < 0.1) {
            std::cout << GREEN << " ✓" << RESET;
        } else {
            std::cout << RED << " ✗" << RESET;
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << GREEN << "✓ Training complete!" << RESET << "\n";
}

// Example 7: Different activation functions
void example7_ActivationComparison() {
    printHeader("EXAMPLE 7: Different Activation Functions");
    
    std::cout << "Comparing behavior with different activations\n\n";
    
    Matrix input(1, 3);
    input.set(0, 0, -1.0);
    input.set(0, 1, 0.5);
    input.set(0, 2, 2.0);
    
    std::cout << "Input: [" << input.get(0,0) << ", " << input.get(0,1) << ", " << input.get(0,2) << "]\n\n";
    
    std::vector<std::pair<std::string, Activation*>> activations = {
        {"Linear (none)", nullptr},
        {"ReLU", new ReLU()},
        {"Sigmoid", new Sigmoid()},
        {"Tanh", new Tanh()},
        {"LeakyReLU", new LeakyReLU(0.1)}
    };
    
    for (auto& [name, act] : activations) {
        DenseLayer layer(3, 3, act);
        
        // Set identity weights and zero bias for clear visualization
        Matrix W = Matrix::identity(3);
        Matrix b(3, 1);
        b.zeros();
        
        layer.setWeights(W);
        layer.setBiases(b);
        
        Matrix output = layer.forward(input);
        
        std::cout << std::setw(15) << name << ": [";
        for (size_t i = 0; i < 3; i++) {
            std::cout << std::fixed << std::setprecision(3) << output.get(0, i);
            if (i < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    std::cout << "\n" << BOLD << "Observations:" << RESET << "\n";
    std::cout << "• Linear: Output = Input (no transformation)\n";
    std::cout << "• ReLU: Negative values → 0\n";
    std::cout << "• Sigmoid: All values mapped to (0, 1)\n";
    std::cout << "• Tanh: All values mapped to (-1, 1)\n";
    std::cout << "• LeakyReLU: Negative values get small gradient\n";
}

int main() {
    std::cout << BOLD << GREEN << R"(
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           NEURAL NETWORK LAYER DEMONSTRATION                      ║
║                                                                   ║
║  Understanding Dense Layers and Training                         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
)" << RESET << "\n";
    
    try {
        example1_LayerBasics();
        example2_WeightInitialization();
        example3_ForwardPass();
        example4_BackwardPass();
        example5_ParameterUpdate();
        example6_TrainingLoop();
        example7_ActivationComparison();
        
        printHeader("SUMMARY");
        std::cout << GREEN << "✓" << RESET << " All examples completed successfully!\n\n";
        
        std::cout << BOLD << "What you learned:" << RESET << "\n";
        std::cout << "1. Layer creation and properties\n";
        std::cout << "2. Weight initialization strategies\n";
        std::cout << "3. Forward pass computation\n";
        std::cout << "4. Backward pass (gradients)\n";
        std::cout << "5. Parameter updates (learning)\n";
        std::cout << "6. Complete training loop\n";
        std::cout << "7. Different activation functions\n\n";
        
        std::cout << BOLD << "Key Concepts:" << RESET << "\n";
        std::cout << "• Forward: Input → Weights → Activation → Output\n";
        std::cout << "• Backward: Gradient → Update Weights → Learn\n";
        std::cout << "• Dense Layer: Every input connects to every output\n";
        std::cout << "• Parameters = (input_size × output_size) + output_size\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
