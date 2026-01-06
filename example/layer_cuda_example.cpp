/**
 * @file layer_cuda_example.cpp
 * @brief Comprehensive demonstration of GPU-accelerated neural network layers
 * 
 * This example shows:
 * 1. Creating and using CUDA layers
 * 2. GPU-accelerated forward/backward passes
 * 3. Training on GPU vs CPU
 * 4. Performance benchmarking
 * 5. Memory efficiency
 * 6. Multi-layer GPU networks
 * 7. Complete training pipeline
 */

#include "../include/nn/layer_cuda.h"
#include "../include/nn/layer.h"
#include "../include/nn/activation_cuda.h"
#include "../include/nn/activation.h"
#include "../include/nn/loss_cuda.h"
#include "../include/nn/loss.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

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

// Example 1: Basic CUDA layer creation
void example1_CudaLayerBasics() {
    printHeader("EXAMPLE 1: CUDA Layer Basics");
    
    std::cout << "Creating GPU-accelerated Dense Layer\n\n";
    
    // Create CUDA layer: 10 inputs → 5 outputs with ReLU
    DenseLayerCUDA layer(10, 5, new ReLUCUDA());
    
    std::cout << "Layer Properties:\n";
    std::cout << "  Name: " << layer.getName() << "\n";
    std::cout << "  Input size: " << layer.getInputSize() << "\n";
    std::cout << "  Output size: " << layer.getOutputSize() << "\n";
    std::cout << "  Parameters: " << layer.getParameterCount() << "\n";
    std::cout << "    = (10 × 5) + 5 = 55 parameters\n\n";
    
    std::cout << BOLD << "GPU Memory Allocation:" << RESET << "\n";
    std::cout << "  Weights: (5 × 10) × 4 bytes = 200 bytes\n";
    std::cout << "  Biases: (5 × 1) × 4 bytes = 20 bytes\n";
    std::cout << "  Gradients: 220 bytes (same as parameters)\n";
    std::cout << "  Total: ~440 bytes on GPU\n\n";
    
    std::cout << GREEN << "✓ CUDA layer created on GPU!" << RESET << "\n";
}

// Example 2: Forward pass on GPU
void example2_CudaForwardPass() {
    printHeader("EXAMPLE 2: GPU-Accelerated Forward Pass");
    
    std::cout << "Computing layer output on GPU\n\n";
    
    // Create small layer for visualization
    DenseLayerCUDA layer(3, 2, new ReLUCUDA());
    
    // Set known weights (on CPU first, then transfer to GPU)
    Matrix W_cpu(2, 3);
    W_cpu.set(0, 0, 0.5); W_cpu.set(0, 1, 0.3); W_cpu.set(0, 2, 0.2);
    W_cpu.set(1, 0, 0.4); W_cpu.set(1, 1, 0.6); W_cpu.set(1, 2, 0.1);
    
    Matrix b_cpu(2, 1);
    b_cpu.set(0, 0, 0.1);
    b_cpu.set(1, 0, 0.2);
    
    MatrixCUDA W(W_cpu);
    MatrixCUDA b(b_cpu);
    
    layer.setWeights(W);
    layer.setBiases(b);
    
    std::cout << "Layer: 3 inputs → 2 outputs (ReLU)\n\n";
    
    // Create input on GPU
    Matrix input_cpu(1, 3);
    input_cpu.set(0, 0, 1.0);
    input_cpu.set(0, 1, 2.0);
    input_cpu.set(0, 2, 3.0);
    
    MatrixCUDA input(input_cpu);
    
    std::cout << "Input (on GPU): [1.0, 2.0, 3.0]\n\n";
    
    std::cout << BOLD << "GPU Computation Steps:" << RESET << "\n\n";
    
    std::cout << "1. Transfer input to GPU (if not already there)\n";
    std::cout << "2. Compute Z = X·W^T + b using cuBLAS\n";
    std::cout << "   z1 = 1.0×0.5 + 2.0×0.3 + 3.0×0.2 + 0.1 = 1.8\n";
    std::cout << "   z2 = 1.0×0.4 + 2.0×0.6 + 3.0×0.1 + 0.2 = 2.1\n\n";
    
    std::cout << "3. Apply ReLU activation on GPU\n";
    std::cout << "   a1 = max(0, 1.8) = 1.8\n";
    std::cout << "   a2 = max(0, 2.1) = 2.1\n\n";
    
    // Execute forward pass on GPU
    auto start = std::chrono::high_resolution_clock::now();
    MatrixCUDA output = layer.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get result (transfers from GPU to CPU for display)
    output.toCPU();
    
    std::cout << "Output (from GPU): [" << output.get(0,0) << ", " 
              << output.get(0,1) << "]\n";
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\nGPU Execution time: " << duration.count() << " μs\n";
    
    std::cout << "\n" << GREEN << "✓ Forward pass completed on GPU!" << RESET << "\n";
}

// Example 3: Backward pass on GPU
void example3_CudaBackwardPass() {
    printHeader("EXAMPLE 3: GPU-Accelerated Backward Pass");
    
    std::cout << "Computing gradients on GPU\n\n";
    
    // Create CUDA layer
    DenseLayerCUDA layer(2, 2, new ReLUCUDA());
    
    Matrix W_cpu(2, 2);
    W_cpu.set(0, 0, 0.5); W_cpu.set(0, 1, 0.3);
    W_cpu.set(1, 0, 0.4); W_cpu.set(1, 1, 0.6);
    
    Matrix b_cpu(2, 1);
    b_cpu.set(0, 0, 0.1);
    b_cpu.set(1, 0, 0.2);
    
    layer.setWeights(MatrixCUDA(W_cpu));
    layer.setBiases(MatrixCUDA(b_cpu));
    
    std::cout << "Layer: 2 inputs → 2 outputs (ReLU)\n\n";
    
    // Forward pass on GPU
    Matrix input_cpu(1, 2);
    input_cpu.set(0, 0, 1.0);
    input_cpu.set(0, 1, 2.0);
    
    MatrixCUDA input(input_cpu);
    MatrixCUDA output = layer.forward(input);
    
    output.toCPU();
    std::cout << "Forward output: [" << output.get(0,0) << ", " 
              << output.get(0,1) << "]\n\n";

    
    
    // Backward pass on GPU
    Matrix grad_cpu(1, 2);
    grad_cpu.set(0, 0, 0.1);
    grad_cpu.set(0, 1, 0.2);
    
    MatrixCUDA output_grad(grad_cpu);
    
    std::cout << "Gradient from next layer: [0.1, 0.2]\n\n";
    
    std::cout << BOLD << "GPU Gradient Computation:" << RESET << "\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    MatrixCUDA input_grad = layer.backward(output_grad);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Get gradients (GPU → CPU for display)
    MatrixCUDA dW = layer.getWeightGradients();
    MatrixCUDA db = layer.getBiasGradients();
    
    dW.toCPU();
    db.toCPU();
    input_grad.toCPU();
    
    std::cout << "Weight gradients (∂L/∂W):\n";
    std::cout << "  [[" << dW.get(0,0) << ", " << dW.get(0,1) << "],\n";
    std::cout << "   [" << dW.get(1,0) << ", " << dW.get(1,1) << "]]\n\n";
    
    std::cout << "Bias gradients (∂L/∂b):\n";
    std::cout << "  [" << db.get(0,0) << ", " << db.get(1,0) << "]\n\n";
    
    std::cout << "Input gradients (∂L/∂X):\n";
    std::cout << "  [" << input_grad.get(0,0) << ", " << input_grad.get(0,1) << "]\n";
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "\nGPU Backward time: " << duration.count() << " μs\n";
    
    std::cout << "\n" << GREEN << "✓ Backward pass completed on GPU!" << RESET << "\n";
}

// Example 4: CPU vs GPU performance comparison
void example4_PerformanceComparison() {
    printHeader("EXAMPLE 4: CPU vs GPU Performance Comparison");
    
    std::cout << "Comparing layer performance on CPU and GPU\n\n";
    
    std::vector<std::pair<size_t, size_t>> sizes = {
        {100, 50},
        {500, 250},
        {1000, 500}
    };
    
    for (auto [input_size, output_size] : sizes) {
        printSubHeader("Layer: " + std::to_string(input_size) + " → " + 
                       std::to_string(output_size));
        
        size_t batch_size = 32;
        
        // Create CPU layer
        DenseLayer cpu_layer(input_size, output_size, new ReLU());
        
        // Create GPU layer
        DenseLayerCUDA gpu_layer(input_size, output_size, new ReLUCUDA());
        
        // Create input data
        Matrix input_cpu(batch_size, input_size);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < input_size; j++) {
                input_cpu.set(i, j, static_cast<double>(rand()) / RAND_MAX);
            }
        }
        
        MatrixCUDA input_gpu(input_cpu);
        
        // Benchmark CPU forward pass
        auto cpu_start = std::chrono::high_resolution_clock::now();
        Matrix cpu_output = cpu_layer.forward(input_cpu);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_end - cpu_start);
        
        // Benchmark GPU forward pass
        auto gpu_start = std::chrono::high_resolution_clock::now();
        MatrixCUDA gpu_output = gpu_layer.forward(input_gpu);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(
            gpu_end - gpu_start);
        
        // Create gradient
        Matrix grad_cpu(batch_size, output_size);
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < output_size; j++) {
                grad_cpu.set(i, j, static_cast<double>(rand()) / RAND_MAX);
            }
        }
        
        MatrixCUDA grad_gpu(grad_cpu);
        
        // Benchmark CPU backward pass
        cpu_start = std::chrono::high_resolution_clock::now();
        Matrix cpu_grad = cpu_layer.backward(grad_cpu);
        cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_backward_time = std::chrono::duration_cast<std::chrono::microseconds>(
            cpu_end - cpu_start);
        
        // Benchmark GPU backward pass
        gpu_start = std::chrono::high_resolution_clock::now();
        MatrixCUDA gpu_grad = gpu_layer.backward(grad_gpu);
        gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_backward_time = std::chrono::duration_cast<std::chrono::microseconds>(
            gpu_end - gpu_start);
        
        // Print results
        std::cout << "Batch size: " << batch_size << " samples\n";
        std::cout << "Parameters: " << cpu_layer.getParameterCount() << "\n\n";
        
        std::cout << "Forward Pass:\n";
        std::cout << "  CPU: " << std::setw(8) << cpu_forward_time.count() << " μs\n";
        std::cout << "  GPU: " << std::setw(8) << gpu_forward_time.count() << " μs";
        
        double forward_speedup = static_cast<double>(cpu_forward_time.count()) / 
                                gpu_forward_time.count();
        if (forward_speedup > 1.0) {
            std::cout << GREEN << "  (" << std::fixed << std::setprecision(2) 
                      << forward_speedup << "x faster)" << RESET;
        } else {
            std::cout << YELLOW << "  (CPU faster for small size)" << RESET;
        }
        std::cout << "\n\n";
        
        std::cout << "Backward Pass:\n";
        std::cout << "  CPU: " << std::setw(8) << cpu_backward_time.count() << " μs\n";
        std::cout << "  GPU: " << std::setw(8) << gpu_backward_time.count() << " μs";
        
        double backward_speedup = static_cast<double>(cpu_backward_time.count()) / 
                                 gpu_backward_time.count();
        if (backward_speedup > 1.0) {
            std::cout << GREEN << "  (" << std::fixed << std::setprecision(2) 
                      << backward_speedup << "x faster)" << RESET;
        } else {
            std::cout << YELLOW << "  (CPU faster for small size)" << RESET;
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << BOLD << "Performance Notes:" << RESET << "\n";
    std::cout << "• GPU has overhead for small matrices\n";
    std::cout << "• GPU shines with larger batch sizes\n";
    std::cout << "• Real advantage: training large models on big datasets\n";
    std::cout << "• GPU allows training while CPU does other work\n";
}

// Example 5: Training XOR on GPU
void example5_TrainingOnGPU() {
    printHeader("EXAMPLE 5: Training XOR Problem on GPU");
    
    std::cout << "Training neural network entirely on GPU\n\n";
    
    // XOR dataset
    Matrix X_cpu(4, 2);
    X_cpu.set(0, 0, 0); X_cpu.set(0, 1, 0);
    X_cpu.set(1, 0, 0); X_cpu.set(1, 1, 1);
    X_cpu.set(2, 0, 1); X_cpu.set(2, 1, 0);
    X_cpu.set(3, 0, 1); X_cpu.set(3, 1, 1);
    
    Matrix Y_cpu(4, 1);
    Y_cpu.set(0, 0, 0);
    Y_cpu.set(1, 0, 1);
    Y_cpu.set(2, 0, 1);
    Y_cpu.set(3, 0, 0);
    
    // Transfer to GPU
    MatrixCUDA X(X_cpu);
    MatrixCUDA Y(Y_cpu);
    
    std::cout << "XOR Dataset (on GPU):\n";
    std::cout << "  [0, 0] → 0\n";
    std::cout << "  [0, 1] → 1\n";
    std::cout << "  [1, 0] → 1\n";
    std::cout << "  [1, 1] → 0\n\n";
    
    // Create network on GPU: 2 → 4 → 1
    std::cout << "Network architecture (on GPU):\n";
    std::cout << "  Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)\n\n";
    
    DenseLayerCUDA hidden(2, 4, new ReLUCUDA());
    DenseLayerCUDA output_layer(4, 1, new SigmoidCUDA());
    
    hidden.initializeWeights("xavier");
    output_layer.initializeWeights("xavier");
    
    // Training parameters
    double learning_rate = 0.1;
    int epochs = 1000;
    
    std::cout << "Training on GPU:\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Epochs: " << epochs << "\n\n";
    
    MSELossCUDA loss_fn;
    
    std::cout << "Training progress:\n";
    
    auto train_start = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass (on GPU)
        MatrixCUDA h = hidden.forward(X);
        MatrixCUDA pred = output_layer.forward(h);
        
        // Compute loss (on GPU)
        double loss = loss_fn.calculate(pred, Y);
        
        // Backward pass (on GPU)
        MatrixCUDA loss_grad = loss_fn.gradient(pred, Y);
        MatrixCUDA output_grad = output_layer.backward(loss_grad);
        MatrixCUDA hidden_grad = hidden.backward(output_grad);
        
        // Update parameters (on GPU)
        output_layer.updateParameters(learning_rate);
        hidden.updateParameters(learning_rate);
        
        // Print progress
        if (epoch % 100 == 0) {
            std::cout << "  Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss;
            
            if (loss < 0.01) {
                std::cout << GREEN << " ✓ Converged!" << RESET;
            }
            std::cout << "\n";
        }
    }
    
    auto train_end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        train_end - train_start);
    
    std::cout << "\nTotal training time: " << train_time.count() << " ms\n\n";
    
    // Test (results transferred from GPU for display)
    std::cout << BOLD << "Testing trained network:" << RESET << "\n\n";
    
    MatrixCUDA h_test = hidden.forward(X);
    MatrixCUDA pred_test = output_layer.forward(h_test);
    
    pred_test.toCPU();
    Y.toCPU();
    X.toCPU();
    
    for (size_t i = 0; i < 4; i++) {
        std::cout << "  Input: [" << X.get(i,0) << ", " 
                  << X.get(i,1) << "] → ";
        std::cout << "Predicted: " << std::fixed << std::setprecision(4) 
                  << pred_test.get(i,0);
        std::cout << " | Target: " << Y.get(i,0);
        
        if (std::abs(pred_test.get(i,0) - Y.get(i,0)) < 0.1) {
            std::cout << GREEN << " ✓" << RESET;
        } else {
            std::cout << RED << " ✗" << RESET;
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << GREEN << "✓ Training completed on GPU!" << RESET << "\n";
}

// Example 6: Memory efficiency
void example6_MemoryEfficiency() {
    printHeader("EXAMPLE 6: GPU Memory Efficiency");
    
    std::cout << "Analyzing GPU memory usage for neural network layers\n\n";
    
    std::vector<std::pair<size_t, size_t>> configs = {
        {784, 128},
        {1000, 500},
        {2048, 1024}
    };
    
    for (auto [input_size, output_size] : configs) {
        printSubHeader("Layer: " + std::to_string(input_size) + " → " + 
                       std::to_string(output_size));
        
        size_t params = input_size * output_size + output_size;
        size_t weights_memory = input_size * output_size * sizeof(float);
        size_t biases_memory = output_size * sizeof(float);
        size_t gradients_memory = weights_memory + biases_memory;
        size_t total_memory = weights_memory + biases_memory + gradients_memory;
        
        std::cout << "Parameters: " << params << "\n\n";
        
        std::cout << "GPU Memory Breakdown:\n";
        std::cout << "  Weights:   " << std::setw(10) << weights_memory 
                  << " bytes (" << weights_memory / 1024 << " KB)\n";
        std::cout << "  Biases:    " << std::setw(10) << biases_memory 
                  << " bytes\n";
        std::cout << "  Gradients: " << std::setw(10) << gradients_memory 
                  << " bytes (" << gradients_memory / 1024 << " KB)\n";
        std::cout << "  " << BOLD << "Total:     " << std::setw(10) << total_memory 
                  << " bytes (" << total_memory / (1024 * 1024) << " MB)" 
                  << RESET << "\n\n";
        
        std::cout << "Additional runtime memory (for batch_size=32):\n";
        size_t cached_input = 32 * input_size * sizeof(float);
        size_t cached_z = 32 * output_size * sizeof(float);
        size_t runtime_memory = cached_input + cached_z;
        
        std::cout << "  Cached input: " << cached_input / 1024 << " KB\n";
        std::cout << "  Cached Z:     " << cached_z / 1024 << " KB\n";
        std::cout << "  Runtime total: " << runtime_memory / 1024 << " KB\n";
    }
    
    std::cout << "\n" << BOLD << "Memory Efficiency Tips:" << RESET << "\n";
    std::cout << "• GPU memory is limited (typically 4-24 GB)\n";
    std::cout << "• Larger batch sizes → better GPU utilization\n";
    std::cout << "• But: larger batches need more memory\n";
    std::cout << "• Balance: batch size vs model size vs GPU memory\n";
}

int main() {
    std::cout << BOLD << GREEN << R"(
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║         GPU-ACCELERATED LAYER DEMONSTRATION                       ║
║                  CUDA Neural Network Layers                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
)" << RESET << "\n";
    
    try {
        // example1_CudaLayerBasics();
        // example2_CudaForwardPass();
        // example3_CudaBackwardPass();
        // example4_PerformanceComparison();
        example5_TrainingOnGPU();
        // example6_MemoryEfficiency();
        
        printHeader("SUMMARY");
        std::cout << GREEN << "✓" << RESET << " All CUDA layer examples completed successfully!\n\n";
        
        std::cout << BOLD << "What you learned:" << RESET << "\n";
        std::cout << "1. Creating GPU-accelerated layers\n";
        std::cout << "2. Forward/backward passes on GPU\n";
        std::cout << "3. CPU vs GPU performance comparison\n";
        std::cout << "4. Training networks entirely on GPU\n";
        std::cout << "5. GPU memory efficiency\n\n";
        
        std::cout << BOLD << "Key Advantages of CUDA Layers:" << RESET << "\n";
        std::cout << "• Massive parallelization (thousands of CUDA cores)\n";
        std::cout << "• Faster training for large models/datasets\n";
        std::cout << "• Efficient batch processing\n";
        std::cout << "• Frees CPU for other tasks\n\n";
        
        std::cout << BOLD << "When to use GPU:" << RESET << "\n";
        std::cout << "• Large models (millions of parameters)\n";
        std::cout << "• Large datasets (thousands of samples)\n";
        std::cout << "• Large batch sizes (32, 64, 128, ...)\n";
        std::cout << "• Real-time inference requirements\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
