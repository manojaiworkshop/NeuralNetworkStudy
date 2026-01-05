/**
 * =====================================================
 * CUDA-ACCELERATED ACTIVATION FUNCTIONS DEMONSTRATION
 * =====================================================
 * 
 * This example demonstrates:
 * 1. CUDA-accelerated activation functions
 * 2. Performance comparison: CPU vs CUDA
 * 3. Different activation types on CUDA
 * 4. Batch processing advantages
 * 5. Forward and backward passes on CUDA
 */

#include "nn/matrix.h"
#include "nn/matrix_cuda.h"
#include "nn/activation.h"
#include "nn/activation_cuda.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// ==================== TIMER UTILITY ====================

class Timer {
private:
    high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// ==================== PART 1: BASIC CUDA ACTIVATION ====================

void demonstrateBasicCUDAActivation() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         BASIC CUDA ACTIVATION DEMONSTRATION             ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "Creating a small matrix (3×4) to demonstrate CUDA activation...\n\n";
    
    // Create input matrix
    Matrix input_cpu(3, 4);
    input_cpu.set(0, 0, -2.0); input_cpu.set(0, 1,  0.5); 
    input_cpu.set(0, 2,  1.5); input_cpu.set(0, 3, -0.8);
    input_cpu.set(1, 0,  2.0); input_cpu.set(1, 1, -1.0); 
    input_cpu.set(1, 2,  0.0); input_cpu.set(1, 3,  3.0);
    input_cpu.set(2, 0, -3.0); input_cpu.set(2, 1,  1.0); 
    input_cpu.set(2, 2, -0.5); input_cpu.set(2, 3,  0.8);
    
    cout << "Input Matrix (CPU):\n";
    input_cpu.print();
    
    // Transfer to CUDA
    MatrixCUDA input_cuda(input_cpu);
    cout << "\n✓ Matrix transferred to CUDA\n";
    
    // Apply ReLU activation on CUDA
    ReLUCUDA relu_cuda;
    MatrixCUDA output_cuda = relu_cuda.forward(input_cuda);
    
    cout << "\nAfter ReLU Activation (CUDA):\n";
    Matrix output_cpu = static_cast<Matrix>(output_cuda);
    output_cpu.print();
    
    cout << "\nNOTE: Negative values → 0, Positive values → unchanged\n";
}

// ==================== PART 2: COMPARE ALL ACTIVATIONS ====================

void compareAllActivations() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         COMPARING DIFFERENT CUDA ACTIVATIONS            ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    // Create input
    Matrix input_cpu(2, 5);
    input_cpu.set(0, 0, -2.0); input_cpu.set(0, 1, -1.0);
    input_cpu.set(0, 2,  0.0); input_cpu.set(0, 3,  1.0);
    input_cpu.set(0, 4,  2.0);
    input_cpu.set(1, 0, -3.0); input_cpu.set(1, 1, -0.5);
    input_cpu.set(1, 2,  0.5); input_cpu.set(1, 3,  1.5);
    input_cpu.set(1, 4,  3.0);
    
    cout << "Input Matrix:\n";
    input_cpu.print();
    
    MatrixCUDA input_cuda(input_cpu);
    
    // Test different activations
    cout << "\n--- 1. SIGMOID CUDA ---\n";
    SigmoidCUDA sigmoid_cuda;
    MatrixCUDA sigmoid_out = sigmoid_cuda.forward(input_cuda);
    Matrix sigmoid_cpu = static_cast<Matrix>(sigmoid_out);
    sigmoid_cpu.print();
    cout << "Range: (0, 1) - Good for binary classification\n";
    
    cout << "\n--- 2. RELU CUDA ---\n";
    ReLUCUDA relu_cuda;
    MatrixCUDA relu_out = relu_cuda.forward(input_cuda);
    Matrix relu_cpu = static_cast<Matrix>(relu_out);
    relu_cpu.print();
    cout << "Range: [0, ∞) - Most popular for hidden layers\n";
    
    cout << "\n--- 3. TANH CUDA ---\n";
    TanhCUDA tanh_cuda;
    MatrixCUDA tanh_out = tanh_cuda.forward(input_cuda);
    Matrix tanh_cpu = static_cast<Matrix>(tanh_out);
    tanh_cpu.print();
    cout << "Range: (-1, 1) - Zero-centered\n";
    
    cout << "\n--- 4. LEAKY RELU CUDA (α=0.1) ---\n";
    LeakyReLUCUDA leaky_relu_cuda(0.1f);
    MatrixCUDA leaky_relu_out = leaky_relu_cuda.forward(input_cuda);
    Matrix leaky_relu_cpu = static_cast<Matrix>(leaky_relu_out);
    leaky_relu_cpu.print();
    cout << "Range: (-∞, ∞) - Allows small negative gradient\n";
    
    cout << "\n--- 5. ELU CUDA (α=1.0) ---\n";
    ELUCUDA elu_cuda(1.0f);
    MatrixCUDA elu_out = elu_cuda.forward(input_cuda);
    Matrix elu_cpu = static_cast<Matrix>(elu_out);
    elu_cpu.print();
    cout << "Range: (-α, ∞) - Smooth negative part\n";
}

// ==================== PART 3: PERFORMANCE BENCHMARK ====================

void benchmarkPerformance(int size) {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║           CPU vs CUDA PERFORMANCE BENCHMARK             ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "Matrix size: " << size << " × " << size << endl;
    cout << "Total elements: " << (size * size) << "\n\n";
    
    // Create large matrix
    Matrix input_cpu(size, size);
    input_cpu.randomize(-1.0, 1.0);
    
    MatrixCUDA input_cuda(input_cpu);
    
    Timer timer;
    
    // CPU ReLU
    ReLU relu_cpu;
    cout << "CPU ReLU Activation..." << endl;
    timer.start();
    Matrix output_cpu = relu_cpu.forward(input_cpu);
    double cpu_time = timer.elapsed_ms();
    cout << "  Time: " << fixed << setprecision(3) << cpu_time << " ms\n";
    
    // CUDA ReLU
    ReLUCUDA relu_cuda;
    cout << "\nCUDA ReLU Activation..." << endl;
    timer.start();
    MatrixCUDA output_cuda = relu_cuda.forward(input_cuda);
    double cuda_time = timer.elapsed_ms();
    cout << "  Time: " << fixed << setprecision(3) << cuda_time << " ms\n";
    
    // Speedup
    double speedup = cpu_time / cuda_time;
    cout << "\n" << string(50, '=') << endl;
    cout << "Speedup: " << fixed << setprecision(2) << speedup << "x";
    if (speedup > 1.0) {
        cout << " ✓ CUDA is faster!\n";
    } else {
        cout << " (CUDA overhead for small matrices)\n";
    }
    cout << string(50, '=') << endl;
    
    // Verify results match
    Matrix output_cuda_cpu = static_cast<Matrix>(output_cuda);
    bool correct = true;
    double max_diff = 0.0;
    
    for (size_t i = 0; i < min(size_t(10), input_cpu.getRows()); i++) {
        for (size_t j = 0; j < min(size_t(10), input_cpu.getCols()); j++) {
            double diff = abs(output_cpu.get(i, j) - output_cuda_cpu.get(i, j));
            max_diff = max(max_diff, diff);
            if (diff > 1e-5) {
                correct = false;
            }
        }
    }
    
    cout << "\nVerification: " << (correct ? "✓ PASSED" : "✗ FAILED") << endl;
    cout << "Max difference: " << scientific << setprecision(2) << max_diff << "\n";
}

// ==================== PART 4: BACKWARD PASS DEMONSTRATION ====================

void demonstrateBackwardPass() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         CUDA BACKWARD PASS (GRADIENT COMPUTATION)       ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "Demonstrating gradient computation on CUDA...\n\n";
    
    // Create input
    Matrix input_cpu(2, 3);
    input_cpu.set(0, 0, -1.0); input_cpu.set(0, 1,  2.0); input_cpu.set(0, 2, 0.5);
    input_cpu.set(1, 0,  3.0); input_cpu.set(1, 1, -2.0); input_cpu.set(1, 2, 1.0);
    
    cout << "Input:\n";
    input_cpu.print();
    
    MatrixCUDA input_cuda(input_cpu);
    
    // Forward pass
    ReLUCUDA relu_cuda;
    MatrixCUDA output_cuda = relu_cuda.forward(input_cuda);
    
    cout << "\nAfter ReLU Forward:\n";
    Matrix output_cpu = static_cast<Matrix>(output_cuda);
    output_cpu.print();
    
    // Simulate gradient from loss
    Matrix grad_output_cpu(2, 3);
    grad_output_cpu.fill(1.0);
    
    cout << "\nGradient from next layer (∂L/∂output):\n";
    grad_output_cpu.print();
    
    MatrixCUDA grad_output_cuda(grad_output_cpu);
    
    // Backward pass
    MatrixCUDA grad_input_cuda = relu_cuda.backward(input_cuda, grad_output_cuda);
    
    cout << "\nGradient with respect to input (∂L/∂input):\n";
    Matrix grad_input_cpu = static_cast<Matrix>(grad_input_cuda);
    grad_input_cpu.print();
    
    cout << "\nEXPLANATION:\n";
    cout << "  • Positive inputs (2.0, 3.0, 0.5, 1.0) → gradient = 1.0 (flows through)\n";
    cout << "  • Negative inputs (-1.0, -2.0) → gradient = 0.0 (blocked)\n";
    cout << "  • This is how backpropagation works on CUDA!\n";
}

// ==================== PART 5: BATCH PROCESSING ====================

void demonstrateBatchProcessing() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         BATCH PROCESSING WITH CUDA ACTIVATIONS          ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "Processing multiple samples in parallel on CUDA...\n\n";
    
    int batch_size = 4;
    int features = 5;
    
    Matrix batch_cpu(batch_size, features);
    batch_cpu.randomize(-2.0, 2.0);
    
    cout << "Input Batch (" << batch_size << " samples × " << features << " features):\n";
    batch_cpu.print();
    
    MatrixCUDA batch_cuda(batch_cpu);
    
    // Apply activation to entire batch
    cout << "\nApplying Sigmoid activation to entire batch on CUDA...\n";
    
    SigmoidCUDA sigmoid_cuda;
    Timer timer;
    timer.start();
    MatrixCUDA activated_batch = sigmoid_cuda.forward(batch_cuda);
    double batch_time = timer.elapsed_ms();
    
    cout << "\nActivated Batch:\n";
    Matrix activated_cpu = static_cast<Matrix>(activated_batch);
    activated_cpu.print();
    
    cout << "\nBatch processing time: " << fixed << setprecision(3) 
         << batch_time << " ms\n";
    
    cout << "\nKEY ADVANTAGE:\n";
    cout << "  • All " << batch_size << " samples processed in parallel!\n";
    cout << "  • CUDA threads work simultaneously on different elements\n";
    cout << "  • Massive speedup for large batches (e.g., 256, 512 samples)\n";
}

// ==================== PART 6: SCALABILITY TEST ====================

void scalabilityTest() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║              SCALABILITY: CPU vs CUDA                   ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    vector<int> sizes = {64, 128, 256, 512, 1024};
    
    cout << setw(10) << "Size"
         << setw(15) << "CPU (ms)"
         << setw(15) << "CUDA (ms)"
         << setw(15) << "Speedup" << endl;
    cout << string(55, '-') << endl;
    
    ReLU relu_cpu;
    ReLUCUDA relu_cuda;
    Timer timer;
    
    for (int size : sizes) {
        // Create matrices
        Matrix input_cpu(size, size);
        input_cpu.randomize(-1.0, 1.0);
        MatrixCUDA input_cuda(input_cpu);
        
        // CPU benchmark
        timer.start();
        Matrix output_cpu = relu_cpu.forward(input_cpu);
        double cpu_time = timer.elapsed_ms();
        
        // CUDA benchmark
        timer.start();
        MatrixCUDA output_cuda = relu_cuda.forward(input_cuda);
        double cuda_time = timer.elapsed_ms();
        
        double speedup = cpu_time / cuda_time;
        
        cout << setw(10) << size
             << setw(15) << fixed << setprecision(2) << cpu_time
             << setw(15) << fixed << setprecision(2) << cuda_time
             << setw(15) << fixed << setprecision(1) << speedup << "x" << endl;
    }
    
    cout << string(55, '-') << endl;
    cout << "\nOBSERVATION:\n";
    cout << "  • Small matrices: CUDA overhead may dominate\n";
    cout << "  • Large matrices: CUDA shows significant speedup\n";
    cout << "  • Larger the data, better the CUDA performance!\n";
}

// ==================== PART 7: NEURAL NETWORK LAYER ====================

void demonstrateNeuralNetworkLayer() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         COMPLETE NEURAL NETWORK LAYER ON CUDA           ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "Simulating: Input → Linear → ReLU (entire layer on CUDA)\n\n";
    
    // Input: 2 samples, 4 features
    Matrix input_cpu(2, 4);
    input_cpu.set(0, 0, 0.5); input_cpu.set(0, 1, 0.8);
    input_cpu.set(0, 2, 0.3); input_cpu.set(0, 3, 0.9);
    input_cpu.set(1, 0, 0.2); input_cpu.set(1, 1, 0.6);
    input_cpu.set(1, 2, 0.7); input_cpu.set(1, 3, 0.4);
    
    cout << "Input (2 samples × 4 features):\n";
    input_cpu.print();
    
    // Weights: 4 inputs → 3 hidden neurons
    Matrix weights_cpu(4, 3);
    weights_cpu.randomize(-0.5, 0.5);
    
    cout << "\nWeights (4×3):\n";
    weights_cpu.print();
    
    // Transfer to CUDA
    MatrixCUDA input_cuda(input_cpu);
    MatrixCUDA weights_cuda(weights_cpu);
    
    cout << "\n✓ Data transferred to CUDA\n";
    
    // Linear transformation on CUDA
    cout << "\nStep 1: Linear transformation (z = input × weights)\n";
    MatrixCUDA z_cuda = input_cuda.multiplyGPU(weights_cuda);
    
    Matrix z_cpu = static_cast<Matrix>(z_cuda);
    cout << "Result z (2×3):\n";
    z_cpu.print();
    
    // ReLU activation on CUDA
    cout << "\nStep 2: ReLU activation (a = ReLU(z))\n";
    ReLUCUDA relu_cuda;
    MatrixCUDA a_cuda = relu_cuda.forward(z_cuda);
    
    Matrix a_cpu = static_cast<Matrix>(a_cuda);
    cout << "Result a (2×3):\n";
    a_cpu.print();
    
    cout << "\n✓ Complete layer computed on CUDA!\n";
    cout << "✓ All operations stayed on CUDA (minimal CPU-CUDA transfers)\n";
}

// ==================== MAIN ====================

int main() {
    cout << "\n";
    cout << "████████████████████████████████████████████████████████████\n";
    cout << "█                                                          █\n";
    cout << "█     CUDA-ACCELERATED ACTIVATION FUNCTIONS                 █\n";
    cout << "█     CUDA Performance Demonstration                       █\n";
    cout << "█                                                          █\n";
    cout << "████████████████████████████████████████████████████████████\n";
    
    // Check CUDA availability
    MatrixCUDA::printDeviceInfo();
    
    cout << "\nPress Enter to start demonstrations...";
    cin.get();
    
    demonstrateBasicCUDAActivation();
    
    cout << "\n\nPress Enter to compare different activations...";
    cin.get();
    compareAllActivations();
    
    cout << "\n\nPress Enter for performance benchmark (512×512 matrix)...";
    cin.get();
    benchmarkPerformance(512);
    
    cout << "\n\nPress Enter to see backward pass...";
    cin.get();
    demonstrateBackwardPass();
    
    cout << "\n\nPress Enter for batch processing demo...";
    cin.get();
    demonstrateBatchProcessing();
    
    cout << "\n\nPress Enter for scalability test...";
    cin.get();
    scalabilityTest();
    
    cout << "\n\nPress Enter for neural network layer demo...";
    cin.get();
    demonstrateNeuralNetworkLayer();
    
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║                 DEMONSTRATION COMPLETE!                ║\n";
    cout << "║                                                        ║\n";
    cout << "║  Key Takeaways:                                        ║\n";
    cout << "║  • CUDA activations process data in parallel            ║\n";
    cout << "║  • Significant speedup for large matrices              ║\n";
    cout << "║  • Both forward and backward passes accelerated        ║\n";
    cout << "║  • Batch processing is highly efficient on CUDA         ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}
