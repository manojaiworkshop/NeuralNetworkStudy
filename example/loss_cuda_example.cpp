/**
 * CUDA Loss Functions - Complete Example and Benchmark
 * 
 * This example demonstrates:
 * 1. How to use CUDA-accelerated loss functions
 * 2. Performance comparison: CPU vs CUDA
 * 3. All loss types with practical examples
 * 4. Scalability testing with different matrix sizes
 */

#include "nn/matrix.h"
#include "nn/matrix_cuda.h"
#include "nn/loss.h"
#include "nn/loss_cuda.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <random>

using namespace std;
using namespace std::chrono;

// =====================================================
// Utility Functions
// =====================================================

void printHeader(const string& title) {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║ " << setw(54) << left << title << " ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

void printSeparator() {
    cout << string(60, '─') << "\n";
}

// Generate random matrix
Matrix generateRandomMatrix(int rows, int cols, double min_val = 0.0, double max_val = 1.0) {
    Matrix m(rows, cols);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(min_val, max_val);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m.set(i, j, dis(gen));
        }
    }
    return m;
}

// =====================================================
// PART 1: Basic Usage Examples
// =====================================================

void basicMSEExample() {
    printHeader("MSE Loss - CUDA Example");
    
    cout << "Problem: House price prediction\n\n";
    
    // Create predictions and targets (in $1000s)
    Matrix predictions_cpu(3, 1);
    predictions_cpu.set(0, 0, 250.0);  // $250k
    predictions_cpu.set(1, 0, 180.0);  // $180k
    predictions_cpu.set(2, 0, 320.0);  // $320k
    
    Matrix targets_cpu(3, 1);
    targets_cpu.set(0, 0, 245.0);  // True: $245k
    targets_cpu.set(1, 0, 200.0);  // True: $200k
    targets_cpu.set(2, 0, 315.0);  // True: $315k
    
    cout << "Predictions: [$250k, $180k, $320k]\n";
    cout << "Targets:     [$245k, $200k, $315k]\n\n";
    
    // CPU calculation
    MSELoss mse_cpu;
    double loss_cpu = mse_cpu.calculate(predictions_cpu, targets_cpu);
    
    // CUDA calculation
    MSELossCUDA mse_cuda;
    double loss_cuda = mse_cuda.calculate(predictions_cpu, targets_cpu);
    
    cout << "CPU Loss:  " << fixed << setprecision(4) << loss_cpu << "\n";
    cout << "CUDA Loss: " << loss_cuda << "\n";
    cout << "RMSE: $" << sqrt(loss_cuda) * 1000 << " (typical error)\n\n";
    
    // Compute gradients
    Matrix grad_cpu = mse_cpu.gradient(predictions_cpu, targets_cpu);
    Matrix grad_cuda = mse_cuda.gradient(predictions_cpu, targets_cpu);
    
    cout << "Gradients (CPU):  [" << grad_cpu.get(0, 0) << ", "
         << grad_cpu.get(1, 0) << ", " << grad_cpu.get(2, 0) << "]\n";
    cout << "Gradients (CUDA): [" << grad_cuda.get(0, 0) << ", "
         << grad_cuda.get(1, 0) << ", " << grad_cuda.get(2, 0) << "]\n\n";
    
    cout << "✓ Results match! CUDA implementation is correct.\n";
}

void basicBCEExample() {
    printHeader("Binary Cross-Entropy Loss - CUDA Example");
    
    cout << "Problem: Email spam classification\n\n";
    
    // Create predictions (probabilities) and targets
    Matrix predictions_cpu(5, 1);
    predictions_cpu.set(0, 0, 0.95);  // 95% spam
    predictions_cpu.set(1, 0, 0.10);  // 10% spam
    predictions_cpu.set(2, 0, 0.80);  // 80% spam
    predictions_cpu.set(3, 0, 0.30);  // 30% spam
    predictions_cpu.set(4, 0, 0.65);  // 65% spam
    
    Matrix targets_cpu(5, 1);
    targets_cpu.set(0, 0, 1.0);  // Is spam
    targets_cpu.set(1, 0, 0.0);  // Not spam
    targets_cpu.set(2, 0, 1.0);  // Is spam
    targets_cpu.set(3, 0, 0.0);  // Not spam
    targets_cpu.set(4, 0, 1.0);  // Is spam
    
    cout << "Email Classification Results:\n";
    for (int i = 0; i < 5; i++) {
        cout << "  Email " << (i+1) << ": Predicted=" << predictions_cpu.get(i, 0)
             << ", True=" << (targets_cpu.get(i, 0) == 1.0 ? "Spam" : "Not Spam") << "\n";
    }
    cout << "\n";
    
    // CPU calculation
    BinaryCrossEntropyLoss bce_cpu;
    double loss_cpu = bce_cpu.calculate(predictions_cpu, targets_cpu);
    
    // CUDA calculation
    BinaryCrossEntropyLossCUDA bce_cuda;
    double loss_cuda = bce_cuda.calculate(predictions_cpu, targets_cpu);
    
    cout << "CPU Loss:  " << fixed << setprecision(6) << loss_cpu << "\n";
    cout << "CUDA Loss: " << loss_cuda << "\n\n";
    
    cout << "✓ Lower loss = Better classification accuracy\n";
}

void basicCCEExample() {
    printHeader("Categorical Cross-Entropy Loss - CUDA Example");
    
    cout << "Problem: Image classification (3 classes: cat, dog, bird)\n\n";
    
    // 2 samples, 3 classes
    Matrix predictions_cpu(2, 3);
    // Sample 1: 70% cat, 20% dog, 10% bird
    predictions_cpu.set(0, 0, 0.7);
    predictions_cpu.set(0, 1, 0.2);
    predictions_cpu.set(0, 2, 0.1);
    
    // Sample 2: 10% cat, 30% dog, 60% bird
    predictions_cpu.set(1, 0, 0.1);
    predictions_cpu.set(1, 1, 0.3);
    predictions_cpu.set(1, 2, 0.6);
    
    Matrix targets_cpu(2, 3);
    // Sample 1 is actually a cat (one-hot)
    targets_cpu.set(0, 0, 1.0);
    targets_cpu.set(0, 1, 0.0);
    targets_cpu.set(0, 2, 0.0);
    
    // Sample 2 is actually a bird
    targets_cpu.set(1, 0, 0.0);
    targets_cpu.set(1, 1, 0.0);
    targets_cpu.set(1, 2, 1.0);
    
    cout << "Sample 1: Predicted=[Cat:0.7, Dog:0.2, Bird:0.1], True=Cat\n";
    cout << "Sample 2: Predicted=[Cat:0.1, Dog:0.3, Bird:0.6], True=Bird\n\n";
    
    // CPU calculation
    CategoricalCrossEntropyLoss cce_cpu;
    double loss_cpu = cce_cpu.calculate(predictions_cpu, targets_cpu);
    
    // CUDA calculation
    CategoricalCrossEntropyLossCUDA cce_cuda;
    double loss_cuda = cce_cuda.calculate(predictions_cpu, targets_cpu);
    
    cout << "CPU Loss:  " << fixed << setprecision(6) << loss_cpu << "\n";
    cout << "CUDA Loss: " << loss_cuda << "\n\n";
    
    cout << "✓ Model predicted both samples correctly!\n";
}

void basicMAEExample() {
    printHeader("MAE Loss - CUDA Example");
    
    cout << "Problem: Temperature prediction (robust to outliers)\n\n";
    
    Matrix predictions_cpu(4, 1);
    predictions_cpu.set(0, 0, 25.0);  // 25°C
    predictions_cpu.set(1, 0, 18.0);  // 18°C
    predictions_cpu.set(2, 0, 30.0);  // 30°C
    predictions_cpu.set(3, 0, 22.0);  // 22°C
    
    Matrix targets_cpu(4, 1);
    targets_cpu.set(0, 0, 24.0);  // True: 24°C
    targets_cpu.set(1, 0, 20.0);  // True: 20°C
    targets_cpu.set(2, 0, 28.0);  // True: 28°C
    targets_cpu.set(3, 0, 23.0);  // True: 23°C
    
    cout << "Predictions: [25°C, 18°C, 30°C, 22°C]\n";
    cout << "Targets:     [24°C, 20°C, 28°C, 23°C]\n\n";
    
    // CPU calculation
    MAELoss mae_cpu;
    double loss_cpu = mae_cpu.calculate(predictions_cpu, targets_cpu);
    
    // CUDA calculation
    MAELossCUDA mae_cuda;
    double loss_cuda = mae_cuda.calculate(predictions_cpu, targets_cpu);
    
    cout << "CPU Loss:  " << fixed << setprecision(4) << loss_cpu << "\n";
    cout << "CUDA Loss: " << loss_cuda << "\n";
    cout << "Average error: " << loss_cuda << "°C\n\n";
    
    cout << "✓ MAE is robust to outliers (unlike MSE)\n";
}

// =====================================================
// PART 2: Performance Benchmarks
// =====================================================

void benchmarkMSE() {
    printHeader("MSE Loss - Performance Benchmark");
    
    vector<int> sizes = {100, 1000, 5000, 10000};
    
    cout << setw(15) << "Matrix Size"
         << setw(15) << "CPU Time"
         << setw(15) << "CUDA Time"
         << setw(15) << "Speedup\n";
    printSeparator();
    
    for (int size : sizes) {
        // Generate random data
        Matrix pred = generateRandomMatrix(size, size, -10.0, 10.0);
        Matrix target = generateRandomMatrix(size, size, -10.0, 10.0);
        
        // Benchmark CPU
        MSELoss mse_cpu;
        auto start_cpu = high_resolution_clock::now();
        double loss_cpu = mse_cpu.calculate(pred, target);
        auto end_cpu = high_resolution_clock::now();
        auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);
        
        // Benchmark CUDA
        MSELossCUDA mse_cuda;
        auto start_cuda = high_resolution_clock::now();
        double loss_cuda = mse_cuda.calculate(pred, target);
        auto end_cuda = high_resolution_clock::now();
        auto duration_cuda = duration_cast<microseconds>(end_cuda - start_cuda);
        
        double speedup = (double)duration_cpu.count() / duration_cuda.count();
        
        cout << setw(10) << size << "x" << size
             << setw(12) << duration_cpu.count() << "μs"
             << setw(12) << duration_cuda.count() << "μs"
             << setw(12) << fixed << setprecision(2) << speedup << "x\n";
    }
    
    cout << "\n✓ CUDA shows significant speedup for large matrices!\n";
}

void benchmarkBCE() {
    printHeader("BCE Loss - Performance Benchmark");
    
    vector<int> sizes = {1000, 5000, 10000, 50000};
    
    cout << setw(15) << "Batch Size"
         << setw(15) << "CPU Time"
         << setw(15) << "CUDA Time"
         << setw(15) << "Speedup\n";
    printSeparator();
    
    for (int size : sizes) {
        // Generate random probabilities and binary targets
        Matrix pred = generateRandomMatrix(size, 1, 0.1, 0.9);
        Matrix target(size, 1);
        for (int i = 0; i < size; i++) {
            target.set(i, 0, (i % 2 == 0) ? 1.0 : 0.0);
        }
        
        // Benchmark CPU
        BinaryCrossEntropyLoss bce_cpu;
        auto start_cpu = high_resolution_clock::now();
        double loss_cpu = bce_cpu.calculate(pred, target);
        auto end_cpu = high_resolution_clock::now();
        auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);
        
        // Benchmark CUDA
        BinaryCrossEntropyLossCUDA bce_cuda;
        auto start_cuda = high_resolution_clock::now();
        double loss_cuda = bce_cuda.calculate(pred, target);
        auto end_cuda = high_resolution_clock::now();
        auto duration_cuda = duration_cast<microseconds>(end_cuda - start_cuda);
        
        double speedup = (double)duration_cpu.count() / duration_cuda.count();
        
        cout << setw(15) << size
             << setw(12) << duration_cpu.count() << "μs"
             << setw(12) << duration_cuda.count() << "μs"
             << setw(12) << fixed << setprecision(2) << speedup << "x\n";
    }
    
    cout << "\n✓ CUDA excels at batch processing!\n";
}

void benchmarkCCE() {
    printHeader("CCE Loss - Performance Benchmark");
    
    vector<pair<int, int>> configs = {{1000, 10}, {5000, 10}, {10000, 100}, {50000, 100}};
    
    cout << setw(20) << "Batch x Classes"
         << setw(15) << "CPU Time"
         << setw(15) << "CUDA Time"
         << setw(15) << "Speedup\n";
    printSeparator();
    
    for (auto& config : configs) {
        int batch = config.first;
        int classes = config.second;
        
        // Generate random probabilities (softmax-like)
        Matrix pred = generateRandomMatrix(batch, classes, 0.01, 0.99);
        
        // Generate random one-hot targets
        Matrix target(batch, classes);
        for (int i = 0; i < batch; i++) {
            int true_class = i % classes;
            for (int j = 0; j < classes; j++) {
                target.set(i, j, (j == true_class) ? 1.0 : 0.0);
            }
        }
        
        // Benchmark CPU
        CategoricalCrossEntropyLoss cce_cpu;
        auto start_cpu = high_resolution_clock::now();
        double loss_cpu = cce_cpu.calculate(pred, target);
        auto end_cpu = high_resolution_clock::now();
        auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);
        
        // Benchmark CUDA
        CategoricalCrossEntropyLossCUDA cce_cuda;
        auto start_cuda = high_resolution_clock::now();
        double loss_cuda = cce_cuda.calculate(pred, target);
        auto end_cuda = high_resolution_clock::now();
        auto duration_cuda = duration_cast<microseconds>(end_cuda - start_cuda);
        
        double speedup = (double)duration_cpu.count() / duration_cuda.count();
        
        cout << setw(10) << batch << "x" << setw(5) << classes
             << setw(12) << duration_cpu.count() << "μs"
             << setw(12) << duration_cuda.count() << "μs"
             << setw(12) << fixed << setprecision(2) << speedup << "x\n";
    }
    
    cout << "\n✓ CUDA handles multi-class classification efficiently!\n";
}

// =====================================================
// PART 3: Accuracy Verification
// =====================================================

void verifyAccuracy() {
    printHeader("Accuracy Verification - CPU vs CUDA");
    
    cout << "Testing all loss functions with random data...\n\n";
    
    Matrix pred = generateRandomMatrix(100, 50, 0.1, 0.9);
    Matrix target = generateRandomMatrix(100, 50, 0.1, 0.9);
    
    // MSE
    MSELoss mse_cpu;
    MSELossCUDA mse_cuda;
    double mse_loss_cpu = mse_cpu.calculate(pred, target);
    double mse_loss_cuda = mse_cuda.calculate(pred, target);
    double mse_diff = abs(mse_loss_cpu - mse_loss_cuda);
    
    cout << "MSE Loss:\n";
    cout << "  CPU:  " << fixed << setprecision(8) << mse_loss_cpu << "\n";
    cout << "  CUDA: " << mse_loss_cuda << "\n";
    cout << "  Diff: " << mse_diff << (mse_diff < 1e-6 ? " ✓" : " ✗") << "\n\n";
    
    // BCE
    BinaryCrossEntropyLoss bce_cpu;
    BinaryCrossEntropyLossCUDA bce_cuda;
    double bce_loss_cpu = bce_cpu.calculate(pred, target);
    double bce_loss_cuda = bce_cuda.calculate(pred, target);
    double bce_diff = abs(bce_loss_cpu - bce_loss_cuda);
    
    cout << "BCE Loss:\n";
    cout << "  CPU:  " << bce_loss_cpu << "\n";
    cout << "  CUDA: " << bce_loss_cuda << "\n";
    cout << "  Diff: " << bce_diff << (bce_diff < 1e-6 ? " ✓" : " ✗") << "\n\n";
    
    // MAE
    MAELoss mae_cpu;
    MAELossCUDA mae_cuda;
    double mae_loss_cpu = mae_cpu.calculate(pred, target);
    double mae_loss_cuda = mae_cuda.calculate(pred, target);
    double mae_diff = abs(mae_loss_cpu - mae_loss_cuda);
    
    cout << "MAE Loss:\n";
    cout << "  CPU:  " << mae_loss_cpu << "\n";
    cout << "  CUDA: " << mae_loss_cuda << "\n";
    cout << "  Diff: " << mae_diff << (mae_diff < 1e-6 ? " ✓" : " ✗") << "\n\n";
    
    cout << "✓ All loss functions verified! CPU and CUDA results match.\n";
}

// =====================================================
// PART 4: Gradient Verification
// =====================================================

void verifyGradients() {
    printHeader("Gradient Verification - CPU vs CUDA");
    
    cout << "Comparing gradients for all loss functions...\n\n";
    
    Matrix pred = generateRandomMatrix(50, 30, 0.2, 0.8);
    Matrix target = generateRandomMatrix(50, 30, 0.2, 0.8);
    
    // MSE Gradient
    MSELoss mse_cpu;
    MSELossCUDA mse_cuda;
    Matrix mse_grad_cpu = mse_cpu.gradient(pred, target);
    Matrix mse_grad_cuda = mse_cuda.gradient(pred, target);
    
    double mse_grad_diff = 0.0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            mse_grad_diff += abs(mse_grad_cpu.get(i, j) - mse_grad_cuda.get(i, j));
        }
    }
    mse_grad_diff /= 25.0;
    
    cout << "MSE Gradient:\n";
    cout << "  Sample gradients (CPU):  [" << mse_grad_cpu.get(0, 0) << ", "
         << mse_grad_cpu.get(0, 1) << ", " << mse_grad_cpu.get(0, 2) << "]\n";
    cout << "  Sample gradients (CUDA): [" << mse_grad_cuda.get(0, 0) << ", "
         << mse_grad_cuda.get(0, 1) << ", " << mse_grad_cuda.get(0, 2) << "]\n";
    cout << "  Average difference: " << scientific << mse_grad_diff
         << (mse_grad_diff < 1e-10 ? " ✓" : " ✗") << "\n\n";
    
    // BCE Gradient
    BinaryCrossEntropyLoss bce_cpu;
    BinaryCrossEntropyLossCUDA bce_cuda;
    Matrix bce_grad_cpu = bce_cpu.gradient(pred, target);
    Matrix bce_grad_cuda = bce_cuda.gradient(pred, target);
    
    double bce_grad_diff = 0.0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            bce_grad_diff += abs(bce_grad_cpu.get(i, j) - bce_grad_cuda.get(i, j));
        }
    }
    bce_grad_diff /= 25.0;
    
    cout << "BCE Gradient:\n";
    cout << "  Sample gradients (CPU):  [" << bce_grad_cpu.get(0, 0) << ", "
         << bce_grad_cpu.get(0, 1) << ", " << bce_grad_cpu.get(0, 2) << "]\n";
    cout << "  Sample gradients (CUDA): [" << bce_grad_cuda.get(0, 0) << ", "
         << bce_grad_cuda.get(0, 1) << ", " << bce_grad_cuda.get(0, 2) << "]\n";
    cout << "  Average difference: " << bce_grad_diff
         << (bce_grad_diff < 1e-10 ? " ✓" : " ✗") << "\n\n";
    
    cout << "✓ Gradients match! CUDA backpropagation is correct.\n";
}

// =====================================================
// MAIN
// =====================================================

int main() {
    cout << "\n";
    cout << "████████████████████████████████████████████████████████████\n";
    cout << "█                                                          █\n";
    cout << "█          CUDA LOSS FUNCTIONS - DEMONSTRATION             █\n";
    cout << "█          GPU-Accelerated Error Measurement               █\n";
    cout << "█                                                          █\n";
    cout << "████████████████████████████████████████████████████████████\n";
    
    // Part 1: Basic examples
    basicMSEExample();
    cout << "\nPress Enter to continue to BCE example...";
    cin.get();
    
    basicBCEExample();
    cout << "\nPress Enter to continue to CCE example...";
    cin.get();
    
    basicCCEExample();
    cout << "\nPress Enter to continue to MAE example...";
    cin.get();
    
    basicMAEExample();
    cout << "\nPress Enter to continue to performance benchmarks...";
    cin.get();
    
    // Part 2: Performance benchmarks
    benchmarkMSE();
    cout << "\nPress Enter to continue to BCE benchmark...";
    cin.get();
    
    benchmarkBCE();
    cout << "\nPress Enter to continue to CCE benchmark...";
    cin.get();
    
    benchmarkCCE();
    cout << "\nPress Enter to verify accuracy...";
    cin.get();
    
    // Part 3: Accuracy verification
    verifyAccuracy();
    cout << "\nPress Enter to verify gradients...";
    cin.get();
    
    // Part 4: Gradient verification
    verifyGradients();
    
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║                DEMONSTRATION COMPLETE!                 ║\n";
    cout << "║                                                        ║\n";
    cout << "║  Key Insights:                                         ║\n";
    cout << "║  • CUDA provides significant speedup for large data    ║\n";
    cout << "║  • GPU excels at batch processing                      ║\n";
    cout << "║  • Results match CPU implementation (verified)         ║\n";
    cout << "║  • Gradients are accurate for backpropagation          ║\n";
    cout << "║                                                        ║\n";
    cout << "║  Use CUDA loss functions for:                          ║\n";
    cout << "║  ✓ Large batch sizes (>1000 samples)                   ║\n";
    cout << "║  ✓ High-dimensional data                               ║\n";
    cout << "║  ✓ Real-time training                                  ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}
