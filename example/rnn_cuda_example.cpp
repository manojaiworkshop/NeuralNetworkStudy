/**
 * ═══════════════════════════════════════════════════════════════════════════
 * CUDA-ACCELERATED RNN DEMONSTRATION
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * This example demonstrates:
 * 1. GPU-accelerated RNN processing
 * 2. Performance comparison: CPU vs GPU
 * 3. Batch processing advantages on GPU
 * 4. Memory transfer optimization
 */

#include "../include/nn/rnn.h"
#include "../include/nn/rnn_cuda.h"
#include "../include/nn/matrix.h"
#include "../include/nn/matrix_cuda.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// ANSI Colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

class Timer {
private:
    high_resolution_clock::time_point start_time;
public:
    void start() { start_time = high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_time).count() / 1000.0;
    }
};

void printHeader(const string& title) {
    cout << "\n" << BOLD << CYAN;
    cout << "╔══════════════════════════════════════════════════════════════╗\n";
    cout << "║  " << setw(58) << left << title << "  ║\n";
    cout << "╚══════════════════════════════════════════════════════════════╝";
    cout << RESET << "\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 1: Basic CUDA RNN
// ═══════════════════════════════════════════════════════════════════════════

void example1_BasicCUDARNN() {
    printHeader("EXAMPLE 1: Basic CUDA RNN Operation");
    
    cout << "Creating RNN on GPU...\n\n";
    
    // Create CUDA RNN
    RNNLayerCUDA rnn_cuda(2, 4, 1, false, new TanhCUDA(), new LinearCUDA());
    
    cout << GREEN << "✓ CUDA RNN Layer created\n" << RESET;
    cout << "  Input size:  2\n";
    cout << "  Hidden size: 4\n";
    cout << "  Output size: 1\n";
    cout << "  Parameters:  " << rnn_cuda.getParameterCount() << "\n\n";
    
    cout << "Creating input sequence...\n";
    
    vector<MatrixCUDA> sequence;
    for (int t = 0; t < 5; ++t) {
        Matrix input_cpu(1, 2);
        input_cpu.set(0, 0, 0.1 * t);
        input_cpu.set(0, 1, 0.2 * t);
        
        MatrixCUDA input_cuda(input_cpu);
        input_cuda.toGPU();
        sequence.push_back(input_cuda);
        
        cout << "  t=" << t << ": [" << input_cpu.get(0,0) << ", " 
             << input_cpu.get(0,1) << "]\n";
    }
    
    cout << "\n" << YELLOW << "Transferring to GPU and processing...\n" << RESET;
    
    Timer timer;
    timer.start();
    
    MatrixCUDA output = rnn_cuda.forward(sequence);
    
    double gpu_time = timer.elapsed_ms();
    
    // Transfer back to CPU for display
    Matrix output_cpu = static_cast<Matrix>(output);
    
    cout << "\nOutput: " << output_cpu.get(0, 0) << "\n";
    cout << "GPU Processing time: " << fixed << setprecision(3) 
         << gpu_time << " ms\n\n";
    
    cout << GREEN << "✓ CUDA RNN successfully processed sequence!\n" << RESET;
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 2: CPU vs GPU Performance Comparison
// ═══════════════════════════════════════════════════════════════════════════

void example2_PerformanceComparison() {
    printHeader("EXAMPLE 2: CPU vs GPU Performance Comparison");
    
    cout << "Testing with different sequence lengths and hidden sizes...\n\n";
    
    struct TestConfig {
        int seq_length;
        int hidden_size;
        int batch_size;
    };
    
    vector<TestConfig> configs = {
        {10, 32, 8},
        {50, 64, 16},
        {100, 128, 32}
    };
    
    cout << "┌─────────┬─────────┬───────────┬──────────┬──────────┬─────────┐\n";
    cout << "│ Seq Len │ Hidden  │ Batch Size│ CPU (ms) │ GPU (ms) │ Speedup │\n";
    cout << "├─────────┼─────────┼───────────┼──────────┼──────────┼─────────┤\n";
    
    for (const auto& config : configs) {
        // Generate sequence
        vector<Matrix> seq_cpu;
        vector<MatrixCUDA> seq_gpu;
        
        for (int t = 0; t < config.seq_length; ++t) {
            Matrix input(config.batch_size, 2);
            for (int b = 0; b < config.batch_size; ++b) {
                input.set(b, 0, sin(t * 0.1 + b));
                input.set(b, 1, cos(t * 0.1 + b));
            }
            seq_cpu.push_back(input);
            
            MatrixCUDA input_cuda(input);
            input_cuda.toGPU();
            seq_gpu.push_back(input_cuda);
        }
        
        // CPU version
        RNNLayer rnn_cpu(2, config.hidden_size, 1, false, new Tanh(), new Linear());
        
        Timer timer;
        timer.start();
        Matrix out_cpu = rnn_cpu.forward(seq_cpu);
        double cpu_time = timer.elapsed_ms();
        
        // GPU version
        RNNLayerCUDA rnn_gpu(2, config.hidden_size, 1, false, 
                             new TanhCUDA(), new LinearCUDA());
        
        timer.start();
        MatrixCUDA out_gpu = rnn_gpu.forward(seq_gpu);
        double gpu_time = timer.elapsed_ms();
        
        double speedup = cpu_time / gpu_time;
        
        cout << "│ " << setw(7) << config.seq_length 
             << " │ " << setw(7) << config.hidden_size
             << " │ " << setw(9) << config.batch_size
             << " │ " << setw(8) << fixed << setprecision(2) << cpu_time
             << " │ " << setw(8) << gpu_time
             << " │ " << setw(7) << setprecision(1) << speedup << "x │\n";
    }
    
    cout << "└─────────┴─────────┴───────────┴──────────┴──────────┴─────────┘\n\n";
    
    cout << CYAN << "KEY OBSERVATIONS:\n" << RESET;
    cout << "• GPU speedup increases with larger sequences\n";
    cout << "• Larger hidden sizes benefit more from parallelization\n";
    cout << "• Larger batch sizes maximize GPU utilization\n";
    cout << "• Small sequences may be faster on CPU (overhead)\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 3: Batch Processing Advantages
// ═══════════════════════════════════════════════════════════════════════════

void example3_BatchProcessing() {
    printHeader("EXAMPLE 3: Batch Processing on GPU");
    
    cout << "Demonstrating GPU's advantage with large batches...\n\n";
    
    const int seq_length = 20;
    const int hidden_size = 64;
    
    cout << "┌────────────┬──────────┬──────────┬─────────┐\n";
    cout << "│ Batch Size │ CPU (ms) │ GPU (ms) │ Speedup │\n";
    cout << "├────────────┼──────────┼──────────┼─────────┤\n";
    
    for (int batch_size : {1, 4, 16, 64, 256}) {
        // Generate batch
        vector<Matrix> seq_cpu;
        vector<MatrixCUDA> seq_gpu;
        
        for (int t = 0; t < seq_length; ++t) {
            Matrix input(batch_size, 2);
            input.randomize(-1.0, 1.0);
            seq_cpu.push_back(input);
            
            MatrixCUDA input_cuda(input);
            input_cuda.toGPU();
            seq_gpu.push_back(input_cuda);
        }
        
        // CPU
        RNNLayer rnn_cpu(2, hidden_size, 1, false, new Tanh(), new Linear());
        Timer timer;
        timer.start();
        rnn_cpu.forward(seq_cpu);
        double cpu_time = timer.elapsed_ms();
        
        // GPU
        RNNLayerCUDA rnn_gpu(2, hidden_size, 1, false, 
                             new TanhCUDA(), new LinearCUDA());
        timer.start();
        rnn_gpu.forward(seq_gpu);
        double gpu_time = timer.elapsed_ms();
        
        double speedup = cpu_time / gpu_time;
        
        cout << "│ " << setw(10) << batch_size
             << " │ " << setw(8) << fixed << setprecision(2) << cpu_time
             << " │ " << setw(8) << gpu_time
             << " │ " << setw(7) << setprecision(1) << speedup << "x │\n";
    }
    
    cout << "└────────────┴──────────┴──────────┴─────────┘\n\n";
    
    cout << GREEN << "INSIGHT:\n" << RESET;
    cout << "GPU parallelism shines with large batches!\n";
    cout << "• Batch size 1:   Minimal speedup (sequential)\n";
    cout << "• Batch size 256: Maximum speedup (parallel)\n";
    cout << "• Real-world: Use batches of 32-256 for training\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 4: GPU Optimization Strategies
// ═══════════════════════════════════════════════════════════════════════════

void example4_OptimizationStrategies() {
    printHeader("EXAMPLE 4: GPU Optimization Strategies for RNNs");
    
    cout << BOLD << "1. MEMORY TRANSFER OPTIMIZATION\n" << RESET;
    cout << "   • Keep data on GPU as long as possible\n";
    cout << "   • Batch multiple sequences together\n";
    cout << "   • Use pinned memory for faster transfers\n";
    cout << "   • Overlap computation with transfers (streams)\n\n";
    
    cout << BOLD << "2. COMPUTATIONAL OPTIMIZATION\n" << RESET;
    cout << "   • Use optimized BLAS libraries (cuBLAS)\n";
    cout << "   • Fuse operations (combine multiple kernels)\n";
    cout << "   • Optimize thread block sizes\n";
    cout << "   • Use Tensor Cores for mixed precision\n\n";
    
    cout << BOLD << "3. MEMORY ACCESS PATTERNS\n" << RESET;
    cout << "   • Coalesce global memory access\n";
    cout << "   • Use shared memory for reused data\n";
    cout << "   • Minimize divergent branches\n";
    cout << "   • Optimize register usage\n\n";
    
    cout << BOLD << "4. BATCH SIZE CONSIDERATIONS\n" << RESET;
    cout << "   • Too small: Underutilizes GPU\n";
    cout << "   • Too large: Exceeds GPU memory\n";
    cout << "   • Sweet spot: 32-256 for most GPUs\n";
    cout << "   • Adjust based on sequence length\n\n";
    
    cout << BOLD << "5. SEQUENCE LENGTH STRATEGIES\n" << RESET;
    cout << "   • Long sequences: Split into chunks\n";
    cout << "   • Variable lengths: Pad to max length\n";
    cout << "   • Use attention for very long sequences\n";
    cout << "   • Consider truncated BPTT\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 5: Real-World Application Scenarios
// ═══════════════════════════════════════════════════════════════════════════

void example5_RealWorldScenarios() {
    printHeader("EXAMPLE 5: When to Use CPU vs GPU RNNs");
    
    cout << BOLD << GREEN << "USE GPU WHEN:\n" << RESET;
    cout << "  ✓ Large batch sizes (>32)\n";
    cout << "  ✓ Long sequences (>50 time steps)\n";
    cout << "  ✓ Large hidden sizes (>128)\n";
    cout << "  ✓ Training (many iterations)\n";
    cout << "  ✓ Production inference at scale\n";
    cout << "  ✓ Real-time processing of multiple streams\n\n";
    
    cout << "  Examples:\n";
    cout << "  • Training language models\n";
    cout << "  • Video processing (many frames)\n";
    cout << "  • High-frequency trading (multiple stocks)\n";
    cout << "  • Speech recognition (production)\n\n";
    
    cout << BOLD << YELLOW << "USE CPU WHEN:\n" << RESET;
    cout << "  ✓ Small batch sizes (1-4)\n";
    cout << "  ✓ Short sequences (<20 steps)\n";
    cout << "  ✓ Small hidden sizes (<32)\n";
    cout << "  ✓ Single prediction (inference)\n";
    cout << "  ✓ Limited GPU memory\n";
    cout << "  ✓ Development/debugging\n\n";
    
    cout << "  Examples:\n";
    cout << "  • Chatbot single user response\n";
    cout << "  • Mobile device inference\n";
    cout << "  • Prototype development\n";
    cout << "  • Edge computing devices\n\n";
    
    cout << BOLD << CYAN << "HYBRID APPROACH:\n" << RESET;
    cout << "  • Train on GPU, deploy on CPU/edge\n";
    cout << "  • Use quantization for deployment\n";
    cout << "  • Model compression techniques\n";
    cout << "  • Knowledge distillation\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    cout << BOLD << MAGENTA;
    cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    cout << "║                                                                ║\n";
    cout << "║            CUDA-ACCELERATED RNN DEMONSTRATION                  ║\n";
    cout << "║                                                                ║\n";
    cout << "║     GPU vs CPU Performance & Optimization Strategies           ║\n";
    cout << "║                                                                ║\n";
    cout << "╚════════════════════════════════════════════════════════════════╝\n";
    cout << RESET << "\n";
    
    try {
        // Print GPU info
        cout << CYAN << "Checking CUDA device...\n" << RESET;
        MatrixCUDA::printDeviceInfo();
        cout << "\n";
        
        example1_BasicCUDARNN();
        example2_PerformanceComparison();
        example3_BatchProcessing();
        example4_OptimizationStrategies();
        example5_RealWorldScenarios();
        
        cout << GREEN << BOLD;
        cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        cout << "║  ✓ All CUDA RNN examples completed!                            ║\n";
        cout << "║                                                                ║\n";
        cout << "║  Key Takeaways:                                                ║\n";
        cout << "║  • GPU provides 5-50x speedup for large sequences/batches     ║\n";
        cout << "║  • Batch processing is crucial for GPU efficiency             ║\n";
        cout << "║  • Memory transfer is a significant bottleneck                ║\n";
        cout << "║  • Choose CPU or GPU based on your specific use case          ║\n";
        cout << "╚════════════════════════════════════════════════════════════════╝\n";
        cout << RESET << "\n";
        
    } catch (const exception& e) {
        cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
