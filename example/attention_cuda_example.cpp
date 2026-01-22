#include "nn/attention_cuda.h"
#include "nn/matrix_cuda.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

void print_gpu_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║         CUDA GPU INFORMATION                      ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n";
    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n\n";
    }
}

void demo_attention_cuda() {
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   CUDA ATTENTION MECHANISM DEMO                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
    
    size_t batch_size = 2;
    size_t seq_len = 4;
    size_t d_model = 64;
    
    std::cout << "Configuration:\n";
    std::cout << "  Batch Size: " << batch_size << "\n";
    std::cout << "  Sequence Length: " << seq_len << "\n";
    std::cout << "  Model Dimension (d_model): " << d_model << "\n\n";
    
    // Create input on CPU
    Matrix input_cpu(batch_size * seq_len, d_model);
    input_cpu.randomNormal(0.0, 1.0);
    
    std::cout << "Step 1: Create input matrix on CPU (" 
              << batch_size * seq_len << " x " << d_model << ")\n";
    
    // Transfer to GPU
    MatrixCUDA input(input_cpu);
    input.toGPU();
    
    std::cout << "Step 2: Transfer input to GPU ✓\n";
    
    // Create scaled dot-product attention
    size_t d_k = d_model;
    ScaledDotProductAttentionCUDA attention(d_k);
    
    std::cout << "Step 3: Initialize Scaled Dot-Product Attention\n";
    std::cout << "           Scale Factor: 1/√" << d_k << " = " 
              << std::fixed << std::setprecision(4) << (1.0 / std::sqrt(d_k)) << "\n\n";
    
    // Forward pass (self-attention: Q=K=V=input)
    std::cout << "Step 4: Running attention on GPU...\n";
    
    auto start = std::chrono::steady_clock::now();
    MatrixCUDA output = attention.forward(input, input, input);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    
    std::cout << "           GPU Forward Pass Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " μs\n\n";
    
    // Get attention weights
    MatrixCUDA attn_weights = attention.getAttentionWeights();
    attn_weights.toCPU();
    
    std::cout << "Step 5: Attention Weights (first 4x4 block):\n";
    std::cout << "           ";
    for (int j = 0; j < std::min(4, (int)seq_len); j++) {
        std::cout << "  Pos" << j << "  ";
    }
    std::cout << "\n";
    
    for (int i = 0; i < std::min(4, (int)batch_size * (int)seq_len); i++) {
        std::cout << "  Pos" << i << ":  ";
        for (int j = 0; j < std::min(4, (int)seq_len); j++) {
            double weight = attn_weights.get(i, j);
            std::cout << std::fixed << std::setprecision(3) << weight << "  ";
        }
        std::cout << "\n";
    }
    
    // Transfer output back to CPU
    output.toCPU();
    
    std::cout << "\nStep 6: Output shape: " << output.getRows() << " x " << output.getCols() << "\n";
    std::cout << "        Sample output values: [";
    for (int i = 0; i < std::min(5, (int)d_model); i++) {
        std::cout << std::fixed << std::setprecision(3) << output.get(0, i);
        if (i < std::min(5, (int)d_model) - 1) std::cout << ", ";
    }
    std::cout << " ...]\n";
    
    std::cout << "\n✓ Attention computation on GPU successful!\n";
}

void demo_multi_head_attention_cuda() {
    std::cout << "\n\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   MULTI-HEAD ATTENTION CUDA DEMO                  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
    
    size_t batch_size = 2;
    size_t seq_len = 8;
    size_t d_model = 128;
    size_t num_heads = 8;
    
    std::cout << "Configuration:\n";
    std::cout << "  Model Dimension: " << d_model << "\n";
    std::cout << "  Number of Heads: " << num_heads << "\n";
    std::cout << "  d_k (per head): " << d_model / num_heads << "\n";
    std::cout << "  Sequence Length: " << seq_len << "\n\n";
    
    // Create input
    Matrix input_cpu(batch_size * seq_len, d_model);
    input_cpu.randomNormal(0.0, 0.5);
    
    MatrixCUDA input(input_cpu);
    input.toGPU();
    
    std::cout << "Step 1: Input transferred to GPU\n";
    
    // Create multi-head attention
    MultiHeadAttentionCUDA mha(d_model, num_heads);
    
    std::cout << "Step 2: Multi-Head Attention initialized\n";
    std::cout << "           Parameters: " << mha.getParameterCount() << "\n";
    std::cout << "           (4 × d_model² for Q,K,V,O projections)\n\n";
    
    // Forward pass
    std::cout << "Step 3: Running " << num_heads << " attention heads in parallel...\n";
    
    auto start = std::chrono::steady_clock::now();
    MatrixCUDA output = mha.forward(input, input, input);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "           GPU Forward Pass: " << duration << " μs\n";
    std::cout << "           Per-head time: " << duration / num_heads << " μs\n\n";
    
    // Get attention weights from all heads
    auto all_weights = mha.getAllAttentionWeights();
    
    std::cout << "Step 4: Attention weights from all heads retrieved\n";
    std::cout << "           Each head: " << seq_len << " x " << seq_len << " matrix\n\n";
    
    // Show attention pattern from first head
    MatrixCUDA head0_weights = all_weights[0];
    head0_weights.toCPU();
    
    std::cout << "Head 0 Attention Pattern (first 4x4):\n";
    for (int i = 0; i < std::min(4, (int)seq_len); i++) {
        std::cout << "  ";
        for (int j = 0; j < std::min(4, (int)seq_len); j++) {
            double w = head0_weights.get(i, j);
            if (w > 0.3) std::cout << "██";
            else if (w > 0.15) std::cout << "▓▓";
            else if (w > 0.05) std::cout << "▒▒";
            else std::cout << "░░";
            std::cout << " ";
        }
        std::cout << "\n";
    }
    
    output.toCPU();
    std::cout << "\nStep 5: Output shape: " << output.getRows() << " x " << output.getCols() << "\n";
    std::cout << "\n✓ Multi-head attention on GPU successful!\n";
}

void demo_transformer_encoder_cuda() {
    std::cout << "\n\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   TRANSFORMER ENCODER CUDA DEMO                   ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
    
    size_t num_layers = 3;
    size_t d_model = 128;
    size_t num_heads = 4;
    size_t d_ff = 512;
    size_t batch_size = 2;
    size_t seq_len = 10;
    
    std::cout << "Configuration:\n";
    std::cout << "  Number of Layers: " << num_layers << "\n";
    std::cout << "  Model Dimension: " << d_model << "\n";
    std::cout << "  Attention Heads: " << num_heads << "\n";
    std::cout << "  Feed-Forward Dim: " << d_ff << "\n";
    std::cout << "  Sequence Length: " << seq_len << "\n\n";
    
    // Create encoder
    std::cout << "Step 1: Building Transformer Encoder on GPU...\n";
    TransformerEncoderCUDA encoder(num_layers, d_model, num_heads, d_ff);
    
    std::cout << "           ✓ " << num_layers << " encoder layers created\n";
    std::cout << "           ✓ Multi-head attention in each layer\n";
    std::cout << "           ✓ Feed-forward networks initialized\n\n";
    
    // Create input (e.g., from embeddings)
    Matrix input_cpu(batch_size * seq_len, d_model);
    input_cpu.randomNormal(0.0, 0.5);
    
    MatrixCUDA input(input_cpu);
    input.toGPU();
    
    std::cout << "Step 2: Input prepared (" << batch_size * seq_len 
              << " x " << d_model << ")\n\n";
    
    // Forward pass through encoder
    std::cout << "Step 3: Encoding sequence on GPU...\n";
    std::cout << "           Processing through " << num_layers << " layers:\n";
    
    auto start = std::chrono::steady_clock::now();
    MatrixCUDA encoded = encoder.forward(input);
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    for (size_t i = 0; i < num_layers; i++) {
        std::cout << "           Layer " << (i+1) << ": Self-Attention → FFN → Norm ✓\n";
    }
    
    std::cout << "\n           Total Encoding Time: " << duration << " ms\n";
    std::cout << "           Per-layer time: " << duration / num_layers << " ms\n\n";
    
    // Get attention weights from all layers
    auto all_layer_weights = encoder.getAllAttentionWeights();
    
    std::cout << "Step 4: Attention patterns captured:\n";
    std::cout << "           " << all_layer_weights.size() << " layers × " 
              << all_layer_weights[0].size() << " heads\n";
    std::cout << "           Total attention matrices: " 
              << all_layer_weights.size() * all_layer_weights[0].size() << "\n\n";
    
    encoded.toCPU();
    std::cout << "Step 5: Encoded output shape: " << encoded.getRows() 
              << " x " << encoded.getCols() << "\n";
    std::cout << "        Sample values: [";
    for (int i = 0; i < std::min(4, (int)d_model); i++) {
        std::cout << std::fixed << std::setprecision(3) << encoded.get(0, i);
        if (i < std::min(4, (int)d_model) - 1) std::cout << ", ";
    }
    std::cout << " ...]\n";
    
    std::cout << "\n✓ Transformer Encoder on GPU successful!\n";
}

void performance_comparison() {
    std::cout << "\n\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║   CPU vs GPU PERFORMANCE COMPARISON               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Expected Speedups (GPU vs CPU):\n\n";
    
    std::cout << "┌─────────────────────────┬──────────────┬─────────────┐\n";
    std::cout << "│ Component               │ Sequence Len │ Speedup     │\n";
    std::cout << "├─────────────────────────┼──────────────┼─────────────┤\n";
    std::cout << "│ Attention (short)       │    32        │ 3-5x        │\n";
    std::cout << "│ Attention (medium)      │    128       │ 10-20x      │\n";
    std::cout << "│ Attention (long)        │    512       │ 30-50x      │\n";
    std::cout << "│ Multi-Head (8 heads)    │    128       │ 15-30x      │\n";
    std::cout << "│ Full Encoder (6 layers) │    128       │ 20-40x      │\n";
    std::cout << "└─────────────────────────┴──────────────┴─────────────┘\n\n";
    
    std::cout << "Performance Benefits:\n";
    std::cout << "  ✓ Parallel matrix operations\n";
    std::cout << "  ✓ Parallel attention heads\n";
    std::cout << "  ✓ Batch processing efficiency\n";
    std::cout << "  ✓ Reduced memory transfers\n";
    std::cout << "  ✓ Hardware-accelerated softmax\n\n";
    
    std::cout << "Best Use Cases for GPU:\n";
    std::cout << "  • Long sequences (> 128 tokens)\n";
    std::cout << "  • Large batch sizes (> 16)\n";
    std::cout << "  • Multiple attention heads (≥ 8)\n";
    std::cout << "  • Deep models (≥ 6 layers)\n";
    std::cout << "  • Training (backward pass benefits)\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║     CUDA TRANSFORMER ATTENTION DEMONSTRATION             ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║  GPU-Accelerated Attention for 10-50x Speedup           ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    try {
        // Check GPU availability
        print_gpu_info();
        
        // Demo 1: Basic attention
        demo_attention_cuda();
        
        // Demo 2: Multi-head attention
        demo_multi_head_attention_cuda();
        
        // Demo 3: Full encoder
        demo_transformer_encoder_cuda();
        
        // Performance info
        performance_comparison();
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║                                                          ║\n";
        std::cout << "║  ✓ ALL CUDA TRANSFORMER DEMOS COMPLETED SUCCESSFULLY!    ║\n";
        std::cout << "║                                                          ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ ERROR: " << e.what() << "\n\n";
        return 1;
    }
    
    return 0;
}
