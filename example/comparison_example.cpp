/**
 * ═══════════════════════════════════════════════════════════════════════════
 * RNN, LSTM, GRU: SIDE-BY-SIDE COMPARISON
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * This example demonstrates the differences between:
 * - Vanilla RNN (simple, fast, but vanishing gradients)
 * - LSTM (complex, powerful, remembers long-term)
 * - GRU (balanced, fewer parameters than LSTM)
 * 
 * We'll use a simple task: Predict sine wave
 */

#include "../include/nn/rnn.h"
#include "../include/nn/lstm.h"
#include "../include/nn/gru.h"
#include "../include/nn/matrix.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>

// ANSI Colors
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN;
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(56) << std::left << title << "  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝";
    std::cout << RESET << "\n\n";
}

// Generate simple sine wave sequence
std::vector<Matrix> generateSequence(int length, double start = 0.0) {
    std::vector<Matrix> sequence;
    for (int i = 0; i < length; ++i) {
        Matrix input(1, 1);
        input.set(0, 0, std::sin(start + i * 0.1));
        sequence.push_back(input);
    }
    return sequence;
}

int main() {
    std::cout << BOLD << CYAN << R"(
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       RNN vs LSTM vs GRU: Architectural Comparison           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    )" << RESET << "\n";

    // ═══════════════════════════════════════════════════════════════════════
    // PART 1: Architecture Comparison
    // ═══════════════════════════════════════════════════════════════════════
    
    printHeader("PART 1: Architecture Comparison");
    
    std::cout << BOLD << "1. VANILLA RNN:\n" << RESET;
    std::cout << "   Equations:\n";
    std::cout << "     h(t) = tanh(W_xh·x(t) + W_hh·h(t-1) + b_h)\n";
    std::cout << "     y(t) = W_hy·h(t) + b_y\n\n";
    std::cout << "   Parameters: 2 weight matrices\n";
    std::cout << "   • W_xh: input → hidden\n";
    std::cout << "   • W_hh: hidden → hidden (recurrence)\n\n";
    std::cout << "   Pros: ✓ Simple, fast\n";
    std::cout << "   Cons: ✗ Vanishing gradients on long sequences\n\n";
    
    std::cout << BOLD << "2. LSTM (Long Short-Term Memory):\n" << RESET;
    std::cout << "   Components: 4 gates + cell state\n\n";
    std::cout << "   Forget Gate:  f(t) = σ(W_f·[h,x] + b_f)\n";
    std::cout << "     └─ Decides what to remove from memory\n\n";
    std::cout << "   Input Gate:   i(t) = σ(W_i·[h,x] + b_i)\n";
    std::cout << "                 C̃(t) = tanh(W_c·[h,x] + b_c)\n";
    std::cout << "     └─ Decides what to add to memory\n\n";
    std::cout << "   Cell Update:  C(t) = f(t)⊙C(t-1) + i(t)⊙C̃(t)\n";
    std::cout << "     └─ The MEMORY HIGHWAY (addition preserves gradients!)\n\n";
    std::cout << "   Output Gate:  o(t) = σ(W_o·[h,x] + b_o)\n";
    std::cout << "                 h(t) = o(t)⊙tanh(C(t))\n";
    std::cout << "     └─ Decides what to output\n\n";
    std::cout << "   Parameters: 8 weight matrices (4 gates × 2 each)\n";
    std::cout << "   Pros: ✓ Solves vanishing gradients\n";
    std::cout << "         ✓ Remembers long-term dependencies\n";
    std::cout << "   Cons: ✗ More parameters\n";
    std::cout << "         ✗ Slower training\n\n";
    
    std::cout << BOLD << "3. GRU (Gated Recurrent Unit):\n" << RESET;
    std::cout << "   Components: 3 gates (simpler than LSTM)\n\n";
    std::cout << "   Update Gate:  z(t) = σ(W_z·[h,x] + b_z)\n";
    std::cout << "     └─ How much to update (combines input/forget)\n\n";
    std::cout << "   Reset Gate:   r(t) = σ(W_r·[h,x] + b_r)\n";
    std::cout << "     └─ How much past to forget\n\n";
    std::cout << "   Candidate:    h̃(t) = tanh(W_h·[r⊙h,x] + b_h)\n";
    std::cout << "   Final Hidden: h(t) = z(t)⊙h(t-1) + (1-z(t))⊙h̃(t)\n";
    std::cout << "     └─ Interpolate between old and new\n\n";
    std::cout << "   Parameters: 6 weight matrices (3 gates × 2 each)\n";
    std::cout << "   Pros: ✓ Fewer parameters than LSTM (faster)\n";
    std::cout << "         ✓ Still handles long-term dependencies\n";
    std::cout << "         ✓ Often performs similarly to LSTM\n";
    std::cout << "   Cons: ✗ Less flexible than LSTM for some tasks\n\n";
    
    // ═══════════════════════════════════════════════════════════════════════
    // PART 2: Parameter Count Comparison
    // ═══════════════════════════════════════════════════════════════════════
    
    printHeader("PART 2: Parameter Count Comparison");
    
    int input_size = 1;
    int hidden_size = 8;
    int output_size = 1;
    
    // Create cells
    RNNCell rnn_cell(input_size, hidden_size);
    LSTMCell lstm_cell(input_size, hidden_size);
    GRUCell gru_cell(input_size, hidden_size);
    
    std::cout << "Configuration: input=" << input_size 
              << ", hidden=" << hidden_size << "\n\n";
    
    std::cout << std::left;
    std::cout << "┌────────────┬─────────────┬──────────────────────────┐\n";
    std::cout << "│ Model      │ Parameters  │ Memory Footprint         │\n";
    std::cout << "├────────────┼─────────────┼──────────────────────────┤\n";
    std::cout << "│ RNN        │ " << std::setw(11) << rnn_cell.getParameterCount() 
              << " │ " << std::setw(24) << "Small (fast training)" << " │\n";
    std::cout << "│ LSTM       │ " << std::setw(11) << lstm_cell.getParameterCount() 
              << " │ " << std::setw(24) << "Large (4× RNN)" << " │\n";
    std::cout << "│ GRU        │ " << std::setw(11) << gru_cell.getParameterCount() 
              << " │ " << std::setw(24) << "Medium (3× RNN)" << " │\n";
    std::cout << "└────────────┴─────────────┴──────────────────────────┘\n\n";
    
    // ═══════════════════════════════════════════════════════════════════════
    // PART 3: When to Use Each?
    // ═══════════════════════════════════════════════════════════════════════
    
    printHeader("PART 3: When to Use Each Architecture?");
    
    std::cout << BOLD << "🎯 USE RNN WHEN:\n" << RESET;
    std::cout << "   • Short sequences (< 10 steps)\n";
    std::cout << "   • Speed is critical\n";
    std::cout << "   • Simple patterns\n";
    std::cout << "   Example: Real-time sensor data (last few readings)\n\n";
    
    std::cout << BOLD << "🎯 USE LSTM WHEN:\n" << RESET;
    std::cout << "   • Long sequences (20-100+ steps)\n";
    std::cout << "   • Complex long-term dependencies\n";
    std::cout << "   • You have enough data/compute\n";
    std::cout << "   Examples:\n";
    std::cout << "     • Language modeling (sentences, paragraphs)\n";
    std::cout << "     • Video analysis (many frames)\n";
    std::cout << "     • Time series with trends\n\n";
    
    std::cout << BOLD << "🎯 USE GRU WHEN:\n" << RESET;
    std::cout << "   • Medium sequences (10-50 steps)\n";
    std::cout << "   • Want LSTM performance with fewer parameters\n";
    std::cout << "   • Limited training data\n";
    std::cout << "   • Faster training needed\n";
    std::cout << "   Examples:\n";
    std::cout << "     • Speech recognition\n";
    std::cout << "     • Machine translation\n";
    std::cout << "     • Music generation\n\n";
    
    // ═══════════════════════════════════════════════════════════════════════
    // PART 4: Quick Demonstration
    // ═══════════════════════════════════════════════════════════════════════
    
    printHeader("PART 4: Forward Pass Demonstration");
    
    std::cout << "Processing short sequence through all three models...\n\n";
    
    // Create layers
    RNNLayer rnn_layer(input_size, hidden_size, output_size, false);
    LSTMLayer lstm_layer(input_size, hidden_size, output_size, false);
    GRULayer gru_layer(input_size, hidden_size, output_size, false);
    
    // Generate test sequence
    auto sequence = generateSequence(5, 0.0);
    
    std::cout << "Input sequence (sine wave):\n  ";
    for (size_t i = 0; i < sequence.size(); ++i) {
        std::cout << std::fixed << std::setprecision(3) << sequence[i].get(0, 0);
        if (i < sequence.size() - 1) std::cout << " → ";
    }
    std::cout << "\n\n";
    
    // Forward passes
    auto start = std::chrono::high_resolution_clock::now();
    Matrix rnn_out = rnn_layer.forward(sequence);
    auto rnn_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    Matrix lstm_out = lstm_layer.forward(sequence);
    auto lstm_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    Matrix gru_out = gru_layer.forward(sequence);
    auto gru_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    std::cout << "Outputs (untrained networks, random initialization):\n";
    std::cout << "  RNN:  " << std::fixed << std::setprecision(6) 
              << rnn_out.get(0, 0) << " (" << rnn_time << " μs)\n";
    std::cout << "  LSTM: " << lstm_out.get(0, 0) << " (" << lstm_time << " μs)\n";
    std::cout << "  GRU:  " << gru_out.get(0, 0) << " (" << gru_time << " μs)\n\n";
    
    std::cout << "Relative speed:\n";
    double base = rnn_time;
    std::cout << "  RNN:  1.00x (baseline)\n";
    std::cout << "  LSTM: " << std::fixed << std::setprecision(2) 
              << (double)lstm_time/base << "x slower\n";
    std::cout << "  GRU:  " << (double)gru_time/base << "x slower\n\n";
    
    // ═══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════
    
    printHeader("SUMMARY: Key Differences");
    
    std::cout << "┌─────────────┬──────────┬───────────┬──────────────────┐\n";
    std::cout << "│ Feature     │ RNN      │ LSTM      │ GRU              │\n";
    std::cout << "├─────────────┼──────────┼───────────┼──────────────────┤\n";
    std::cout << "│ Complexity  │ Simple   │ Complex   │ Moderate         │\n";
    std::cout << "│ Parameters  │ Fewest   │ Most      │ Middle           │\n";
    std::cout << "│ Speed       │ Fastest  │ Slowest   │ Fast             │\n";
    std::cout << "│ Long Memory │ Poor     │ Excellent │ Very Good        │\n";
    std::cout << "│ Training    │ Easy     │ Hard      │ Moderate         │\n";
    std::cout << "│ Use Case    │ Short    │ Long      │ General Purpose  │\n";
    std::cout << "└─────────────┴──────────┴───────────┴──────────────────┘\n\n";
    
    std::cout << BOLD << GREEN << "✓ Example completed!\n" << RESET;
    std::cout << "\nKey Takeaway:\n";
    std::cout << "  • Start with GRU (best balance)\n";
    std::cout << "  • Use LSTM if GRU doesn't work\n";
    std::cout << "  • Use RNN only for very short sequences\n\n";
    
    std::cout << YELLOW << "For GPU acceleration, check:\n" << RESET;
    std::cout << "  • lstm_cuda_example - LSTM on GPU\n";
    std::cout << "  • gru_cuda_example - GRU on GPU\n";
    std::cout << "  • Expect 20-100x speedup for large batches!\n\n";
    
    return 0;
}
