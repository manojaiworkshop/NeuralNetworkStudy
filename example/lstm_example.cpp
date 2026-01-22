/**
 * @file lstm_example.cpp
 * @brief Comprehensive LSTM demonstration comparing RNN, LSTM, and GRU
 * 
 * This example demonstrates:
 * 1. Simple sequence prediction task
 * 2. Why RNN fails on long sequences
 * 3. How LSTM solves the problem
 * 4. GRU as a simpler alternative
 * 5. Performance comparison
 */

#include "../include/nn/rnn.h"
#include "../include/nn/lstm.h"
#include "../include/nn/gru.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

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
    std::cout << "\n" << BOLD << CYAN;
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(56) << std::left << title << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝";
    std::cout << RESET << "\n\n";
}

void printSection(const std::string& title) {
    std::cout << "\n" << BOLD << YELLOW << "─── " << title << " ───" << RESET << "\n\n";
}

/**
 * @brief Create a sequence memorization task
 * 
 * Task: Given a sequence of numbers, predict the next number
 * Example: [1, 2, 3, 4] → predict 5
 */
void createSequenceData(int seq_length, int num_samples,
                       std::vector<std::vector<Matrix>>& sequences,
                       std::vector<Matrix>& targets) {
    sequences.clear();
    targets.clear();
    
    for (int s = 0; s < num_samples; ++s) {
        std::vector<Matrix> sequence;
        
        // Create a simple counting sequence
        for (int t = 0; t < seq_length; ++t) {
            Matrix input(1, 1);
            input.set(0, 0, (s + t) / 10.0);  // Normalized values
            sequence.push_back(input);
        }
        
        // Target is the next number in sequence
        Matrix target(1, 1);
        target.set(0, 0, (s + seq_length) / 10.0);
        
        sequences.push_back(sequence);
        targets.push_back(target);
    }
}

/**
 * @brief Example 1: Understanding the Problem - Why vanilla RNN struggles
 */
void example1_VanillaRNNLimitations() {
    printHeader("EXAMPLE 1: Vanilla RNN Limitations");
    
    std::cout << "Task: Learn to predict the next number in a sequence\n";
    std::cout << "Example: [0.0, 0.1, 0.2, 0.3] → 0.4\n\n";
    
    std::cout << BOLD << "WHY RNN STRUGGLES:" << RESET << "\n";
    std::cout << "• Vanishing gradients over long sequences\n";
    std::cout << "• h(t) = tanh(W·x + W·h(t-1))\n";
    std::cout << "• Gradient: ∏ tanh'(x) where each term < 1\n";
    std::cout << "• After 20 steps: 0.5^20 ≈ 0.00000095 (vanishes!)\n\n";
    
    printSection("Training Simple RNN (Short Sequence)");
    
    // Create short sequence data (5 steps)
    std::vector<std::vector<Matrix>> sequences;
    std::vector<Matrix> targets;
    createSequenceData(5, 10, sequences, targets);
    
    std::cout << "Sequence length: 5 steps (manageable for RNN)\n";
    std::cout << "Training samples: 10\n\n";
    
    RNNNetwork rnn;
    rnn.addLayer(new RNNLayer(1, 8, 1, false, new Tanh(), new Linear()));
    
    std::cout << GREEN << "Training RNN..." << RESET << "\n";
    rnn.train(sequences, targets, 50, 0.05, false);
    
    // Test
    std::cout << "\n" << BOLD << "Test Results:" << RESET << "\n";
    for (int i = 0; i < 3; ++i) {
        Matrix pred = rnn.predict(sequences[i]);
        std::cout << "  Input sequence → Target: " << targets[i].get(0, 0) 
                  << ", Predicted: " << pred.get(0, 0) << "\n";
    }
    std::cout << GREEN << "✓ RNN works well on short sequences!" << RESET << "\n";
}

/**
 * @brief Example 2: LSTM to the Rescue
 */
void example2_LSTMAdvantages() {
    printHeader("EXAMPLE 2: LSTM Solves Long-Term Dependencies");
    
    std::cout << BOLD << "LSTM ARCHITECTURE:" << RESET << "\n\n";
    std::cout << "  4 Components:\n";
    std::cout << "  1. FORGET GATE (f): What to remove from memory\n";
    std::cout << "     f(t) = σ(W_f·[h,x] + b_f)\n\n";
    std::cout << "  2. INPUT GATE (i): What to add to memory\n";
    std::cout << "     i(t) = σ(W_i·[h,x] + b_i)\n";
    std::cout << "     C̃(t) = tanh(W_c·[h,x] + b_c)\n\n";
    std::cout << "  3. CELL STATE (C): The memory highway\n";
    std::cout << "     " << CYAN << "C(t) = f(t)⊙C(t-1) + i(t)⊙C̃(t)" << RESET << "\n";
    std::cout << "     ↑ Mostly ADDITION (not multiplication)\n";
    std::cout << "     → Gradient preserved!\n\n";
    std::cout << "  4. OUTPUT GATE (o): What to output\n";
    std::cout << "     o(t) = σ(W_o·[h,x] + b_o)\n";
    std::cout << "     h(t) = o(t)⊙tanh(C(t))\n\n";
    
    printSection("Training LSTM (Long Sequence)");
    
    // Create longer sequence data (20 steps)
    std::vector<std::vector<Matrix>> sequences;
    std::vector<Matrix> targets;
    createSequenceData(20, 15, sequences, targets);
    
    std::cout << "Sequence length: 20 steps (challenging!)\n";
    std::cout << "Training samples: 15\n\n";
    
    LSTMNetwork lstm;
    lstm.addLayer(new LSTMLayer(1, 16, 1, false, new Linear()));
    
    std::cout << GREEN << "Training LSTM..." << RESET << "\n";
    lstm.train(sequences, targets, 100, 0.01, true);
    
    // Test
    std::cout << "\n" << BOLD << "Test Results:" << RESET << "\n";
    for (int i = 0; i < 3; ++i) {
        Matrix pred = lstm.predict(sequences[i]);
        std::cout << "  Sequence → Target: " << std::fixed << std::setprecision(3)
                  << targets[i].get(0, 0) 
                  << ", Predicted: " << pred.get(0, 0)
                  << ", Error: " << std::abs(targets[i].get(0, 0) - pred.get(0, 0))
                  << "\n";
    }
    std::cout << GREEN << "✓ LSTM handles long sequences effectively!" << RESET << "\n";
}

/**
 * @brief Example 3: GRU - Simpler Alternative
 */
void example3_GRUComparison() {
    printHeader("EXAMPLE 3: GRU - The Simpler Alternative");
    
    std::cout << BOLD << "GRU ARCHITECTURE:" << RESET << "\n\n";
    std::cout << "  3 Components (simpler than LSTM):\n";
    std::cout << "  1. RESET GATE (r): How much past to forget\n";
    std::cout << "     r(t) = σ(W_r·[h,x])\n\n";
    std::cout << "  2. UPDATE GATE (z): How much to update\n";
    std::cout << "     z(t) = σ(W_z·[h,x])\n\n";
    std::cout << "  3. CANDIDATE + OUTPUT:\n";
    std::cout << "     h̃(t) = tanh(W_h·[r⊙h,x])\n";
    std::cout << "     " << CYAN << "h(t) = (1-z)⊙h(t-1) + z⊙h̃(t)" << RESET << "\n\n";
    
    std::cout << BOLD << "KEY DIFFERENCES:" << RESET << "\n";
    std::cout << "  • NO separate cell state (simpler!)\n";
    std::cout << "  • 2 gates vs LSTM's 3 gates\n";
    std::cout << "  • ~25% fewer parameters\n";
    std::cout << "  • Faster training\n";
    std::cout << "  • Often similar performance to LSTM\n\n";
    
    printSection("Training GRU");
    
    // Create sequence data
    std::vector<std::vector<Matrix>> sequences;
    std::vector<Matrix> targets;
    createSequenceData(20, 15, sequences, targets);
    
    std::cout << "Sequence length: 20 steps\n";
    std::cout << "Training samples: 15\n\n";
    
    GRUNetwork gru;
    gru.addLayer(new GRULayer(1, 16, 1, false, new Linear()));
    
    std::cout << GREEN << "Training GRU..." << RESET << "\n";
    gru.train(sequences, targets, 100, 0.01, true);
    
    // Test
    std::cout << "\n" << BOLD << "Test Results:" << RESET << "\n";
    for (int i = 0; i < 3; ++i) {
        Matrix pred = gru.predict(sequences[i]);
        std::cout << "  Sequence → Target: " << std::fixed << std::setprecision(3)
                  << targets[i].get(0, 0) 
                  << ", Predicted: " << pred.get(0, 0)
                  << ", Error: " << std::abs(targets[i].get(0, 0) - pred.get(0, 0))
                  << "\n";
    }
    std::cout << GREEN << "✓ GRU also handles long sequences well!" << RESET << "\n";
}

/**
 * @brief Example 4: Head-to-Head Comparison
 */
void example4_Comparison() {
    printHeader("EXAMPLE 4: Performance Comparison");
    
    std::cout << BOLD << "┌────────────┬──────────┬─────────────┬──────────────┬─────────────┐\n";
    std::cout << "│ Model      │ Gates    │ Parameters  │ Memory       │ Best For    │\n";
    std::cout << "├────────────┼──────────┼─────────────┼──────────────┼─────────────┤\n";
    std::cout << "│ RNN        │ 0        │ 3h²+3ih+3h  │ h(t)         │ Short seq   │\n";
    std::cout << "│            │          │ (baseline)  │              │ < 10 steps  │\n";
    std::cout << "├────────────┼──────────┼─────────────┼──────────────┼─────────────┤\n";
    std::cout << "│ GRU        │ 2        │ 6h²+6ih+6h  │ h(t)         │ Medium seq  │\n";
    std::cout << "│            │ (r, z)   │ (2× RNN)    │              │ 10-100 step │\n";
    std::cout << "├────────────┼──────────┼─────────────┼──────────────┼─────────────┤\n";
    std::cout << "│ LSTM       │ 3        │ 8h²+8ih+8h  │ h(t) + C(t)  │ Long seq    │\n";
    std::cout << "│            │ (f,i,o)  │ (2.67× RNN) │              │ > 100 steps │\n";
    std::cout << "└────────────┴──────────┴─────────────┴──────────────┴─────────────┘" << RESET << "\n\n";
    
    std::cout << BOLD << "PARAMETER COMPARISON (hidden_size=16, input_size=1):" << RESET << "\n";
    int h = 16, i = 1;
    int rnn_params = 3*h*h + 3*i*h + 3*h;
    int gru_params = 6*h*h + 6*i*h + 6*h;
    int lstm_params = 8*h*h + 8*i*h + 8*h;
    
    std::cout << "  RNN:  " << rnn_params << " parameters\n";
    std::cout << "  GRU:  " << gru_params << " parameters (+" 
              << ((gru_params - rnn_params) * 100 / rnn_params) << "%)\n";
    std::cout << "  LSTM: " << lstm_params << " parameters (+" 
              << ((lstm_params - rnn_params) * 100 / rnn_params) << "%)\n\n";
    
    std::cout << BOLD << "WHEN TO USE EACH:" << RESET << "\n\n";
    
    std::cout << RED << "❌ RNN:" << RESET << "\n";
    std::cout << "   • Short sequences (< 10 time steps)\n";
    std::cout << "   • Simple temporal patterns\n";
    std::cout << "   • Baseline/prototype\n";
    std::cout << "   • Real-time with minimal computation\n\n";
    
    std::cout << YELLOW << "⚡ GRU:" << RESET << "\n";
    std::cout << "   • Medium sequences (10-100 time steps)\n";
    std::cout << "   • Limited training data\n";
    std::cout << "   • Faster training needed\n";
    std::cout << "   • Text generation, sentiment analysis\n";
    std::cout << "   • Mobile/edge deployment\n\n";
    
    std::cout << GREEN << "✓ LSTM:" << RESET << "\n";
    std::cout << "   • Long sequences (> 100 time steps)\n";
    std::cout << "   • Complex temporal dependencies\n";
    std::cout << "   • Large dataset available\n";
    std::cout << "   • Speech recognition, machine translation\n";
    std::cout << "   • Time series with multiple patterns\n\n";
}

/**
 * @brief Example 5: Real-World Applications
 */
void example5_Applications() {
    printHeader("EXAMPLE 5: Real-World Applications");
    
    std::cout << BOLD << "1. TEXT GENERATION" << RESET << "\n";
    std::cout << "   Architecture: Character-level LSTM\n";
    std::cout << "   Input:  \"Hello wor\"\n";
    std::cout << "   Output: \"ld\"\n";
    std::cout << "   Use case: Auto-complete, chatbots\n\n";
    
    std::cout << BOLD << "2. SENTIMENT ANALYSIS" << RESET << "\n";
    std::cout << "   Architecture: GRU (efficient for text)\n";
    std::cout << "   Input:  \"This movie was amazing!\"\n";
    std::cout << "   Output: Positive (0.95)\n";
    std::cout << "   Use case: Product reviews, social media\n\n";
    
    std::cout << BOLD << "3. MACHINE TRANSLATION" << RESET << "\n";
    std::cout << "   Architecture: Seq2Seq LSTM (Encoder-Decoder)\n";
    std::cout << "   Input:  \"Hello\" (English)\n";
    std::cout << "   Output: \"Bonjour\" (French)\n";
    std::cout << "   Use case: Google Translate\n\n";
    
    std::cout << BOLD << "4. SPEECH RECOGNITION" << RESET << "\n";
    std::cout << "   Architecture: Bidirectional LSTM\n";
    std::cout << "   Input:  Audio waveform\n";
    std::cout << "   Output: \"Hello world\"\n";
    std::cout << "   Use case: Siri, Alexa\n\n";
    
    std::cout << BOLD << "5. TIME SERIES PREDICTION" << RESET << "\n";
    std::cout << "   Architecture: Stacked LSTM/GRU\n";
    std::cout << "   Input:  Stock prices [100, 102, 98, 105]\n";
    std::cout << "   Output: 107 (predicted next price)\n";
    std::cout << "   Use case: Financial forecasting, weather\n\n";
    
    std::cout << BOLD << "6. VIDEO ANALYSIS" << RESET << "\n";
    std::cout << "   Architecture: CNN + LSTM\n";
    std::cout << "   Input:  Video frames (sequence of images)\n";
    std::cout << "   Output: Action classification (\"running\")\n";
    std::cout << "   Use case: Security, sports analytics\n\n";
}

int main() {
    std::cout << BOLD << CYAN;
    std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                               ║\n";
    std::cout << "║       RECURRENT NEURAL NETWORKS: RNN vs LSTM vs GRU          ║\n";
    std::cout << "║              Complete Tutorial and Comparison                 ║\n";
    std::cout << "║                                                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝";
    std::cout << RESET << "\n";
    
    try {
        example1_VanillaRNNLimitations();
        example2_LSTMAdvantages();
        example3_GRUComparison();
        example4_Comparison();
        example5_Applications();
        
        std::cout << "\n" << BOLD << GREEN;
        std::cout << "═══════════════════════════════════════════════════════════════\n";
        std::cout << "                    TUTORIAL COMPLETE!                         \n";
        std::cout << "═══════════════════════════════════════════════════════════════";
        std::cout << RESET << "\n\n";
        
        std::cout << "Key Takeaways:\n";
        std::cout << "• RNN: Simple but suffers from vanishing gradients\n";
        std::cout << "• LSTM: Complex but handles long sequences (3 gates + cell)\n";
        std::cout << "• GRU: Sweet spot - simpler than LSTM, better than RNN (2 gates)\n";
        std::cout << "• Choose based on: sequence length, data size, compute budget\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}
