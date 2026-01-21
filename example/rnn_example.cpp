/**
 * ═══════════════════════════════════════════════════════════════════════════
 * RNN (RECURRENT NEURAL NETWORK) COMPLETE DEMONSTRATION
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * This example explains and demonstrates:
 * 
 * 1. WHAT IS AN RNN?
 *    - A neural network designed for SEQUENTIAL DATA
 *    - Maintains "memory" through hidden states
 *    - Processes data step-by-step, considering context
 * 
 * 2. HOW RNNs DIFFER FROM FEEDFORWARD NETWORKS:
 *    
 *    FEEDFORWARD NETWORK:
 *    ┌─────┐    ┌─────┐    ┌─────┐
 *    │ X₁  │───▶│Layer│───▶│ Y₁  │  (Independent processing)
 *    └─────┘    └─────┘    └─────┘
 *    
 *    ┌─────┐    ┌─────┐    ┌─────┐
 *    │ X₂  │───▶│Layer│───▶│ Y₂  │  (No memory of X₁)
 *    └─────┘    └─────┘    └─────┘
 * 
 *    RECURRENT NETWORK:
 *    ┌─────┐    ┌─────┐    ┌─────┐
 *    │ X₁  │───▶│ RNN │───▶│ Y₁  │
 *    └─────┘    └──┬──┘    └─────┘
 *                  │ h₁ (memory)
 *                  ▼
 *    ┌─────┐    ┌─────┐    ┌─────┐
 *    │ X₂  │───▶│ RNN │───▶│ Y₂  │  (Uses memory h₁ from X₁)
 *    └─────┘    └──┬──┘    └─────┘
 *                  │ h₂
 *                  ▼
 * 
 * 3. WHEN TO USE RNNs:
 *    ✓ Time series prediction (stock prices, weather)
 *    ✓ Natural language processing (text generation, translation)
 *    ✓ Speech recognition
 *    ✓ Video analysis
 *    ✓ Music generation
 *    ✗ Static images (use CNN instead)
 *    ✗ Tabular data without order (use feedforward)
 * 
 * 4. EXAMPLES DEMONSTRATED:
 *    - Sine wave prediction (time series)
 *    - Sequence memorization
 *    - Pattern recognition in sequences
 */

#include "../include/nn/rnn.h"
#include "../include/nn/matrix.h"
#include "../include/nn/activation.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// ANSI Colors
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
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << std::setw(58) << std::left << title << "  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝";
    std::cout << RESET << "\n\n";
}

void printSubHeader(const std::string& title) {
    std::cout << "\n" << BOLD << YELLOW << "─── " << title << " ───" << RESET << "\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 1: Understanding RNN Architecture
// ═══════════════════════════════════════════════════════════════════════════

void example1_RNNArchitecture() {
    printHeader("EXAMPLE 1: Understanding RNN Architecture");
    
    std::cout << BOLD << "RNN EQUATIONS:\n" << RESET;
    std::cout << "At each time step t:\n";
    std::cout << "  1. h(t) = tanh(W_xh·x(t) + W_hh·h(t-1) + b_h)  [Hidden state update]\n";
    std::cout << "  2. y(t) = W_hy·h(t) + b_y                       [Output computation]\n\n";
    
    std::cout << "Where:\n";
    std::cout << "  • x(t)   = input at time t\n";
    std::cout << "  • h(t)   = hidden state (memory) at time t\n";
    std::cout << "  • h(t-1) = previous hidden state (memory from past)\n";
    std::cout << "  • y(t)   = output at time t\n";
    std::cout << "  • W_xh   = input-to-hidden weights\n";
    std::cout << "  • W_hh   = hidden-to-hidden weights (recurrent)\n";
    std::cout << "  • W_hy   = hidden-to-output weights\n\n";
    
    printSubHeader("Creating a Simple RNN");
    
    std::cout << "Let's create an RNN with:\n";
    std::cout << "  - Input size:  2 (e.g., [temperature, humidity])\n";
    std::cout << "  - Hidden size: 4 (memory capacity)\n";
    std::cout << "  - Output size: 1 (e.g., rainfall prediction)\n\n";
    
    RNNLayer rnn(2, 4, 1, false, new Tanh(), new Linear());
    
    std::cout << GREEN << "✓ RNN Layer created!\n" << RESET;
    std::cout << "  Total parameters: " << rnn.getParameterCount() << "\n";
    std::cout << "    = (2×4) + (4×4) + 4 + (1×4) + 1\n";
    std::cout << "    = 8 + 16 + 4 + 4 + 1 = 33 parameters\n\n";
    
    printSubHeader("Processing a Sequence");
    
    std::cout << "Input sequence (3 time steps):\n";
    std::vector<Matrix> sequence;
    
    // Time step 1
    Matrix x1(1, 2);
    x1.set(0, 0, 0.5);  // temperature
    x1.set(0, 1, 0.3);  // humidity
    sequence.push_back(x1);
    std::cout << "  t=1: [" << x1.get(0,0) << ", " << x1.get(0,1) << "]\n";
    
    // Time step 2
    Matrix x2(1, 2);
    x2.set(0, 0, 0.6);
    x2.set(0, 1, 0.5);
    sequence.push_back(x2);
    std::cout << "  t=2: [" << x2.get(0,0) << ", " << x2.get(0,1) << "]\n";
    
    // Time step 3
    Matrix x3(1, 2);
    x3.set(0, 0, 0.7);
    x3.set(0, 1, 0.8);
    sequence.push_back(x3);
    std::cout << "  t=3: [" << x3.get(0,0) << ", " << x3.get(0,1) << "]\n\n";
    
    // Forward pass
    Matrix output = rnn.forward(sequence);
    
    std::cout << "Output (prediction after seeing all 3 steps):\n";
    std::cout << "  Rainfall prediction: " << output.get(0, 0) << "\n\n";
    
    std::cout << YELLOW << "KEY INSIGHT:\n" << RESET;
    std::cout << "The RNN 'remembers' information from t=1 and t=2 when making\n";
    std::cout << "the prediction at t=3. This is stored in the hidden state h(t)!\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 2: RNN vs Feedforward - Why Sequence Matters
// ═══════════════════════════════════════════════════════════════════════════

void example2_RNNvsFeedforward() {
    printHeader("EXAMPLE 2: RNN vs Feedforward Network");
    
    std::cout << BOLD << "PROBLEM: Predicting the next number in a sequence\n" << RESET;
    std::cout << "Sequence: [0.0, 0.5, 1.0, ?]\n";
    std::cout << "Pattern: increasing by 0.5\n";
    std::cout << "Expected next value: 1.5\n\n";
    
    printSubHeader("Why Feedforward Networks Fail");
    
    std::cout << RED << "FEEDFORWARD PROBLEM:\n" << RESET;
    std::cout << "• Treats each input independently\n";
    std::cout << "• Cannot see the pattern: 0.0 → 0.5 → 1.0\n";
    std::cout << "• No memory of previous inputs\n";
    std::cout << "• Prediction: random (no context)\n\n";
    
    printSubHeader("Why RNNs Succeed");
    
    std::cout << GREEN << "RNN ADVANTAGE:\n" << RESET;
    std::cout << "• Processes sequence step-by-step\n";
    std::cout << "• At t=1: sees 0.0, stores in h(1)\n";
    std::cout << "• At t=2: sees 0.5, combines with h(1), detects +0.5 pattern\n";
    std::cout << "• At t=3: sees 1.0, confirms pattern, predicts 1.5\n\n";
    
    std::cout << BOLD << "Visual Comparison:\n" << RESET;
    std::cout << "\n";
    std::cout << "FEEDFORWARD:          RNN:\n";
    std::cout << "┌────┐               ┌────┐\n";
    std::cout << "│0.0 │──▶NN──▶?      │0.0 │──▶RNN──▶h₁\n";
    std::cout << "└────┘               └────┘     │\n";
    std::cout << "                     ┌────┐     ▼\n";
    std::cout << "┌────┐               │0.5 │──▶RNN──▶h₂ (knows: +0.5)\n";
    std::cout << "│0.5 │──▶NN──▶?      └────┘     │\n";
    std::cout << "└────┘               ┌────┐     ▼\n";
    std::cout << "                     │1.0 │──▶RNN──▶h₃ (confirms: +0.5)\n";
    std::cout << "┌────┐               └────┘     │\n";
    std::cout << "│1.0 │──▶NN──▶?                 ▼\n";
    std::cout << "└────┘               Prediction: 1.5 ✓\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 3: Sine Wave Prediction (Time Series)
// ═══════════════════════════════════════════════════════════════════════════

void example3_SineWavePrediction() {
    printHeader("EXAMPLE 3: Sine Wave Prediction (Time Series)");
    
    std::cout << "TASK: Learn to predict the next value in a sine wave\n";
    std::cout << "This simulates real-world time series like:\n";
    std::cout << "  • Stock prices\n";
    std::cout << "  • Temperature cycles\n";
    std::cout << "  • Heart rate monitoring\n\n";
    
    printSubHeader("Generating Training Data");
    
    // Generate sine wave sequences
    const int num_sequences = 50;
    const int sequence_length = 10;
    std::vector<std::vector<Matrix>> sequences;
    std::vector<Matrix> targets;
    
    std::cout << "Creating " << num_sequences << " sequences of length " 
              << sequence_length << "...\n";
    
    for (int i = 0; i < num_sequences; ++i) {
        std::vector<Matrix> seq;
        double start_t = static_cast<double>(i) / 10.0;
        
        for (int t = 0; t < sequence_length; ++t) {
            Matrix input(1, 1);
            double value = std::sin(start_t + t * 0.1);
            input.set(0, 0, value);
            seq.push_back(input);
        }
        
        // Target: next value in sequence
        Matrix target(1, 1);
        double next_value = std::sin(start_t + sequence_length * 0.1);
        target.set(0, 0, next_value);
        
        sequences.push_back(seq);
        targets.push_back(target);
    }
    
    std::cout << GREEN << "✓ Generated " << sequences.size() << " training sequences\n" << RESET;
    
    std::cout << "\nExample sequence:\n";
    for (int t = 0; t < 5; ++t) {
        std::cout << "  t=" << t << ": " << std::fixed << std::setprecision(4) 
                  << sequences[0][t].get(0, 0) << "\n";
    }
    std::cout << "  ...\n";
    std::cout << "  Target: " << targets[0].get(0, 0) << "\n\n";
    
    printSubHeader("Training RNN");
    
    // Create and train RNN
    RNNNetwork network;
    network.addLayer(new RNNLayer(1, 8, 1, false, new Tanh(), new Linear()));
    
    std::cout << "Network architecture:\n";
    network.summary();
    
    std::cout << "Training...\n";
    network.train(sequences, targets, 500, 0.01, true);
    
    printSubHeader("Testing Predictions");
    
    // Test on a new sequence
    std::vector<Matrix> test_seq;
    for (int t = 0; t < sequence_length; ++t) {
        Matrix input(1, 1);
        input.set(0, 0, std::sin(3.0 + t * 0.1));
        test_seq.push_back(input);
    }
    
    Matrix prediction = network.predict(test_seq);
    double actual = std::sin(3.0 + sequence_length * 0.1);
    
    std::cout << "\nTest sequence (last 3 values):\n";
    for (int t = sequence_length - 3; t < sequence_length; ++t) {
        std::cout << "  " << test_seq[t].get(0, 0) << "\n";
    }
    std::cout << "\n";
    std::cout << "Predicted next value: " << prediction.get(0, 0) << "\n";
    std::cout << "Actual next value:    " << actual << "\n";
    std::cout << "Error:                " << std::abs(prediction.get(0, 0) - actual) << "\n\n";
    
    if (std::abs(prediction.get(0, 0) - actual) < 0.1) {
        std::cout << GREEN << "✓ Good prediction! RNN learned the pattern.\n" << RESET;
    } else {
        std::cout << YELLOW << "⚠ Needs more training or larger hidden size.\n" << RESET;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 4: Sequence Memory Test
// ═══════════════════════════════════════════════════════════════════════════

void example4_SequenceMemory() {
    printHeader("EXAMPLE 4: Sequence Memory Test");
    
    std::cout << "TASK: Remember and reproduce a sequence\n";
    std::cout << "Input:  [1, 0, 1, 1, 0]\n";
    std::cout << "Output: Should predict the sequence pattern\n\n";
    
    std::cout << BOLD << "This demonstrates:\n" << RESET;
    std::cout << "• RNN's ability to maintain temporal context\n";
    std::cout << "• Memory persistence across time steps\n";
    std::cout << "• Sequential pattern recognition\n\n";
    
    // Create simple binary sequence
    std::vector<Matrix> sequence;
    double pattern[] = {1.0, 0.0, 1.0, 1.0, 0.0};
    
    for (int i = 0; i < 5; ++i) {
        Matrix input(1, 1);
        input.set(0, 0, pattern[i]);
        sequence.push_back(input);
    }
    
    // Create RNN
    RNNLayer rnn(1, 4, 1, true, new Tanh(), new Sigmoid());  // return_sequences=true
    
    std::cout << "Processing sequence through RNN (return_sequences=True):\n\n";
    
    Matrix outputs = rnn.forward(sequence);
    
    std::cout << "Time Step | Input | Hidden State Activated | Output\n";
    std::cout << "──────────┼───────┼───────────────────────┼────────\n";
    
    for (int t = 0; t < 5; ++t) {
        std::cout << "    " << t << "     │  " << pattern[t] 
                  << "    │        (internal)       │  " 
                  << std::fixed << std::setprecision(4) 
                  << outputs.get(t, 0) << "\n";
    }
    
    std::cout << "\n" << CYAN << "Notice:\n" << RESET;
    std::cout << "• Each output depends on ALL previous inputs\n";
    std::cout << "• The hidden state accumulates information over time\n";
    std::cout << "• Later time steps have more context than earlier ones\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 5: Key Concepts Summary
// ═══════════════════════════════════════════════════════════════════════════

void example5_KeyConcepts() {
    printHeader("EXAMPLE 5: Key Concepts Summary");
    
    std::cout << BOLD << "1. HIDDEN STATE (Memory)\n" << RESET;
    std::cout << "   • The 'memory' of the RNN\n";
    std::cout << "   • Updated at each time step\n";
    std::cout << "   • Carries information from past to future\n";
    std::cout << "   • Size determines memory capacity\n\n";
    
    std::cout << BOLD << "2. BACKPROPAGATION THROUGH TIME (BPTT)\n" << RESET;
    std::cout << "   • How RNNs learn from sequences\n";
    std::cout << "   • Unfolds the RNN across time\n";
    std::cout << "   • Computes gradients backward through sequence\n";
    std::cout << "   • Can suffer from vanishing gradients (use LSTM/GRU)\n\n";
    
    std::cout << BOLD << "3. RETURN SEQUENCES vs RETURN LAST\n" << RESET;
    std::cout << "   • return_sequences=False: Only last output (many-to-one)\n";
    std::cout << "     Use for: classification, regression\n";
    std::cout << "   • return_sequences=True: All outputs (many-to-many)\n";
    std::cout << "     Use for: sequence translation, generation\n\n";
    
    std::cout << BOLD << "4. COMMON ARCHITECTURES\n" << RESET;
    std::cout << "   • Many-to-One:  Sequence → Single output (sentiment analysis)\n";
    std::cout << "   • Many-to-Many: Sequence → Sequence (translation)\n";
    std::cout << "   • One-to-Many:  Single → Sequence (image captioning)\n\n";
    
    std::cout << BOLD << "5. ADVANTAGES OVER FEEDFORWARD\n" << RESET;
    std::cout << "   ✓ Handles variable-length sequences\n";
    std::cout << "   ✓ Shares parameters across time (efficient)\n";
    std::cout << "   ✓ Maintains temporal context\n";
    std::cout << "   ✓ Can process sequences of any length\n\n";
    
    std::cout << BOLD << "6. LIMITATIONS\n" << RESET;
    std::cout << "   ✗ Vanishing/exploding gradients\n";
    std::cout << "   ✗ Difficult to learn long-term dependencies\n";
    std::cout << "   ✗ Sequential (can't parallelize like feedforward)\n";
    std::cout << "   ✗ Solutions: LSTM, GRU, attention mechanisms\n\n";
    
    std::cout << BOLD << "7. WHEN TO USE RNN vs FEEDFORWARD\n" << RESET;
    std::cout << "\n";
    std::cout << "   Use RNN when:\n";
    std::cout << "   • Data is sequential (order matters)\n";
    std::cout << "   • Need to remember past information\n";
    std::cout << "   • Variable-length inputs\n";
    std::cout << "   • Time-dependent patterns\n\n";
    std::cout << "   Use Feedforward when:\n";
    std::cout << "   • Data is independent (tabular, images)\n";
    std::cout << "   • No temporal dependencies\n";
    std::cout << "   • Fixed-size inputs\n";
    std::cout << "   • Order doesn't matter\n\n";
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << BOLD << MAGENTA;
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║       RECURRENT NEURAL NETWORKS (RNN) - COMPLETE GUIDE        ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "║  Understanding how RNNs overcome limitations of feedforward   ║\n";
    std::cout << "║  networks for sequential data processing                      ║\n";
    std::cout << "║                                                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << RESET << "\n";
    
    try {
        example1_RNNArchitecture();
        example2_RNNvsFeedforward();
        example3_SineWavePrediction();
        example4_SequenceMemory();
        example5_KeyConcepts();
        
        std::cout << GREEN << BOLD;
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  ✓ All examples completed successfully!                        ║\n";
        std::cout << "║                                                                ║\n";
        std::cout << "║  You now understand:                                           ║\n";
        std::cout << "║  • What RNNs are and how they work                            ║\n";
        std::cout << "║  • Why they're superior to feedforward for sequences          ║\n";
        std::cout << "║  • How hidden states provide memory                           ║\n";
        std::cout << "║  • When to use RNNs vs other architectures                    ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << RESET << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
