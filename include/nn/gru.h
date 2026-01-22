#ifndef GRU_H
#define GRU_H

#include "matrix.h"
#include "activation.h"
#include <memory>
#include <vector>
#include <string>

/**
 * @file gru.h
 * @brief Gated Recurrent Unit (GRU) Network Implementation
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * GRU: SIMPLIFIED LSTM WITH FEWER PARAMETERS
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * COMPARISON:
 * ───────────
 * RNN:  1 gate,  Simple, fast, vanishing gradients
 * LSTM: 3 gates, Complex, many parameters, solves vanishing gradients
 * GRU:  2 gates, Balanced, fewer parameters, similar performance to LSTM
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY GRU?
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * ADVANTAGES OVER LSTM:
 * • 25% fewer parameters (2 gates vs 3 gates + cell state)
 * • Faster training (fewer computations)
 * • Often performs as well as LSTM
 * • Easier to tune
 * • Better for smaller datasets
 * 
 * ADVANTAGES OVER RNN:
 * • Solves vanishing gradient problem
 * • Can learn long-term dependencies
 * • Gates control information flow
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * GRU ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * GRU has 2 GATES (vs LSTM's 3):
 * 
 * 1. UPDATE GATE (z): How much of past to keep
 *    z(t) = σ(W_z·[h(t-1), x(t)])
 * 
 * 2. RESET GATE (r): How much past to forget
 *    r(t) = σ(W_r·[h(t-1), x(t)])
 * 
 * 3. CANDIDATE (h̃): New memory content
 *    h̃(t) = tanh(W_h·[r(t) ⊙ h(t-1), x(t)])
 * 
 * 4. FINAL OUTPUT: Interpolation between old and new
 *    h(t) = (1 - z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
 *          └────────────────┘   └─────────┘
 *          Keep old memory      Add new memory
 * 
 * KEY INSIGHT: No separate cell state! Hidden state = output
 * 
 * Visual Flow:
 * 
 *    h(t-1) ──┬────────────────┐
 *             │                │
 *             │   ┌────────┐   │
 *             ├──►│ Reset  │   │
 *             │   │Gate (r)│───┼──┐
 *             │   └────────┘   │  │
 *    x(t) ────┼────────────────┤  │
 *             │   ┌────────┐   │  │
 *             ├──►│Update  │   │  │
 *             │   │Gate (z)│───┼──┼──┐
 *             │   └────────┘   │  │  │
 *             │                │  │  │
 *             │                ▼  │  │
 *             │              ┌───┐│  │
 *             │              │ × ││  │
 *             │              └─┬─┘│  │
 *             │                │  │  │
 *             │   ┌────────┐  │  │  │
 *             └──►│  tanh  │◄─┘  │  │
 *                 │  (h̃)   │     │  │
 *                 └────┬───┘     │  │
 *                      │         │  │
 *                    ┌─▼─┐       │  │
 *                    │ × │◄──────┘  │
 *                    └─┬─┘          │
 *                      │            │
 *                    ┌─▼─┐        ┌─▼─┐
 *                    │ + │◄───────│1-z│
 *                    └─┬─┘        └───┘
 *                      │
 *                      ▼
 *                     h(t)
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * WHEN TO USE EACH?
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * USE RNN WHEN:
 * • Short sequences (< 10 steps)
 * • Simple patterns
 * • Need speed
 * • Baseline model
 * 
 * USE LSTM WHEN:
 * • Very long sequences (> 100 steps)
 * • Complex dependencies
 * • Large dataset available
 * • Need explicit memory control
 * • Speech recognition, translation
 * 
 * USE GRU WHEN:
 * • Medium sequences (10-100 steps)
 * • Limited data
 * • Need faster training
 * • Similar performance to LSTM acceptable
 * • Text generation, sentiment analysis
 * 
 * PARAMETER COUNT COMPARISON (hidden_size = h, input_size = i):
 * • RNN:  3h² + 3ih + 3h
 * • GRU:  6h² + 6ih + 6h  (2x RNN)
 * • LSTM: 8h² + 8ih + 8h  (2.67x RNN, 1.33x GRU)
 */

/**
 * @brief GRU Cell - The building block
 */
class GRUCell {
private:
    size_t input_size;
    size_t hidden_size;
    
    // Parameters (3 sets: reset, update, candidate)
    Matrix W_r, W_z, W_h;  // Input weights (hidden × input)
    Matrix U_r, U_z, U_h;  // Hidden weights (hidden × hidden)
    Matrix b_r, b_z, b_h;  // Biases (hidden × 1)
    
    // Gradients
    Matrix dW_r, dW_z, dW_h;
    Matrix dU_r, dU_z, dU_h;
    Matrix db_r, db_z, db_h;
    
    // Cached values for backward pass
    Matrix cached_input, cached_prev_hidden;
    Matrix cached_reset_gate, cached_update_gate;
    Matrix cached_candidate, cached_hidden;
    Matrix cached_reset_hidden;  // r ⊙ h(t-1)
    
    // Activations
    std::unique_ptr<Sigmoid> sigmoid;
    std::unique_ptr<Tanh> tanh_activation;
    
public:
    /**
     * @brief Constructor
     * @param input_size Dimension of input vector
     * @param hidden_size Dimension of hidden state
     */
    GRUCell(size_t input_size, size_t hidden_size);
    
    /**
     * @brief Initialize weights
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    /**
     * @brief Forward pass for one time step
     * @param input Current input x(t)
     * @param prev_hidden Previous hidden state h(t-1)
     * @return New hidden state h(t)
     */
    Matrix forward(const Matrix& input, const Matrix& prev_hidden);
    
    /**
     * @brief Backward pass for one time step
     * @param grad_hidden Gradient w.r.t. hidden state
     * @return Pair of (grad_input, grad_prev_hidden)
     */
    std::pair<Matrix, Matrix> backward(const Matrix& grad_hidden);
    
    /**
     * @brief Update parameters using gradients
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Reset all gradients to zero
     */
    void resetGradients();
    
    // Getters
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }
    
    int getParameterCount() const {
        return 3 * (input_size * hidden_size +    // W matrices
                   hidden_size * hidden_size +    // U matrices  
                   hidden_size);                  // biases
    }
};

/**
 * @brief GRU Layer - Processes entire sequences
 */
class GRULayer {
private:
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    bool return_sequences;
    
    GRUCell cell;
    
    // Output projection
    Matrix W_hy;  // Hidden to output
    Matrix b_y;   // Output bias
    Matrix dW_hy, db_y;
    
    // Cached sequences
    std::vector<Matrix> hidden_states;
    std::vector<Matrix> inputs;
    
    std::unique_ptr<Activation> output_activation;
    
public:
    /**
     * @brief Constructor
     */
    GRULayer(size_t input_size, size_t hidden_size, size_t output_size,
             bool return_sequences = false,
             Activation* output_activation = nullptr);
    
    /**
     * @brief Forward pass through sequence
     */
    Matrix forward(const std::vector<Matrix>& sequence,
                   const Matrix& initial_hidden = Matrix());
    
    /**
     * @brief Backward pass (BPTT)
     */
    std::vector<Matrix> backward(const std::vector<Matrix>& grad_output);
    
    /**
     * @brief Update all parameters
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Reset gradients
     */
    void resetGradients();
    
    /**
     * @brief Initialize weights
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    // Getters
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }
    size_t getOutputSize() const { return output_size; }
    bool getReturnSequences() const { return return_sequences; }
    
    int getParameterCount() const {
        return cell.getParameterCount() + 
               (output_size * hidden_size) + 
               output_size;
    }
};

/**
 * @brief GRU Network for sequence tasks
 */
class GRUNetwork {
private:
    std::vector<std::unique_ptr<GRULayer>> layers;
    
public:
    GRUNetwork() = default;
    ~GRUNetwork() = default;
    
    void addLayer(GRULayer* layer);
    
    Matrix forward(const std::vector<Matrix>& sequence);
    
    void train(const std::vector<std::vector<Matrix>>& sequences,
               const std::vector<Matrix>& targets,
               int epochs,
               double learning_rate = 0.01,
               bool verbose = true);
    
    Matrix predict(const std::vector<Matrix>& sequence);
    
    void summary() const;
};

#endif // GRU_H
