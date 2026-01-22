#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include "activation.h"
#include <memory>
#include <vector>
#include <string>
#include <tuple>

/**
 * @file lstm.h
 * @brief Long Short-Term Memory (LSTM) Network Implementation
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY LSTM? THE PROBLEM WITH VANILLA RNN
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * VANILLA RNN PROBLEM: Vanishing Gradients
 * ─────────────────────────────────────────
 * In RNN: h(t) = tanh(W_xh·x(t) + W_hh·h(t-1) + b_h)
 * 
 * During backpropagation through time (BPTT):
 * ∂L/∂h(1) = ∂L/∂h(T) · ∏(∂h(t)/∂h(t-1))
 *                         └──────────────┘
 *                         Many small values < 1
 *                         multiply together
 *                         → gradient vanishes!
 * 
 * Example: 100 time steps with gradient 0.5 at each step:
 * 0.5^100 ≈ 7.9 × 10^-31 (effectively ZERO!)
 * 
 * Result: Network can't learn long-term dependencies
 * 
 * ═══════════════════════════════════════════════════════════════════════════
 * LSTM SOLUTION: Constant Error Carousel
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * LSTM Architecture has 3 GATES + 1 CELL STATE:
 * 
 * 1. FORGET GATE (f): What to remove from memory
 *    f(t) = σ(W_f·[h(t-1), x(t)] + b_f)
 * 
 * 2. INPUT GATE (i): What new information to store
 *    i(t) = σ(W_i·[h(t-1), x(t)] + b_i)
 *    C̃(t) = tanh(W_c·[h(t-1), x(t)] + b_c)
 * 
 * 3. OUTPUT GATE (o): What to output from memory
 *    o(t) = σ(W_o·[h(t-1), x(t)] + b_o)
 * 
 * 4. CELL STATE (C): The memory highway
 *    C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)
 *    h(t) = o(t) ⊙ tanh(C(t))
 * 
 * KEY INSIGHT: Cell state C(t) flows with minimal changes!
 * C(t) = f·C(t-1) + i·C̃  (mostly addition, not multiplication)
 * 
 * Gradient flow: ∂L/∂C(1) = ∂L/∂C(T) · ∏f(t)
 *                                      └────┘
 *                                      Values close to 1
 *                                      → gradient preserved!
 */

/**
 * @brief LSTM Cell - The building block
 */
class LSTMCell {
private:
    size_t input_size;
    size_t hidden_size;
    
    // Parameters (4 sets: forget, input, candidate, output)
    Matrix W_f, W_i, W_c, W_o;  // Input weights (hidden × input)
    Matrix U_f, U_i, U_c, U_o;  // Hidden weights (hidden × hidden)
    Matrix b_f, b_i, b_c, b_o;  // Biases (hidden × 1)
    
    // Gradients
    Matrix dW_f, dW_i, dW_c, dW_o;
    Matrix dU_f, dU_i, dU_c, dU_o;
    Matrix db_f, db_i, db_c, db_o;
    
    // Cached values for backward pass
    Matrix cached_input, cached_prev_hidden, cached_prev_cell;
    Matrix cached_forget_gate, cached_input_gate, cached_candidate, cached_output_gate;
    Matrix cached_cell, cached_hidden;
    
    // Activations
    std::unique_ptr<Sigmoid> sigmoid;
    std::unique_ptr<Tanh> tanh_activation;
    
public:
    /**
     * @brief Constructor
     * @param input_size Dimension of input vector
     * @param hidden_size Dimension of hidden state and cell state
     */
    LSTMCell(size_t input_size, size_t hidden_size);
    
    /**
     * @brief Initialize weights (Xavier initialization)
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    /**
     * @brief Forward pass for one time step
     * @param input Current input x(t)
     * @param prev_hidden Previous hidden state h(t-1)
     * @param prev_cell Previous cell state C(t-1)
     * @return Pair of (new_hidden, new_cell)
     */
    std::pair<Matrix, Matrix> forward(const Matrix& input, 
                                      const Matrix& prev_hidden,
                                      const Matrix& prev_cell);
    
    /**
     * @brief Backward pass for one time step
     * @param grad_hidden Gradient w.r.t. hidden state
     * @param grad_cell Gradient w.r.t. cell state
     * @return Tuple of (grad_input, grad_prev_hidden, grad_prev_cell)
     */
    std::tuple<Matrix, Matrix, Matrix> backward(const Matrix& grad_hidden,
                                                 const Matrix& grad_cell);
    
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
        return 4 * (input_size * hidden_size +    // W matrices
                   hidden_size * hidden_size +    // U matrices  
                   hidden_size);                  // biases
    }
};

/**
 * @brief LSTM Layer - Processes entire sequences
 */
class LSTMLayer {
private:
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    bool return_sequences;
    
    LSTMCell cell;
    
    // Output projection
    Matrix W_hy;  // Hidden to output
    Matrix b_y;   // Output bias
    Matrix dW_hy, db_y;
    
    // Cached sequences
    std::vector<Matrix> hidden_states;
    std::vector<Matrix> cell_states;
    std::vector<Matrix> inputs;
    
    std::unique_ptr<Activation> output_activation;
    
public:
    /**
     * @brief Constructor
     */
    LSTMLayer(size_t input_size, size_t hidden_size, size_t output_size,
              bool return_sequences = false,
              Activation* output_activation = nullptr);
    
    /**
     * @brief Forward pass through sequence
     */
    Matrix forward(const std::vector<Matrix>& sequence,
                   const Matrix& initial_hidden = Matrix(),
                   const Matrix& initial_cell = Matrix());
    
    /**
     * @brief Backward pass (BPTT through LSTM)
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
 * @brief LSTM Network for sequence tasks
 */
class LSTMNetwork {
private:
    std::vector<std::unique_ptr<LSTMLayer>> layers;
    
public:
    LSTMNetwork() = default;
    ~LSTMNetwork() = default;
    
    void addLayer(LSTMLayer* layer);
    
    Matrix forward(const std::vector<Matrix>& sequence);
    
    void train(const std::vector<std::vector<Matrix>>& sequences,
               const std::vector<Matrix>& targets,
               int epochs,
               double learning_rate = 0.01,
               bool verbose = true);
    
    Matrix predict(const std::vector<Matrix>& sequence);
    
    void summary() const;
};

#endif // LSTM_H
