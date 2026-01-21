#ifndef RNN_H
#define RNN_H

#include "matrix.h"
#include "activation.h"
#include <memory>
#include <vector>
#include <string>

/**
 * @file rnn.h
 * @brief Recurrent Neural Network (RNN) implementation
 * 
 * RNNs are designed for SEQUENTIAL DATA where order matters:
 * - Time series prediction
 * - Text generation
 * - Speech recognition
 * - Video analysis
 * 
 * KEY DIFFERENCE FROM FEEDFORWARD NETWORKS:
 * - Feedforward: Each input is processed independently
 * - RNN: Maintains "memory" (hidden state) across time steps
 * 
 * ARCHITECTURE:
 * At each time step t:
 *   h(t) = activation(W_hh * h(t-1) + W_xh * x(t) + b_h)
 *   y(t) = activation(W_hy * h(t) + b_y)
 * 
 * Where:
 *   h(t) = hidden state at time t (memory)
 *   x(t) = input at time t
 *   y(t) = output at time t
 */

/**
 * @brief Basic RNN Cell (Vanilla RNN)
 * 
 * Processes one time step at a time
 * Maintains hidden state between time steps
 */
class RNNCell {
private:
    size_t input_size;
    size_t hidden_size;
    
    // Parameters
    Matrix W_xh;  // Input to hidden weights (hidden_size x input_size)
    Matrix W_hh;  // Hidden to hidden weights (hidden_size x hidden_size)
    Matrix b_h;   // Hidden bias (hidden_size x 1)
    
    // Gradients
    Matrix dW_xh;
    Matrix dW_hh;
    Matrix db_h;
    
    // Activation function
    std::unique_ptr<Activation> activation;
    
    // Cache for backward pass
    Matrix cached_input;
    Matrix cached_prev_hidden;
    Matrix cached_hidden;
    
public:
    /**
     * @brief Constructor
     * @param input_size Size of input vector
     * @param hidden_size Size of hidden state
     * @param activation Activation function (default: Tanh)
     */
    RNNCell(size_t input_size, size_t hidden_size, Activation* activation = nullptr);
    
    /**
     * @brief Initialize weights
     * @param strategy "xavier", "he", or "random"
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    /**
     * @brief Forward pass for one time step
     * @param input Input at current time step (batch_size x input_size)
     * @param prev_hidden Previous hidden state (batch_size x hidden_size)
     * @return New hidden state (batch_size x hidden_size)
     */
    Matrix forward(const Matrix& input, const Matrix& prev_hidden);
    
    /**
     * @brief Backward pass for one time step
     * @param grad_hidden Gradient w.r.t. hidden state
     * @param grad_next_hidden Gradient from next time step
     * @return Gradient w.r.t. input
     */
    Matrix backward(const Matrix& grad_hidden, const Matrix& grad_next_hidden);
    
    /**
     * @brief Update parameters
     * @param learning_rate Learning rate
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Reset gradients
     */
    void resetGradients();
    
    // Getters
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }
    Matrix getWeightsXH() const { return W_xh; }
    Matrix getWeightsHH() const { return W_hh; }
    Matrix getBiasH() const { return b_h; }
    
    // Setters (for custom initialization)
    void setWeightsXH(const Matrix& w) { W_xh = w; }
    void setWeightsHH(const Matrix& w) { W_hh = w; }
    void setBiasH(const Matrix& b) { b_h = b; }
};

/**
 * @brief Complete RNN Layer
 * 
 * Processes entire sequences
 * Can be stacked to create deep RNNs
 */
class RNNLayer {
private:
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    bool return_sequences;  // Return all time steps or just last one
    
    RNNCell cell;
    
    // Output layer parameters
    Matrix W_hy;  // Hidden to output weights (output_size x hidden_size)
    Matrix b_y;   // Output bias (output_size x 1)
    
    // Output gradients
    Matrix dW_hy;
    Matrix db_y;
    
    // Cached states for entire sequence
    std::vector<Matrix> hidden_states;
    std::vector<Matrix> inputs;
    
    std::unique_ptr<Activation> output_activation;
    
public:
    /**
     * @brief Constructor
     * @param input_size Size of input features
     * @param hidden_size Size of hidden state
     * @param output_size Size of output
     * @param return_sequences If true, return output for all time steps
     * @param hidden_activation Activation for hidden state (default: Tanh)
     * @param output_activation Activation for output (default: Linear)
     */
    RNNLayer(size_t input_size, size_t hidden_size, size_t output_size,
             bool return_sequences = false,
             Activation* hidden_activation = nullptr,
             Activation* output_activation = nullptr);
    
    /**
     * @brief Forward pass through sequence
     * @param sequence Input sequence (seq_length x batch_size x input_size)
     *                 Can be Matrix with seq_length*batch_size rows
     * @param initial_hidden Initial hidden state (default: zeros)
     * @return Output sequence or final output
     */
    Matrix forward(const std::vector<Matrix>& sequence, 
                   const Matrix& initial_hidden = Matrix());
    
    /**
     * @brief Backward pass through sequence (BPTT - Backpropagation Through Time)
     * @param grad_output Gradient w.r.t. output
     * @return Gradient w.r.t. input sequence
     */
    std::vector<Matrix> backward(const std::vector<Matrix>& grad_output);
    
    /**
     * @brief Update all parameters
     * @param learning_rate Learning rate
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
        return (input_size * hidden_size) +    // W_xh
               (hidden_size * hidden_size) +   // W_hh
               hidden_size +                   // b_h
               (output_size * hidden_size) +   // W_hy
               output_size;                    // b_y
    }
};

/**
 * @brief RNN Network for sequence-to-sequence tasks
 * 
 * Can handle:
 * - Many-to-one (sequence classification)
 * - Many-to-many (sequence translation)
 * - One-to-many (sequence generation)
 */
class RNNNetwork {
private:
    std::vector<std::unique_ptr<RNNLayer>> layers;
    
public:
    RNNNetwork() = default;
    ~RNNNetwork() = default;
    
    /**
     * @brief Add RNN layer
     * @param layer RNN layer (network takes ownership)
     */
    void addLayer(RNNLayer* layer);
    
    /**
     * @brief Forward pass through network
     * @param sequence Input sequence
     * @return Output
     */
    Matrix forward(const std::vector<Matrix>& sequence);
    
    /**
     * @brief Train on sequence data
     * @param sequences Training sequences
     * @param targets Target outputs
     * @param epochs Number of epochs
     * @param learning_rate Learning rate
     * @param verbose Print progress
     */
    void train(const std::vector<std::vector<Matrix>>& sequences,
               const std::vector<Matrix>& targets,
               int epochs,
               double learning_rate = 0.01,
               bool verbose = true);
    
    /**
     * @brief Make prediction
     * @param sequence Input sequence
     * @return Predicted output
     */
    Matrix predict(const std::vector<Matrix>& sequence);
    
    /**
     * @brief Print network summary
     */
    void summary() const;
};

#endif // RNN_H
