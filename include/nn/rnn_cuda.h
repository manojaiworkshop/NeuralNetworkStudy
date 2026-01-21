#ifndef RNN_CUDA_H
#define RNN_CUDA_H

#include "matrix_cuda.h"
#include "activation_cuda.h"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

/**
 * @file rnn_cuda.h
 * @brief GPU-accelerated Recurrent Neural Network implementation
 * 
 * CUDA-accelerated RNN for high-performance sequence processing
 * 
 * KEY OPTIMIZATIONS:
 * - Parallel processing across batch and hidden dimensions
 * - Efficient memory layout for sequential data
 * - Fused operations to reduce kernel launches
 * - Stream processing for overlapping computation
 * 
 * PERFORMANCE BENEFITS:
 * - 10-50x speedup on large sequences
 * - Efficient for large batch sizes
 * - Best for hidden_size >= 128
 */

/**
 * @brief CUDA-accelerated RNN Cell
 */
class RNNCellCUDA {
private:
    size_t input_size;
    size_t hidden_size;
    
    // Device (GPU) parameters
    float* d_W_xh;  // Input to hidden weights
    float* d_W_hh;  // Hidden to hidden weights
    float* d_b_h;   // Hidden bias
    
    // Host (CPU) copies for weight updates
    MatrixCUDA W_xh;
    MatrixCUDA W_hh;
    MatrixCUDA b_h;
    
    // Gradients on GPU
    float* d_dW_xh;
    float* d_dW_hh;
    float* d_db_h;
    
    // Activation
    std::unique_ptr<ActivationCUDA> activation;
    
    // Cached data for backward pass
    MatrixCUDA cached_input;
    MatrixCUDA cached_prev_hidden;
    MatrixCUDA cached_hidden;
    
    // Memory management
    void allocateGPU();
    void freeGPU();
    void copyWeightsToGPU();
    void copyWeightsFromGPU();
    
public:
    /**
     * @brief Constructor
     */
    RNNCellCUDA(size_t input_size, size_t hidden_size, 
                ActivationCUDA* activation = nullptr);
    
    /**
     * @brief Destructor
     */
    ~RNNCellCUDA();
    
    /**
     * @brief Initialize weights
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    /**
     * @brief Forward pass on GPU
     * @param input Input at current time step
     * @param prev_hidden Previous hidden state
     * @return New hidden state
     */
    MatrixCUDA forward(const MatrixCUDA& input, const MatrixCUDA& prev_hidden);
    
    /**
     * @brief Backward pass on GPU
     */
    MatrixCUDA backward(const MatrixCUDA& grad_hidden, 
                       const MatrixCUDA& grad_next_hidden);
    
    /**
     * @brief Update parameters on GPU
     */
    void updateParameters(double learning_rate);
    
    /**
     * @brief Reset gradients
     */
    void resetGradients();
    
    // Getters
    size_t getInputSize() const { return input_size; }
    size_t getHiddenSize() const { return hidden_size; }
};

/**
 * @brief CUDA-accelerated RNN Layer
 */
class RNNLayerCUDA {
private:
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    bool return_sequences;
    
    RNNCellCUDA cell;
    
    // Output layer on GPU
    MatrixCUDA W_hy;
    MatrixCUDA b_y;
    MatrixCUDA dW_hy;
    MatrixCUDA db_y;
    
    // Cached sequences
    std::vector<MatrixCUDA> hidden_states;
    std::vector<MatrixCUDA> inputs;
    
    std::unique_ptr<ActivationCUDA> output_activation;
    
public:
    /**
     * @brief Constructor
     */
    RNNLayerCUDA(size_t input_size, size_t hidden_size, size_t output_size,
                 bool return_sequences = false,
                 ActivationCUDA* hidden_activation = nullptr,
                 ActivationCUDA* output_activation = nullptr);
    
    /**
     * @brief Forward pass through sequence on GPU
     */
    MatrixCUDA forward(const std::vector<MatrixCUDA>& sequence,
                      const MatrixCUDA& initial_hidden = MatrixCUDA());
    
    /**
     * @brief Backward pass (BPTT) on GPU
     */
    std::vector<MatrixCUDA> backward(const std::vector<MatrixCUDA>& grad_output);
    
    /**
     * @brief Update parameters
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
        return (input_size * hidden_size) +
               (hidden_size * hidden_size) +
               hidden_size +
               (output_size * hidden_size) +
               output_size;
    }
};

/**
 * @brief CUDA-accelerated RNN Network
 */
class RNNNetworkCUDA {
private:
    std::vector<std::unique_ptr<RNNLayerCUDA>> layers;
    
public:
    RNNNetworkCUDA() = default;
    ~RNNNetworkCUDA() = default;
    
    /**
     * @brief Add RNN layer
     */
    void addLayer(RNNLayerCUDA* layer);
    
    /**
     * @brief Forward pass
     */
    MatrixCUDA forward(const std::vector<MatrixCUDA>& sequence);
    
    /**
     * @brief Train on GPU
     */
    void train(const std::vector<std::vector<MatrixCUDA>>& sequences,
               const std::vector<MatrixCUDA>& targets,
               int epochs,
               double learning_rate = 0.01,
               bool verbose = true);
    
    /**
     * @brief Predict on GPU
     */
    MatrixCUDA predict(const std::vector<MatrixCUDA>& sequence);
    
    /**
     * @brief Print summary
     */
    void summary() const;
};

#endif // RNN_CUDA_H
