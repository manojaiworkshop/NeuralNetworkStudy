/**
 * @file layer_cuda.h
 * @brief GPU-accelerated neural network layer implementations using CUDA
 * 
 * This file provides CUDA versions of neural network layers that leverage
 * GPU acceleration through MatrixCUDA and ActivationCUDA. All computations
 * are performed on the GPU for maximum performance.
 * 
 * Features:
 * - GPU-accelerated forward pass
 * - GPU-accelerated backward pass (gradient computation)
 * - Efficient memory management on GPU
 * - Multiple weight initialization strategies
 * - Support for various activation functions
 * 
 * Usage:
 *   DenseLayerCUDA layer(784, 128, new ReLUCUDA());
 *   MatrixCUDA output = layer.forward(input);
 *   MatrixCUDA grad = layer.backward(output_grad);
 */

#ifndef LAYER_CUDA_H
#define LAYER_CUDA_H

#include "matrix_cuda.h"
#include "activation_cuda.h"
#include <memory>
#include <string>

/**
 * @class LayerCUDA
 * @brief Abstract base class for GPU-accelerated neural network layers
 * 
 * This class defines the interface that all CUDA layer types must implement.
 * It provides the fundamental operations needed for neural network training:
 * - Forward propagation (computing outputs from inputs)
 * - Backward propagation (computing gradients for learning)
 * - Parameter updates (applying gradients to learn)
 * 
 * Design Philosophy:
 * - All operations performed on GPU
 * - Minimal CPU-GPU data transfers
 * - Compatible with CPU layer interface
 * - Efficient memory usage
 */
class LayerCUDA {
public:
    virtual ~LayerCUDA() = default;
    
    /**
     * @brief Forward propagation - compute layer output from input
     * 
     * This is the prediction step where we compute:
     *   Z = X·W^T + b    (linear transformation)
     *   A = σ(Z)         (activation function)
     * 
     * The computation happens entirely on the GPU using CUDA kernels
     * launched by MatrixCUDA operations.
     * 
     * @param input Input matrix (batch_size × input_features)
     * @return Output matrix (batch_size × output_features)
     * 
     * Example:
     *   MatrixCUDA input(32, 784);  // 32 images, 784 pixels each
     *   MatrixCUDA output = layer.forward(input);
     *   // output: (32 × 128) if layer outputs 128 features
     */
    virtual MatrixCUDA forward(const MatrixCUDA& input) = 0;
    
    /**
     * @brief Backward propagation - compute gradients for learning
     * 
     * This computes three types of gradients:
     * 1. ∂L/∂W (weight gradients) - how to update weights
     * 2. ∂L/∂b (bias gradients) - how to update biases
     * 3. ∂L/∂X (input gradients) - pass to previous layer
     * 
     * All gradient computations use GPU-accelerated matrix operations.
     * 
     * @param output_gradient Gradient from next layer (∂L/∂output)
     * @return Gradient with respect to input (∂L/∂input)
     * 
     * Example:
     *   MatrixCUDA loss_grad(32, 10);  // Gradient from loss function
     *   MatrixCUDA input_grad = layer.backward(loss_grad);
     *   // input_grad: (32 × 128) to pass to previous layer
     */
    virtual MatrixCUDA backward(const MatrixCUDA& output_gradient) = 0;
    
    /**
     * @brief Update layer parameters using computed gradients
     * 
     * Applies gradient descent update:
     *   W = W - learning_rate × ∂L/∂W
     *   b = b - learning_rate × ∂L/∂b
     * 
     * Updates are performed directly on GPU memory.
     * 
     * @param learning_rate Step size for gradient descent
     * 
     * Example:
     *   layer.backward(loss_grad);      // Compute gradients
     *   layer.updateParameters(0.01);   // Apply updates
     */
    virtual void updateParameters(double learning_rate) = 0;
    
    /**
     * @brief Get the name of this layer type
     * @return Layer type name (e.g., "DenseCUDA")
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the number of input features
     * @return Input dimensionality
     */
    virtual size_t getInputSize() const = 0;
    
    /**
     * @brief Get the number of output features
     * @return Output dimensionality
     */
    virtual size_t getOutputSize() const = 0;
    
    /**
     * @brief Get total number of trainable parameters
     * @return Total parameters (weights + biases)
     * 
     * For dense layer: (input_size × output_size) + output_size
     */
    virtual size_t getParameterCount() const = 0;
};

/**
 * @class DenseLayerCUDA
 * @brief GPU-accelerated fully connected (dense) layer
 * 
 * A dense layer connects every input to every output. Each output neuron
 * computes a weighted sum of all inputs plus a bias, then applies an
 * activation function:
 * 
 *   output_i = activation(Σ(w_ij × input_j) + b_i)
 * 
 * GPU Acceleration:
 * - Matrix multiplications use cuBLAS
 * - Element-wise operations use CUDA kernels
 * - All data remains on GPU during training
 * - Minimal CPU-GPU transfers
 * 
 * Memory Layout:
 * - weights: (output_size × input_size) stored on GPU
 * - biases: (output_size × 1) stored on GPU
 * - gradients: Same shapes as parameters, on GPU
 * - cached data: Saved on GPU for backward pass
 * 
 * Example Usage:
 *   // Create layer: 784 inputs → 256 outputs with ReLU
 *   DenseLayerCUDA layer(784, 256, new ReLUCUDA());
 *   
 *   // Initialize weights
 *   layer.initializeWeights("he");
 *   
 *   // Forward pass (on GPU)
 *   MatrixCUDA input(32, 784);      // 32 samples
 *   MatrixCUDA output = layer.forward(input);
 *   
 *   // Backward pass (on GPU)
 *   MatrixCUDA grad = layer.backward(output_gradient);
 *   
 *   // Update (on GPU)
 *   layer.updateParameters(0.01);
 */
class DenseLayerCUDA : public LayerCUDA {
private:
    // Layer dimensions
    size_t input_size;   ///< Number of input features
    size_t output_size;  ///< Number of output features (neurons)
    
    // Parameters (stored on GPU)
    MatrixCUDA weights;           ///< Weight matrix (output × input)
    MatrixCUDA biases;            ///< Bias vector (output × 1)
    MatrixCUDA weight_gradients;  ///< Accumulated weight gradients
    MatrixCUDA bias_gradients;    ///< Accumulated bias gradients
    
    // Cached values for backward pass (stored on GPU)
    MatrixCUDA cached_input;  ///< Input from forward pass
    MatrixCUDA cached_z;      ///< Pre-activation values (Z = X·W^T + b)
    
    // Activation function (GPU version)
    std::unique_ptr<ActivationCUDA> activation;  ///< Optional activation function
    
public:
    /**
     * @brief Construct a dense layer with GPU acceleration
     * 
     * Allocates GPU memory for:
     * - Weight matrix (output_size × input_size)
     * - Bias vector (output_size × 1)
     * - Gradient matrices (same shapes as parameters)
     * 
     * Initializes weights using Xavier initialization by default.
     * 
     * @param input_size Number of input features
     * @param output_size Number of output neurons
     * @param act Activation function (nullptr for linear)
     * 
     * Example:
     *   // Layer with 100 inputs, 50 outputs, ReLU activation
     *   DenseLayerCUDA layer(100, 50, new ReLUCUDA());
     *   
     *   // Linear layer (no activation)
     *   DenseLayerCUDA linear(100, 50, nullptr);
     */
    DenseLayerCUDA(size_t input_size, size_t output_size, 
                   ActivationCUDA* act = nullptr);
    
    /**
     * @brief Initialize layer weights using various strategies
     * 
     * Strategies:
     * 1. "xavier" (Glorot): Var(W) = 2/(n_in + n_out)
     *    - Best for: Sigmoid, Tanh activations
     *    - Keeps gradient variance stable
     * 
     * 2. "he": Var(W) = 2/n_in
     *    - Best for: ReLU, LeakyReLU activations
     *    - Accounts for ReLU killing half the neurons
     * 
     * 3. "random": Uniform[-1, 1]
     *    - General purpose random initialization
     * 
     * 4. "zeros": All zeros (⚠️ causes symmetry problem!)
     *    - Only use for debugging
     * 
     * Initialization happens on CPU, then transferred to GPU.
     * 
     * @param strategy Initialization strategy name
     * 
     * Example:
     *   DenseLayerCUDA layer(784, 128, new ReLUCUDA());
     *   layer.initializeWeights("he");  // He init for ReLU
     */
    void initializeWeights(const std::string& strategy = "xavier");
    
    // Implement LayerCUDA interface
    MatrixCUDA forward(const MatrixCUDA& input) override;
    MatrixCUDA backward(const MatrixCUDA& output_gradient) override;
    void updateParameters(double learning_rate) override;
    
    // Getters
    std::string getName() const override { return "DenseCUDA"; }
    size_t getInputSize() const override { return input_size; }
    size_t getOutputSize() const override { return output_size; }
    size_t getParameterCount() const override { 
        return input_size * output_size + output_size; 
    }
    
    /**
     * @brief Get weight matrix (returns copy from GPU to CPU)
     * 
     * This triggers a GPU→CPU memory transfer. Use sparingly
     * during training. Useful for:
     * - Saving model weights
     * - Debugging
     * - Visualization
     * 
     * @return Weight matrix copy on CPU
     */
    MatrixCUDA getWeights() const { return weights; }
    
    /**
     * @brief Get bias vector (returns copy from GPU to CPU)
     * @return Bias vector copy on CPU
     */
    MatrixCUDA getBiases() const { return biases; }
    
    /**
     * @brief Set weight matrix (transfers from CPU to GPU)
     * 
     * Useful for:
     * - Loading pre-trained weights
     * - Transfer learning
     * - Custom initialization
     * 
     * @param w New weight matrix
     */
    void setWeights(const MatrixCUDA& w) { weights = w; }
    
    /**
     * @brief Set bias vector (transfers from CPU to GPU)
     * @param b New bias vector
     */
    void setBiases(const MatrixCUDA& b) { biases = b; }
    
    /**
     * @brief Get weight gradients (GPU memory)
     * @return Weight gradients on GPU
     */
    MatrixCUDA getWeightGradients() const { return weight_gradients; }
    
    /**
     * @brief Get bias gradients (GPU memory)
     * @return Bias gradients on GPU
     */
    MatrixCUDA getBiasGradients() const { return bias_gradients; }
    
    /**
     * @brief Reset accumulated gradients to zero
     * 
     * Call this after updating parameters to clear gradients
     * for the next batch. Operation happens on GPU.
     * 
     * Example:
     *   for (batch in dataset) {
     *       layer.resetGradients();  // Clear before batch
     *       // ... forward, backward ...
     *       layer.updateParameters(lr);
     *   }
     */
    void resetGradients();
};

#endif // LAYER_CUDA_H
