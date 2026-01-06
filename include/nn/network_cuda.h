/**
 * @file network_cuda.h
 * @brief GPU-accelerated Neural Network implementation using CUDA
 * 
 * This file provides a CUDA version of the NeuralNetwork class that leverages
 * GPU acceleration through MatrixCUDA, LayerCUDA, ActivationCUDA, and LossCUDA.
 * All computations are performed on the GPU for maximum performance.
 * 
 * Features:
 * - GPU-accelerated forward/backward propagation
 * - Efficient batch training on GPU
 * - Minimal CPU-GPU data transfers
 * - Compatible with CPU Network interface
 * - Support for various optimizers
 * 
 * Usage:
 *   NeuralNetworkCUDA network;
 *   network.addLayer(new DenseLayerCUDA(784, 128, new ReLUCUDA()));
 *   network.addLayer(new DenseLayerCUDA(128, 10, new SoftmaxCUDA()));
 *   network.setLoss(new CategoricalCrossEntropyLossCUDA());
 *   network.setOptimizer(new SGDOptimizerCUDA(0.01));
 *   network.train(X_train, y_train, epochs, batch_size);
 */

#ifndef NETWORK_CUDA_H
#define NETWORK_CUDA_H

#include "layer_cuda.h"
#include "loss_cuda.h"
#include "optimizer_cuda.h"
#include "matrix_cuda.h"
#include <vector>
#include <memory>
#include <string>

/**
 * @class NeuralNetworkCUDA
 * @brief GPU-accelerated Neural Network that manages layers, training, and prediction
 * 
 * This class provides a complete neural network implementation that runs entirely
 * on the GPU. It manages:
 * - Multiple layers (all on GPU)
 * - Forward and backward propagation (GPU)
 * - Loss calculation (GPU)
 * - Parameter updates (GPU)
 * - Mini-batch training (GPU)
 * 
 * Design Philosophy:
 * - Keep data on GPU as much as possible
 * - Minimize CPU-GPU transfers
 * - Leverage GPU parallelism for all operations
 * - Provide same interface as CPU version
 */
class NeuralNetworkCUDA {
private:
    std::vector<std::unique_ptr<LayerCUDA>> layers;        // Network layers (on GPU)
    std::unique_ptr<LossCUDA> loss_function;                // Loss function (GPU)
    std::unique_ptr<Optimizer_CUDA> optimizer;              // Optimizer (GPU)
    
    bool use_optimizer;  // Flag to use optimizer vs simple gradient descent
    
    // Training history
    std::vector<double> train_losses;
    std::vector<double> val_losses;
    std::vector<double> train_accuracies;
    std::vector<double> val_accuracies;
    
    /**
     * @brief Create mini-batches from data (on GPU)
     * @param X Input data
     * @param y Target data
     * @param batch_size Batch size
     * @return Vector of batches
     */
    std::vector<std::pair<MatrixCUDA, MatrixCUDA>> createBatches(
        const MatrixCUDA& X, const MatrixCUDA& y, int batch_size);
    
    /**
     * @brief Shuffle data for stochastic gradient descent
     * @param X Input data (modified in-place)
     * @param y Target data (modified in-place)
     */
    void shuffleData(MatrixCUDA& X, MatrixCUDA& y);
    
public:
    /**
     * @brief Constructor
     */
    NeuralNetworkCUDA();
    
    /**
     * @brief Destructor
     */
    ~NeuralNetworkCUDA() = default;
    
    /**
     * @brief Add a layer to the network
     * @param layer Layer to add (network takes ownership)
     * 
     * Example:
     *   network.addLayer(new DenseLayerCUDA(784, 128, new ReLUCUDA()));
     */
    void addLayer(LayerCUDA* layer);
    
    /**
     * @brief Set loss function
     * @param loss Loss function (network takes ownership)
     * 
     * Example:
     *   network.setLoss(new MSELossCUDA());
     */
    void setLoss(LossCUDA* loss);
    
    /**
     * @brief Set optimizer for advanced training
     * @param opt Optimizer (network takes ownership)
     * 
     * Example:
     *   network.setOptimizer(new Adam_CUDA(0.001));
     */
    void setOptimizer(Optimizer_CUDA* opt);
    
    /**
     * @brief Forward pass through the network (on GPU)
     * @param input Input matrix (on GPU)
     * @return Network output (on GPU)
     * 
     * Computation flow:
     *   input → layer1 → layer2 → ... → layerN → output
     * All operations happen on GPU with no CPU transfers
     */
    MatrixCUDA forward(const MatrixCUDA& input);
    
    /**
     * @brief Backward pass through the network (on GPU)
     * @param loss_gradient Gradient of loss with respect to output
     * 
     * Computation flow:
     *   loss_grad → layerN → ... → layer2 → layer1
     * Computes gradients for all parameters on GPU
     */
    void backward(const MatrixCUDA& loss_gradient);
    
    /**
     * @brief Update network parameters (on GPU)
     * @param learning_rate Learning rate (used if no optimizer is set)
     * 
     * Updates all weights and biases using computed gradients
     * Operations performed directly on GPU memory
     */
    void updateParameters(double learning_rate = 0.01);
    
    /**
     * @brief Train the network on a dataset (entirely on GPU)
     * @param X_train Training inputs
     * @param y_train Training targets
     * @param epochs Number of training epochs
     * @param batch_size Batch size for mini-batch training
     * @param learning_rate Learning rate (if not using optimizer)
     * @param verbose Print training progress
     * 
     * Training loop:
     * for each epoch:
     *   1. Shuffle data (on GPU)
     *   2. Create mini-batches (on GPU)
     *   3. For each batch:
     *      - Forward pass (GPU)
     *      - Compute loss (GPU)
     *      - Backward pass (GPU)
     *      - Update parameters (GPU)
     *   4. Print progress (transfer only loss scalar to CPU)
     */
    void train(const MatrixCUDA& X_train, const MatrixCUDA& y_train, 
              int epochs, int batch_size = 32, 
              double learning_rate = 0.01, bool verbose = true);
    
    /**
     * @brief Train with validation data for monitoring
     * @param X_train Training inputs
     * @param y_train Training targets
     * @param X_val Validation inputs
     * @param y_val Validation targets
     * @param epochs Number of training epochs
     * @param batch_size Batch size
     * @param learning_rate Learning rate
     * @param verbose Print progress
     * 
     * Same as train() but evaluates on validation set after each epoch
     * Useful for detecting overfitting and early stopping
     */
    void trainWithValidation(const MatrixCUDA& X_train, const MatrixCUDA& y_train,
                            const MatrixCUDA& X_val, const MatrixCUDA& y_val,
                            int epochs, int batch_size = 32,
                            double learning_rate = 0.01, bool verbose = true);
    
    /**
     * @brief Make predictions on input data (on GPU)
     * @param input Input matrix (on GPU)
     * @return Predictions (on GPU)
     * 
     * Performs forward pass without storing intermediate values
     * Faster than train mode, useful for inference
     */
    MatrixCUDA predict(const MatrixCUDA& input);
    
    /**
     * @brief Evaluate the network on test data
     * @param X_test Test inputs
     * @param y_test Test targets
     * @return Loss value
     * 
     * Computes loss on test set
     * Returns scalar value (transferred from GPU)
     */
    double evaluate(const MatrixCUDA& X_test, const MatrixCUDA& y_test);
    
    /**
     * @brief Calculate accuracy for classification tasks
     * @param X Test inputs
     * @param y Test targets (one-hot encoded or class indices)
     * @return Accuracy percentage
     * 
     * For classification: compares argmax(prediction) with argmax(target)
     * Returns percentage of correct predictions
     */
    double accuracy(const MatrixCUDA& X, const MatrixCUDA& y);
    
    /**
     * @brief Get number of layers
     * @return Number of layers in network
     */
    size_t getNumLayers() const { return layers.size(); }
    
    /**
     * @brief Get layer at index
     * @param index Layer index
     * @return Pointer to layer
     */
    LayerCUDA* getLayer(size_t index);
    
    /**
     * @brief Print network summary
     * 
     * Displays:
     * - Architecture (layer types and sizes)
     * - Parameter counts
     * - GPU memory usage
     * - Total parameters
     */
    void summary() const;
    
    /**
     * @brief Reset optimizer state (clears momentum, etc.)
     */
    void resetOptimizer();
    
    /**
     * @brief Visualize the network architecture
     * @param show_weights If true, display weight statistics
     * @param filename Optional: save to file
     * 
     * Creates ASCII art visualization of network:
     * - Layer connections
     * - Neuron counts
     * - Activation functions
     * - Parameter counts
     */
    void visualizeNetwork(bool show_weights = true, const std::string& filename = "") const;
    
    /**
     * @brief Display trained weights for all layers
     * @param detailed If true, show full weight matrices
     * 
     * Transfers weights from GPU to CPU for display
     * Shows weight statistics (min, max, mean, std)
     */
    void displayWeights(bool detailed = false) const;
    
    /**
     * @brief Get training history
     * @return Vector of loss values per epoch
     */
    const std::vector<double>& getTrainLosses() const { return train_losses; }
    const std::vector<double>& getValLosses() const { return val_losses; }
    const std::vector<double>& getTrainAccuracies() const { return train_accuracies; }
    const std::vector<double>& getValAccuracies() const { return val_accuracies; }
    
    /**
     * @brief Plot training history
     * 
     * Creates ASCII plot of:
     * - Training loss vs epoch
     * - Validation loss vs epoch (if available)
     * - Training accuracy vs epoch
     * - Validation accuracy vs epoch (if available)
     */
    void plotTrainingHistory() const;
    
    /**
     * @brief Save the trained model to file
     * @param filename Path to save file
     * @return True if successful
     * 
     * Saves:
     * - Network architecture
     * - All layer parameters (weights, biases)
     * - Optimizer state
     * 
     * Transfers data from GPU to CPU for saving
     */
    bool saveModel(const std::string& filename) const;
    
    /**
     * @brief Load a trained model from file
     * @param filename Path to model file
     * @return True if successful
     * 
     * Loads model and transfers parameters to GPU
     */
    bool loadModel(const std::string& filename);
    
    /**
     * @brief Print GPU memory usage
     * 
     * Shows:
     * - Total GPU memory
     * - Memory used by network
     * - Memory per layer
     * - Available memory
     */
    void printGPUMemoryUsage() const;
    
    /**
     * @brief Get total number of parameters
     * @return Total trainable parameters
     */
    size_t getTotalParameters() const;
};

#endif // NETWORK_CUDA_H
