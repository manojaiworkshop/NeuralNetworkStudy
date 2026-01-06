#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "loss.h"
#include "optimizer.h"
#include "matrix.h"
#include <vector>
#include <memory>
#include <string>

/**
 * @brief Neural Network class that manages layers, training, and prediction
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss_function;
    std::unique_ptr<Optimizer> optimizer;
    
    bool use_optimizer;  // Flag to use optimizer vs simple gradient descent
    
public:
    NeuralNetwork();
    ~NeuralNetwork() = default;
    
    /**
     * @brief Add a layer to the network
     * @param layer Layer to add (network takes ownership)
     */
    void addLayer(Layer* layer);
    
    /**
     * @brief Set loss function
     * @param loss Loss function (network takes ownership)
     */
    void setLoss(Loss* loss);
    
    /**
     * @brief Set optimizer
     * @param opt Optimizer (network takes ownership)
     */
    void setOptimizer(Optimizer* opt);
    
    /**
     * @brief Forward pass through the network
     * @param input Input matrix
     * @return Network output
     */
    Matrix forward(const Matrix& input);
    
    /**
     * @brief Backward pass through the network
     * @param loss_gradient Gradient of loss with respect to output
     */
    void backward(const Matrix& loss_gradient);
    
    /**
     * @brief Update network parameters
     * @param learning_rate Learning rate (used if no optimizer is set)
     */
    void updateParameters(double learning_rate = 0.01);
    
    /**
     * @brief Train the network on a dataset
     * @param X_train Training inputs
     * @param y_train Training targets
     * @param epochs Number of training epochs
     * @param batch_size Batch size for mini-batch training
     * @param learning_rate Learning rate (if not using optimizer)
     * @param verbose Print training progress
     */
    void train(const Matrix& X_train, const Matrix& y_train, 
              int epochs, int batch_size = 32, 
              double learning_rate = 0.01, bool verbose = true);
    
    /**
     * @brief Train with validation data
     */
    void trainWithValidation(const Matrix& X_train, const Matrix& y_train,
                            const Matrix& X_val, const Matrix& y_val,
                            int epochs, int batch_size = 32,
                            double learning_rate = 0.01, bool verbose = true);
    
    /**
     * @brief Make predictions on input data
     * @param input Input matrix
     * @return Predictions
     */
    Matrix predict(const Matrix& input);
    
    /**
     * @brief Evaluate the network on test data
     * @param X_test Test inputs
     * @param y_test Test targets
     * @return Loss value
     */
    double evaluate(const Matrix& X_test, const Matrix& y_test);
    
    /**
     * @brief Calculate accuracy for classification tasks
     * @param X Test inputs
     * @param y Test targets (one-hot encoded or class indices)
     * @return Accuracy percentage
     */
    double accuracy(const Matrix& X, const Matrix& y);
    
    /**
     * @brief Get number of layers
     */
    size_t getNumLayers() const { return layers.size(); }
    
    /**
     * @brief Get layer at index
     */
    Layer* getLayer(size_t index);
    
    /**
     * @brief Print network summary
     */
    void summary() const;
    
    /**
     * @brief Reset optimizer state
     */
    void resetOptimizer();
    
    /**
     * @brief Visualize the network architecture with weights
     * @param show_weights If true, display weight statistics
     * @param filename Optional: save to file (default: print to console)
     */
    void visualizeNetwork(bool show_weights = true, const std::string& filename = "") const;
    
    /**
     * @brief Display trained weights for all layers
     * @param detailed If true, show full weight matrices
     */
    void displayWeights(bool detailed = false) const;
    
    /**
     * @brief Save the trained model to file
     * @param filename Path to save file
     * @return True if successful
     */
    bool saveModel(const std::string& filename) const;
    
    /**
     * @brief Load a trained model from file
     * @param filename Path to model file
     * @return True if successful
     */
    bool loadModel(const std::string& filename);
    
    /**
     * @brief Export network graph to DOT format (for GraphViz)
     * @param filename Output filename
     */
    void exportToDot(const std::string& filename) const;
    
private:
    /**
     * @brief Create mini-batches from data
     */
    std::vector<std::pair<Matrix, Matrix>> createBatches(
        const Matrix& X, const Matrix& y, int batch_size);
    
    /**
     * @brief Shuffle data for training
     */
    void shuffleData(Matrix& X, Matrix& y);
};

#endif // NETWORK_H
