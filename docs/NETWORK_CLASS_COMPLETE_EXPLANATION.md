# 🧠 Complete Network Class Explanation - Building Neural Networks

## Overview: What is the Network Class?

The `NeuralNetwork` class is a **container** that manages multiple layers, coordinates training, and handles predictions. Think of it as the **brain** that orchestrates all the individual neurons (layers).

```
┌────────────────────────────────────────────────────────────────────┐
│                    NEURAL NETWORK CLASS                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Container holding:                                          │ │
│  │  • Multiple Layers (vector<Layer*>)                         │ │
│  │  • Loss Function (Loss*)                                    │ │
│  │  • Optimizer (Optimizer*)                                   │ │
│  │                                                             │ │
│  │  Provides methods:                                          │ │
│  │  • addLayer() - build network                              │ │
│  │  • train() - learn from data                               │ │
│  │  • predict() - make predictions                            │ │
│  │  • forward() - propagate data forward                      │ │
│  │  • backward() - propagate gradients backward               │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## Network.h - Header File Line-by-Line

### Class Members (Private Data)

```cpp
// ============================================================================
// LINES 16-20: Private member variables - What the network stores
// ============================================================================

private:
    std::vector<std::unique_ptr<Layer>> layers;
    // Vector = dynamic array that can grow/shrink
    // unique_ptr = smart pointer that manages memory automatically
    // Layer = pointer to any layer type (DenseLayer, ConvLayer, etc.)
    //
    // Example:
    //   layers[0] → DenseLayer (784 → 128)
    //   layers[1] → DenseLayer (128 → 64)
    //   layers[2] → DenseLayer (64 → 10)
    //
    // Why vector?
    //   - Can add layers dynamically: layers.push_back()
    //   - Access by index: layers[i]
    //   - Iterate: for (auto& layer : layers)
    
    std::unique_ptr<Loss> loss_function;
    // Pointer to loss function object
    // Examples: MSELoss, BinaryCrossEntropyLoss, etc.
    // Used to measure how wrong predictions are
    //
    // Usage:
    //   double loss = loss_function->calculate(pred, target);
    
    std::unique_ptr<Optimizer> optimizer;
    // Optional optimizer for advanced training
    // Examples: SGD with momentum, Adam, RMSprop
    // If not set, uses simple gradient descent
    
    bool use_optimizer;
    // Flag: true = use optimizer, false = simple gradient descent


// ============================================================================
// KEY CONCEPT: Smart Pointers
// ============================================================================
//
// unique_ptr<Layer> means:
// • Automatic memory management (no memory leaks!)
// • When network is destroyed, layers are automatically deleted
// • Transfer ownership: network takes ownership of layers
//
// Without unique_ptr (old C++ way):
//   Layer* layer = new DenseLayer(10, 5);
//   // ... use layer ...
//   delete layer;  // MUST remember to free memory!
//
// With unique_ptr (modern C++):
//   std::unique_ptr<Layer> layer = std::make_unique<DenseLayer>(10, 5);
//   // ... use layer ...
//   // Automatic deletion when layer goes out of scope!
```

### Public Methods - Building the Network

```cpp
// ============================================================================
// LINES 30-34: addLayer() - Building the network architecture
// ============================================================================

void addLayer(Layer* layer);
// Add a layer to the network
//
// Parameters:
//   layer: Pointer to layer object (network takes ownership)
//
// What it does:
//   1. Takes raw pointer as input
//   2. Wraps in unique_ptr for automatic memory management
//   3. Adds to layers vector
//
// Example usage:
//   NeuralNetwork network;
//   network.addLayer(new DenseLayer(784, 128, new ReLU()));
//   network.addLayer(new DenseLayer(128, 64, new ReLU()));
//   network.addLayer(new DenseLayer(64, 10, new Sigmoid()));
//
// After these calls:
//   layers[0] → DenseLayer(784→128, ReLU)
//   layers[1] → DenseLayer(128→64, ReLU)
//   layers[2] → DenseLayer(64→10, Sigmoid)
//
// Network structure:
//   Input(784) → [Layer0] → (128) → [Layer1] → (64) → [Layer2] → Output(10)


// ============================================================================
// LINES 36-40: setLoss() - Defining how to measure error
// ============================================================================

void setLoss(Loss* loss);
// Set the loss function for training
//
// Parameters:
//   loss: Pointer to loss function object
//
// Example:
//   network.setLoss(new MSELoss());           // For regression
//   network.setLoss(new BinaryCrossEntropy()); // For binary classification
//   network.setLoss(new CategoricalCrossEntropy()); // For multi-class
//
// Loss function is used to:
//   1. Calculate error: loss = loss_function->calculate(pred, target)
//   2. Compute gradients: grad = loss_function->gradient(pred, target)


// ============================================================================
// LINES 42-46: setOptimizer() - Advanced training strategies
// ============================================================================

void setOptimizer(Optimizer* opt);
// Set optimizer for parameter updates
//
// Parameters:
//   opt: Pointer to optimizer object
//
// Example:
//   network.setOptimizer(new SGD(0.01, 0.9));  // SGD with momentum
//   network.setOptimizer(new Adam(0.001));     // Adam optimizer
//
// Without optimizer: Simple gradient descent
//   W = W - learning_rate × gradient
//
// With optimizer: Smarter updates
//   - Momentum: Accumulates past gradients for smoother updates
//   - Adam: Adapts learning rate per parameter
//   - RMSprop: Divides gradient by running average
```

### Forward and Backward Propagation

```cpp
// ============================================================================
// LINES 48-53: forward() - Compute network output
// ============================================================================

Matrix forward(const Matrix& input);
// Propagate input through all layers to get output
//
// Parameters:
//   input: Input data matrix (batch_size × input_features)
//
// Returns:
//   Output matrix (batch_size × output_features)
//
// What it does internally:
//   Matrix output = input;
//   for each layer in layers:
//       output = layer->forward(output);
//   return output;
//
// Example with 3 layers:
//   Input X (4×784) → Layer0 → H1 (4×128) → Layer1 → H2 (4×64) → Layer2 → Y (4×10)
//
// Pseudocode:
//   forward(X):
//       H1 = layers[0]->forward(X)    // 784 → 128
//       H2 = layers[1]->forward(H1)   // 128 → 64
//       Y  = layers[2]->forward(H2)   // 64 → 10
//       return Y


// ============================================================================
// LINES 55-59: backward() - Compute gradients for learning
// ============================================================================

void backward(const Matrix& loss_gradient);
// Backpropagate gradients through all layers
//
// Parameters:
//   loss_gradient: Gradient from loss function (∂L/∂output)
//
// What it does internally:
//   Matrix gradient = loss_gradient;
//   for layer in reversed(layers):
//       gradient = layer->backward(gradient);
//
// Example with 3 layers:
//   Loss gradient → Layer2 → grad2 → Layer1 → grad1 → Layer0 → grad0
//
// Pseudocode:
//   backward(loss_grad):
//       grad2 = layers[2]->backward(loss_grad)  // Backprop through output layer
//       grad1 = layers[1]->backward(grad2)      // Backprop through hidden layer 2
//       grad0 = layers[0]->backward(grad1)      // Backprop through hidden layer 1
//
// Each layer computes:
//   1. Gradient w.r.t. its weights: ∂L/∂W
//   2. Gradient w.r.t. its biases: ∂L/∂b
//   3. Gradient to pass back: ∂L/∂input (returned)


// ============================================================================
// LINES 61-65: updateParameters() - Apply gradient descent
// ============================================================================

void updateParameters(double learning_rate = 0.01);
// Update all layer parameters using computed gradients
//
// Parameters:
//   learning_rate: Step size for gradient descent (default 0.01)
//
// Two modes:
//
// 1. Simple gradient descent (no optimizer):
//    for each layer:
//        W = W - learning_rate × ∂L/∂W
//        b = b - learning_rate × ∂L/∂b
//
// 2. With optimizer (SGD, Adam, etc.):
//    for each layer:
//        W_new = optimizer->update(W, ∂L/∂W, "layer_i_weights")
//        b_new = optimizer->update(b, ∂L/∂b, "layer_i_biases")
//
// Example:
//   Layer has W = [0.5, 0.3] and ∂L/∂W = [0.02, -0.01]
//   With learning_rate = 0.1:
//     W_new = [0.5, 0.3] - 0.1 × [0.02, -0.01]
//           = [0.5 - 0.002, 0.3 + 0.001]
//           = [0.498, 0.301]
```

### Training Methods

```cpp
// ============================================================================
// LINES 67-74: train() - Main training loop
// ============================================================================

void train(const Matrix& X_train, const Matrix& y_train, 
          int epochs, int batch_size = 32, 
          double learning_rate = 0.01, bool verbose = true);
// Train the network on dataset
//
// Parameters:
//   X_train: Training inputs (num_samples × num_features)
//   y_train: Training targets (num_samples × num_outputs)
//   epochs: Number of times to iterate through entire dataset
//   batch_size: Number of samples per mini-batch (default: 32)
//   learning_rate: Step size for updates (default: 0.01)
//   verbose: Print progress (default: true)
//
// What it does:
//   for epoch in 1..epochs:
//       1. Shuffle data
//       2. Split into mini-batches
//       3. For each batch:
//          a. Forward pass
//          b. Calculate loss
//          c. Backward pass
//          d. Update parameters
//       4. Print progress
//
// Example:
//   network.train(X_train, y_train, 
//                 epochs=100, 
//                 batch_size=32, 
//                 learning_rate=0.01);
//
// Output:
//   Epoch    1/100 - Loss: 0.523456
//   Epoch   10/100 - Loss: 0.234567
//   ...
//   Epoch  100/100 - Loss: 0.012345


// ============================================================================
// KEY CONCEPT: Mini-Batch Training
// ============================================================================
//
// Why mini-batches?
//
// 1. Full Batch (batch_size = all samples):
//    ✓ Accurate gradient
//    ✗ Slow (must process all data before updating)
//    ✗ Large memory requirement
//
// 2. Stochastic (batch_size = 1):
//    ✓ Fast updates
//    ✗ Noisy gradients
//    ✗ Unstable training
//
// 3. Mini-Batch (batch_size = 32, 64, 128, etc.):
//    ✓ Good gradient estimate
//    ✓ Faster than full batch
//    ✓ More stable than stochastic
//    ✓ Efficient GPU utilization
//
// Example with 1000 samples, batch_size=32:
//   Batch 1: samples 0-31
//   Batch 2: samples 32-63
//   ...
//   Batch 32: samples 992-999 (last 8 samples)
//
//   Total batches per epoch: 32
//   Parameters updated 32 times per epoch
```

### Prediction and Evaluation

```cpp
// ============================================================================
// LINES 95-99: predict() - Make predictions on new data
// ============================================================================

Matrix predict(const Matrix& input);
// Make predictions without training
//
// Parameters:
//   input: Input data (num_samples × num_features)
//
// Returns:
//   Predictions (num_samples × num_outputs)
//
// Example:
//   Matrix test_data(10, 784);  // 10 test images
//   Matrix predictions = network.predict(test_data);
//   // predictions: (10 × 10) for 10-class classification
//
// Usage:
//   For binary classification:
//     if (predictions.get(0, 0) > 0.5) { /* class 1 */ } 
//     else { /* class 0 */ }
//
//   For multi-class:
//     int predicted_class = argmax(predictions.get(0, :));


// ============================================================================
// LINES 101-107: evaluate() - Test network performance
// ============================================================================

double evaluate(const Matrix& X_test, const Matrix& y_test);
// Calculate loss on test data
//
// Parameters:
//   X_test: Test inputs
//   y_test: Test targets
//
// Returns:
//   Loss value (double)
//
// Example:
//   double test_loss = network.evaluate(X_test, y_test);
//   std::cout << "Test Loss: " << test_loss << "\n";
//
// Used to check if network generalizes to unseen data


// ============================================================================
// LINES 109-116: accuracy() - Calculate classification accuracy
// ============================================================================

double accuracy(const Matrix& X, const Matrix& y);
// Calculate percentage of correct predictions
//
// Parameters:
//   X: Input data
//   y: True labels (one-hot encoded or class indices)
//
// Returns:
//   Accuracy percentage (0-100)
//
// What it does:
//   1. Make predictions
//   2. For each sample:
//      a. Find predicted class (argmax of predictions)
//      b. Find true class (argmax of targets)
//      c. If same, increment correct counter
//   3. Return (correct / total) × 100
//
// Example:
//   Predictions: [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
//   Targets:     [[0, 1, 0],       [1, 0, 0]]
//   
//   Sample 1: argmax([0.1, 0.8, 0.1]) = 1, argmax([0,1,0]) = 1 → correct ✓
//   Sample 2: argmax([0.7, 0.2, 0.1]) = 0, argmax([1,0,0]) = 0 → correct ✓
//   
//   Accuracy = 2/2 × 100 = 100%
```

---

## Network.cpp - Implementation Details

### Constructor and Setup

```cpp
// ============================================================================
// LINES 12-28: Constructor and basic setup methods
// ============================================================================

// Constructor
NeuralNetwork::NeuralNetwork() : use_optimizer(false) {}
// Creates empty network
// use_optimizer = false means simple gradient descent by default

// Add layer
void NeuralNetwork::addLayer(Layer* layer) {
    layers.push_back(std::unique_ptr<Layer>(layer));
}
// 1. Takes raw pointer
// 2. Wraps in unique_ptr (smart pointer)
// 3. Adds to layers vector
//
// Memory management:
//   - Caller creates: new DenseLayer(...)
//   - Network takes ownership: unique_ptr manages memory
//   - Network destructor automatically deletes all layers

// Set loss function
void NeuralNetwork::setLoss(Loss* loss) {
    loss_function = std::unique_ptr<Loss>(loss);
}
// Similar to addLayer - wraps in unique_ptr

// Set optimizer
void NeuralNetwork::setOptimizer(Optimizer* opt) {
    optimizer = std::unique_ptr<Optimizer>(opt);
    use_optimizer = true;  // Enable optimizer mode
}
```

### Forward Propagation Implementation

```cpp
// ============================================================================
// LINES 30-40: forward() - Data flows through layers
// ============================================================================

Matrix NeuralNetwork::forward(const Matrix& input) {
    // Check if network has layers
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    // Safety check - can't forward pass with no layers!
    
    Matrix output = input;
    // Start with input matrix
    
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    // Pass through each layer sequentially
    // output becomes input to next layer
    
    return output;
    // Final output after all layers
}

// Example execution:
//   Input: X (4×784)
//
//   Iteration 1:
//     output = X (4×784)
//     output = layers[0]->forward(X)  → (4×128)
//
//   Iteration 2:
//     output = (4×128)
//     output = layers[1]->forward(output)  → (4×64)
//
//   Iteration 3:
//     output = (4×64)
//     output = layers[2]->forward(output)  → (4×10)
//
//   Return: output (4×10)
```

### Backward Propagation Implementation

```cpp
// ============================================================================
// LINES 42-50: backward() - Gradients flow backward
// ============================================================================

void NeuralNetwork::backward(const Matrix& loss_gradient) {
    Matrix gradient = loss_gradient;
    // Start with loss gradient from loss function
    
    // Backpropagate through layers in REVERSE order
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }
    // Each layer:
    //   1. Computes ∂L/∂W and ∂L/∂b (stores internally)
    //   2. Returns ∂L/∂input (gradient for previous layer)
}

// Example execution with 3 layers:
//   loss_gradient = ∂L/∂ŷ (4×10)
//
//   i = 2 (Layer 2 - Output):
//     gradient = (4×10)
//     gradient = layers[2]->backward(gradient)  → (4×64)
//     Layer 2 stores: ∂L/∂W₂, ∂L/∂b₂
//
//   i = 1 (Layer 1 - Hidden):
//     gradient = (4×64)
//     gradient = layers[1]->backward(gradient)  → (4×128)
//     Layer 1 stores: ∂L/∂W₁, ∂L/∂b₁
//
//   i = 0 (Layer 0 - Input):
//     gradient = (4×128)
//     gradient = layers[0]->backward(gradient)  → (4×784)
//     Layer 0 stores: ∂L/∂W₀, ∂L/∂b₀
//
// After backward(), all layers have gradients ready for update
```

### Parameter Updates

```cpp
// ============================================================================
// LINES 52-87: updateParameters() - Apply gradient descent
// ============================================================================

void NeuralNetwork::updateParameters(double learning_rate) {
    if (use_optimizer && optimizer) {
        // ─── OPTIMIZER MODE ───────────────────────────────────────
        
        for (size_t i = 0; i < layers.size(); ++i) {
            // Cast to DenseLayer to access weight methods
            DenseLayer* dense = dynamic_cast<DenseLayer*>(layers[i].get());
            
            if (dense) {
                // Update weights using optimizer
                std::string weight_id = "layer" + std::to_string(i) + "_weights";
                Matrix new_weights = optimizer->update(
                    dense->getWeights(),         // Current weights
                    dense->getWeightGradients(), // Gradients
                    weight_id                    // Unique ID for momentum/state
                );
                dense->setWeights(new_weights);
                
                // Update biases using optimizer
                std::string bias_id = "layer" + std::to_string(i) + "_biases";
                Matrix new_biases = optimizer->update(
                    dense->getBiases(),
                    dense->getBiasGradients(),
                    bias_id
                );
                dense->setBiases(new_biases);
                
                // Reset gradients for next iteration
                dense->resetGradients();
            }
        }
        
    } else {
        // ─── SIMPLE GRADIENT DESCENT MODE ─────────────────────────
        
        for (auto& layer : layers) {
            layer->updateParameters(learning_rate);
            // Each layer does: W = W - lr × ∂L/∂W
        }
    }
}

// Why two modes?
//
// Simple Gradient Descent:
//   ✓ Easy to understand
//   ✓ Works for simple problems
//   ✗ Can be slow to converge
//   ✗ Sensitive to learning rate
//
// With Optimizer (Adam, SGD+Momentum):
//   ✓ Faster convergence
//   ✓ Adaptive learning rates
//   ✓ Handles noisy gradients better
//   ✗ More complex
//   ✗ Requires tuning hyperparameters
```

### Training Loop Implementation

```cpp
// ============================================================================
// LINES 154-204: train() - Complete training procedure
// ============================================================================

void NeuralNetwork::train(const Matrix& X_train, const Matrix& y_train, 
                         int epochs, int batch_size, 
                         double learning_rate, bool verbose) {
    
    // Check loss function is set
    if (!loss_function) {
        throw std::runtime_error("Loss function not set");
    }
    
    // ─── EPOCH LOOP ───────────────────────────────────────────────
    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        // Step 1: Shuffle data
        Matrix X_shuffled = X_train;
        Matrix y_shuffled = y_train;
        shuffleData(X_shuffled, y_shuffled);
        // Why shuffle? Prevents network from learning order of samples
        
        // Step 2: Create mini-batches
        auto batches = createBatches(X_shuffled, y_shuffled, batch_size);
        // Splits data into chunks of size batch_size
        // Example: 1000 samples, batch_size=32 → 32 batches
        
        double total_loss = 0.0;
        
        // ─── BATCH LOOP ───────────────────────────────────────────
        for (const auto& batch : batches) {
            const Matrix& X_batch = batch.first;   // Input batch
            const Matrix& y_batch = batch.second;  // Target batch
            
            // Step 3: Forward pass
            Matrix predictions = forward(X_batch);
            // Run input through network to get predictions
            
            // Step 4: Compute loss
            double batch_loss = loss_function->calculate(predictions, y_batch);
            total_loss += batch_loss * X_batch.getRows();
            // Accumulate loss weighted by batch size
            
            // Step 5: Backward pass
            Matrix loss_grad = loss_function->gradient(predictions, y_batch);
            backward(loss_grad);
            // Compute gradients for all layers
            
            // Step 6: Update parameters
            updateParameters(learning_rate);
            // Apply gradient descent
        }
        
        // Step 7: Calculate average loss for epoch
        double avg_loss = total_loss / X_train.getRows();
        
        // Step 8: Print progress
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << epochs 
                     << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss 
                     << std::endl;
        }
    }
}

// Complete training cycle visualization:
//
// Epoch 1:
//   Shuffle → Batch 1 → Forward → Loss → Backward → Update
//          → Batch 2 → Forward → Loss → Backward → Update
//          → ...
//          → Batch 32 → Forward → Loss → Backward → Update
//   Print: "Epoch 1/100 - Loss: 0.523"
//
// Epoch 2:
//   Shuffle → Batch 1 → ...
//   ...
//
// Epoch 100:
//   Shuffle → Batch 1 → ...
//   Print: "Epoch 100/100 - Loss: 0.012"
```

---

## Complete Network Flow Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK COMPLETE FLOW                      │
└────────────────────────────────────────────────────────────────────┘

BUILDING PHASE:
═══════════════

NeuralNetwork network;

network.addLayer(new DenseLayer(784, 128, new ReLU()));
         ↓
    layers[0] = DenseLayer
         ├─ weights: (128×784)
         ├─ biases: (128×1)
         └─ activation: ReLU

network.addLayer(new DenseLayer(128, 64, new ReLU()));
         ↓
    layers[1] = DenseLayer
         ├─ weights: (64×128)
         ├─ biases: (64×1)
         └─ activation: ReLU

network.addLayer(new DenseLayer(64, 10, new Sigmoid()));
         ↓
    layers[2] = DenseLayer
         ├─ weights: (10×64)
         ├─ biases: (10×1)
         └─ activation: Sigmoid

network.setLoss(new MSELoss());
         ↓
    loss_function = MSELoss

network.setOptimizer(new Adam(0.001));
         ↓
    optimizer = Adam
    use_optimizer = true


TRAINING PHASE:
═══════════════

network.train(X_train, y_train, epochs=100, batch_size=32, lr=0.01);

For each epoch:
  │
  ├─ 1. Shuffle data
  │    X_train, y_train → randomize order
  │
  ├─ 2. Create batches
  │    1000 samples, batch_size=32 → 32 batches
  │
  └─ 3. For each batch:
       │
       ├─ a. Forward Pass
       │    X_batch (32×784)
       │       ↓
       │    layers[0]->forward(X)
       │       ↓ ReLU(X·W₀ᵀ + b₀)
       │    H1 (32×128)
       │       ↓
       │    layers[1]->forward(H1)
       │       ↓ ReLU(H1·W₁ᵀ + b₁)
       │    H2 (32×64)
       │       ↓
       │    layers[2]->forward(H2)
       │       ↓ Sigmoid(H2·W₂ᵀ + b₂)
       │    ŷ (32×10)
       │
       ├─ b. Calculate Loss
       │    loss = MSELoss->calculate(ŷ, y_batch)
       │         = (1/32) × Σ(y - ŷ)²
       │
       ├─ c. Backward Pass
       │    loss_grad = MSELoss->gradient(ŷ, y_batch)
       │              = -2(y - ŷ) / 32
       │       ↓
       │    layers[2]->backward(loss_grad)
       │       ↓ Compute ∂L/∂W₂, ∂L/∂b₂, return ∂L/∂H2
       │    grad2 (32×64)
       │       ↓
       │    layers[1]->backward(grad2)
       │       ↓ Compute ∂L/∂W₁, ∂L/∂b₁, return ∂L/∂H1
       │    grad1 (32×128)
       │       ↓
       │    layers[0]->backward(grad1)
       │       ↓ Compute ∂L/∂W₀, ∂L/∂b₀, return ∂L/∂X
       │    grad0 (32×784)
       │
       └─ d. Update Parameters
            For each layer:
              W_new = optimizer->update(W, ∂L/∂W, "layer_i_weights")
              b_new = optimizer->update(b, ∂L/∂b, "layer_i_biases")


PREDICTION PHASE:
═════════════════

Matrix predictions = network.predict(X_test);

  X_test (10×784)
     ↓
  forward(X_test)
     ↓
  layers[0]->forward → (10×128)
     ↓
  layers[1]->forward → (10×64)
     ↓
  layers[2]->forward → (10×10)
     ↓
  predictions (10×10)


EVALUATION PHASE:
═════════════════

double test_loss = network.evaluate(X_test, y_test);
  = loss_function->calculate(network.forward(X_test), y_test)

double acc = network.accuracy(X_test, y_test);
  = (correct_predictions / total_samples) × 100
```

---

## Complete Working Example

Now let me create a complete example program that shows everything working together:

```cpp
#include "nn/network.h"
#include "nn/layer.h"
#include "nn/activation.h"
#include "nn/loss.h"
#include "nn/optimizer.h"
#include <iostream>

int main() {
    std::cout << "=== Building Neural Network ===" << std::endl;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 1: Create Network Object
    // ═══════════════════════════════════════════════════════════
    NeuralNetwork network;
    // Empty container ready to hold layers
    
    // ═══════════════════════════════════════════════════════════
    // STEP 2: Add Layers (Build Architecture)
    // ═══════════════════════════════════════════════════════════
    
    // Input Layer → Hidden Layer 1
    network.addLayer(new DenseLayer(2, 4, new ReLU()));
    // 2 inputs (x₁, x₂)
    // 4 hidden neurons
    // ReLU activation
    // Parameters: 2×4 + 4 = 12
    
    // Hidden Layer 1 → Output Layer
    network.addLayer(new DenseLayer(4, 1, new Sigmoid()));
    // 4 inputs (from previous layer)
    // 1 output (binary classification)
    // Sigmoid activation (outputs probability)
    // Parameters: 4×1 + 1 = 5
    
    std::cout << "Network structure: 2 → 4 → 1" << std::endl;
    std::cout << "Total parameters: 12 + 5 = 17" << std::endl;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 3: Set Loss Function
    // ═══════════════════════════════════════════════════════════
    network.setLoss(new MSELoss());
    // Mean Squared Error for regression/binary classification
    
    // ═══════════════════════════════════════════════════════════
    // STEP 4: Set Optimizer (Optional)
    // ═══════════════════════════════════════════════════════════
    network.setOptimizer(new SGD(0.1, 0.9));
    // SGD with learning_rate=0.1, momentum=0.9
    
    // ═══════════════════════════════════════════════════════════
    // STEP 5: Prepare Training Data (XOR Problem)
    // ═══════════════════════════════════════════════════════════
    
    Matrix X_train(4, 2);
    X_train.set(0, 0, 0); X_train.set(0, 1, 0);  // [0, 0]
    X_train.set(1, 0, 0); X_train.set(1, 1, 1);  // [0, 1]
    X_train.set(2, 0, 1); X_train.set(2, 1, 0);  // [1, 0]
    X_train.set(3, 0, 1); X_train.set(3, 1, 1);  // [1, 1]
    
    Matrix y_train(4, 1);
    y_train.set(0, 0, 0);  // 0 XOR 0 = 0
    y_train.set(1, 0, 1);  // 0 XOR 1 = 1
    y_train.set(2, 0, 1);  // 1 XOR 0 = 1
    y_train.set(3, 0, 0);  // 1 XOR 1 = 0
    
    std::cout << "\n=== Training Network ===" << std::endl;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 6: Train the Network
    // ═══════════════════════════════════════════════════════════
    network.train(
        X_train,        // Training inputs
        y_train,        // Training targets
        1000,           // epochs
        4,              // batch_size (use all 4 samples)
        0.1,            // learning_rate
        true            // verbose (print progress)
    );
    
    // What happens during training:
    // 
    // Epoch 1:
    //   Shuffle data
    //   Batch 1 (all 4 samples):
    //     Forward:  X → Layer0 → H1 → Layer1 → ŷ
    //     Loss:     L = MSE(ŷ, y) = high (random weights)
    //     Backward: ∂L/∂ŷ → Layer1 → Layer0
    //     Update:   W = W - lr × ∂L/∂W
    //   Print: "Epoch 1/1000 - Loss: 0.250000"
    //
    // Epoch 10:
    //   (same process)
    //   Print: "Epoch 10/1000 - Loss: 0.210000"
    //
    // ...
    //
    // Epoch 1000:
    //   (same process)
    //   Print: "Epoch 1000/1000 - Loss: 0.000800"
    
    std::cout << "\n=== Testing Network ===" << std::endl;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 7: Make Predictions
    // ═══════════════════════════════════════════════════════════
    Matrix predictions = network.predict(X_train);
    
    std::cout << "\nPredictions:" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        std::cout << "Input: [" << X_train.get(i,0) << ", " 
                  << X_train.get(i,1) << "] → ";
        std::cout << "Predicted: " << std::fixed << std::setprecision(4) 
                  << predictions.get(i,0);
        std::cout << " | Target: " << y_train.get(i,0);
        
        if (std::abs(predictions.get(i,0) - y_train.get(i,0)) < 0.1) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // STEP 8: Evaluate Performance
    // ═══════════════════════════════════════════════════════════
    double test_loss = network.evaluate(X_train, y_train);
    std::cout << "\nTest Loss: " << test_loss << std::endl;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 9: Display Network Summary
    // ═══════════════════════════════════════════════════════════
    network.summary();
    
    // Output:
    // ========== Neural Network Summary ==========
    // Total Layers: 2
    // Optimizer: SGD
    // Loss Function: MSELoss
    //
    // --- Layer Details ---
    // Layer 1: DenseLayer (2 -> 4)
    // Layer 2: DenseLayer (4 -> 1)
    // ============================================
    
    return 0;
}
```

---

## Key Concepts Summary

```
┌────────────────────────────────────────────────────────────────────┐
│  NEURAL NETWORK CLASS: THE BIG PICTURE                            │
└────────────────────────────────────────────────────────────────────┘

1. Network = Container
   ─────────────────────
   • Holds multiple layers (vector<Layer*>)
   • Holds loss function (Loss*)
   • Holds optimizer (Optimizer*)
   • Orchestrates training process

2. Layers = Building Blocks
   ─────────────────────────
   • Each layer transforms input to output
   • Layers connected sequentially
   • Output of layer i = input of layer i+1

3. Forward Pass = Prediction
   ──────────────────────────
   • Data flows INPUT → Layer0 → Layer1 → ... → OUTPUT
   • Each layer: output = activation(input·W + b)
   • Returns final prediction

4. Loss Function = Error Measure
   ───────────────────────────────
   • Compares prediction to target
   • Returns scalar: how wrong we are
   • Provides gradient for backprop

5. Backward Pass = Learning
   ───────────────────────────
   • Gradients flow OUTPUT → ... → Layer1 → Layer0 → INPUT
   • Each layer computes ∂L/∂W and ∂L/∂b
   • Chain rule connects all gradients

6. Update Parameters = Improvement
   ──────────────────────────────────
   • Use computed gradients to adjust weights
   • W_new = W_old - learning_rate × ∂L/∂W
   • Optimizer makes this smarter (momentum, adaptive rates)

7. Training Loop = Repeated Learning
   ────────────────────────────────────
   • Repeat many times (epochs):
     1. Shuffle data
     2. Split into batches
     3. Forward → Loss → Backward → Update
   • Loss decreases, accuracy increases

8. Prediction = Using Trained Network
   ────────────────────────────────────
   • Forward pass without backward
   • No parameter updates
   • Just get predictions for new data
```

---

## Memory Management Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│  HOW NETWORK MANAGES MEMORY                                        │
└────────────────────────────────────────────────────────────────────┘

CREATION:
═════════

NeuralNetwork network;
   │
   ├─ layers (empty vector)
   ├─ loss_function (null)
   └─ optimizer (null)


ADDING LAYERS:
══════════════

network.addLayer(new DenseLayer(2, 4, new ReLU()));
   │
   │ Create on heap:
   │   DenseLayer* layer = new DenseLayer(2, 4, new ReLU());
   │                          │
   │                          ├─ Allocates W (4×2)
   │                          ├─ Allocates b (4×1)
   │                          └─ Stores ReLU* activation
   │
   └─ Network wraps in unique_ptr:
        layers.push_back(unique_ptr<Layer>(layer))
        
        Memory ownership transferred to network!
        Network will delete layer when destroyed


DESTRUCTION:
════════════

} // network goes out of scope

network destructor called:
   │
   ├─ layers vector destroyed
   │    │
   │    ├─ layers[0].~unique_ptr()
   │    │    └─ delete DenseLayer
   │    │         └─ DenseLayer destructor
   │    │              ├─ delete activation (ReLU)
   │    │              ├─ free W memory
   │    │              └─ free b memory
   │    │
   │    ├─ layers[1].~unique_ptr()
   │    │    └─ (same process)
   │    │
   │    └─ ...
   │
   ├─ loss_function.~unique_ptr()
   │    └─ delete MSELoss
   │
   └─ optimizer.~unique_ptr()
        └─ delete Adam

All memory automatically freed!
No memory leaks!
```

---

This comprehensive guide shows you how the `NeuralNetwork` class ties everything together - layers, activations, loss, and training - into a complete machine learning system!
