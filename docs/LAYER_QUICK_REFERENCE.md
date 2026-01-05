# Neural Network Layer - Quick Reference

## 📋 Table of Contents
- [Basic Concepts](#basic-concepts)
- [Layer Architecture](#layer-architecture)
- [Forward Pass](#forward-pass)
- [Backward Pass](#backward-pass)
- [Weight Initialization](#weight-initialization)
- [Common Usage Patterns](#common-usage-patterns)
- [Troubleshooting](#troubleshooting)

---

## Basic Concepts

### What is a Dense Layer?
A **fully connected layer** where every input connects to every output.

```
Input (3)      Dense Layer (4)      Output (4)
   x1 ──┬──────→ o1
        │   ┌───→ o2
   x2 ──┼───┼───→ o3
        │   │  └→ o4
   x3 ──┴───┴────→

Each output receives all inputs with different weights
```

### Parameters
```
Total Parameters = (input_size × output_size) + output_size
                   \_________Weights________/   \__Biases__/

Example: 100 inputs → 50 outputs
Parameters = (100 × 50) + 50 = 5,050
```

---

## Layer Architecture

### Class Hierarchy
```cpp
Layer (Abstract Base)
   ├── forward()          // Pure virtual
   ├── backward()         // Pure virtual
   └── updateParameters() // Pure virtual

DenseLayer : public Layer
   ├── weights            // (output × input) matrix
   ├── biases            // (output × 1) vector
   ├── weight_gradients  // Same shape as weights
   ├── bias_gradients    // Same shape as biases
   ├── cached_input      // Saved for backward pass
   ├── cached_z          // Pre-activation values
   └── activation        // Optional activation function
```

### Creating a Layer
```cpp
// 100 inputs → 50 outputs with ReLU activation
DenseLayer layer(100, 50, new ReLU());

// Properties
layer.getName();            // Returns "DenseLayer"
layer.getInputSize();       // Returns 100
layer.getOutputSize();      // Returns 50
layer.getParameterCount();  // Returns 5,050
```

---

## Forward Pass

### Formula
```
Z = X·W^T + b    (Linear transformation)
A = σ(Z)         (Activation function)

Where:
  X: Input (batch_size × input_size)
  W: Weights (output_size × input_size)
  b: Biases (output_size × 1)
  Z: Pre-activation (batch_size × output_size)
  A: Output (batch_size × output_size)
```

### Implementation
```cpp
Matrix input(1, 100);  // 1 sample, 100 features
// ... fill input ...

Matrix output = layer.forward(input);
// output shape: (1 × 50)
```

### What Happens Inside
```cpp
1. Z = input * weights.transpose() + biases
   - Matrix multiplication
   - Broadcast addition of biases

2. Cache input and Z for backward pass

3. If activation exists:
   A = activation->activate(Z)
   else:
   A = Z (linear layer)

4. Return A
```

---

## Backward Pass

### Formula
```
Given ∂L/∂A (gradient from next layer):

1. Activation gradient:
   ∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
   (⊙ = element-wise multiplication)

2. Weight gradient:
   ∂L/∂W = (∂L/∂Z)^T · X

3. Bias gradient:
   ∂L/∂b = sum(∂L/∂Z, axis=0)

4. Input gradient (for previous layer):
   ∂L/∂X = (∂L/∂Z) · W
```

### Implementation
```cpp
Matrix output_gradient(1, 50);
// ... gradient from next layer or loss ...

Matrix input_gradient = layer.backward(output_gradient);
// input_gradient shape: (1 × 100)

// Access computed gradients
Matrix dW = layer.getWeightGradients();
Matrix db = layer.getBiasGradients();
```

### Gradient Dimensions
```
If layer is (100 → 50):
  weight_gradients: (50 × 100)  [same as weights]
  bias_gradients:   (50 × 1)    [same as biases]
  input_gradient:   (1 × 100)   [same as input]
```

---

## Weight Initialization

### Available Strategies

#### 1. Xavier (Glorot) Initialization
```cpp
layer.initializeWeights("xavier");

// Formula: W ~ N(0, 2/(n_in + n_out))
// Best for: Sigmoid, Tanh activations
// Keeps variance stable across layers
```

#### 2. He Initialization
```cpp
layer.initializeWeights("he");

// Formula: W ~ N(0, 2/n_in)
// Best for: ReLU, LeakyReLU activations
// Accounts for ReLU killing half the neurons
```

#### 3. Random Initialization
```cpp
layer.initializeWeights("random");

// Formula: W ~ U(-1, 1)
// Uniform random between -1 and 1
```

#### 4. Zero Initialization
```cpp
layer.initializeWeights("zeros");

// ⚠️ WARNING: Causes symmetry problem!
// All neurons learn the same thing
// Only use for debugging
```

### Default Initialization
```cpp
DenseLayer layer(100, 50, new ReLU());
// Automatically uses Xavier initialization
```

### Choosing Initialization

| Activation | Recommended | Why? |
|-----------|-------------|------|
| Sigmoid   | Xavier      | Symmetric around 0 |
| Tanh      | Xavier      | Symmetric around 0 |
| ReLU      | He          | Accounts for dead neurons |
| LeakyReLU | He          | Similar to ReLU |
| Linear    | Xavier      | Safe default |

---

## Common Usage Patterns

### Pattern 1: Single Layer
```cpp
DenseLayer layer(784, 10, new Softmax());

// Forward pass
Matrix output = layer.forward(input);

// Backward pass
Matrix input_grad = layer.backward(output_grad);

// Update weights
layer.updateParameters(learning_rate);
```

### Pattern 2: Multi-Layer Network
```cpp
// Create layers
DenseLayer hidden1(784, 128, new ReLU());
DenseLayer hidden2(128, 64, new ReLU());
DenseLayer output(64, 10, new Softmax());

// Forward pass
Matrix h1 = hidden1.forward(input);
Matrix h2 = hidden2.forward(h1);
Matrix pred = output.forward(h2);

// Backward pass (reverse order!)
Matrix grad3 = output.backward(loss_grad);
Matrix grad2 = hidden2.backward(grad3);
Matrix grad1 = hidden1.backward(grad2);

// Update all layers
output.updateParameters(lr);
hidden2.updateParameters(lr);
hidden1.updateParameters(lr);
```

### Pattern 3: Training Loop
```cpp
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& [input, target] : dataset) {
        // Forward
        Matrix h1 = layer1.forward(input);
        Matrix pred = layer2.forward(h1);
        
        // Loss
        double loss = loss_fn.calculate(pred, target);
        Matrix loss_grad = loss_fn.gradient(pred, target);
        
        // Backward
        Matrix grad2 = layer2.backward(loss_grad);
        Matrix grad1 = layer1.backward(grad2);
        
        // Update
        layer2.updateParameters(learning_rate);
        layer1.updateParameters(learning_rate);
    }
}
```

### Pattern 4: Manual Weight Setting
```cpp
DenseLayer layer(3, 2);

// Set custom weights
Matrix W(2, 3);
W.set(0, 0, 0.5); W.set(0, 1, 0.3); W.set(0, 2, 0.2);
W.set(1, 0, 0.4); W.set(1, 1, 0.6); W.set(1, 2, 0.1);
layer.setWeights(W);

// Set custom biases
Matrix b(2, 1);
b.set(0, 0, 0.1);
b.set(1, 0, 0.2);
layer.setBiases(b);
```

### Pattern 5: Gradient Accumulation
```cpp
layer.resetGradients();  // Clear gradients

for (int i = 0; i < batch_size; i++) {
    Matrix output = layer.forward(inputs[i]);
    layer.backward(gradients[i]);
    // Gradients accumulate!
}

// Update with accumulated gradients
layer.updateParameters(learning_rate);
layer.resetGradients();  // Clear for next batch
```

---

## Parameter Updates

### Simple Gradient Descent
```cpp
// θ = θ - α·∇θ
layer.updateParameters(0.01);  // learning_rate = 0.01

// Internally:
// weights = weights - learning_rate * weight_gradients
// biases = biases - learning_rate * bias_gradients
```

### Manual Updates (for custom optimizers)
```cpp
// Get current parameters
Matrix W = layer.getWeights();
Matrix b = layer.getBiases();

// Get gradients
Matrix dW = layer.getWeightGradients();
Matrix db = layer.getBiasGradients();

// Apply custom update (e.g., with momentum)
W = W - learning_rate * dW + momentum * prev_update;
b = b - learning_rate * db + momentum * prev_bias_update;

// Set updated parameters
layer.setWeights(W);
layer.setBiases(b);

// Clear gradients
layer.resetGradients();
```

---

## Troubleshooting

### Problem: Exploding Gradients
**Symptoms:** Loss becomes NaN, weights grow very large

**Solutions:**
```cpp
1. Use proper initialization
   layer.initializeWeights("xavier");  // or "he"

2. Lower learning rate
   layer.updateParameters(0.001);  // Instead of 0.1

3. Add gradient clipping (manual)
   Matrix dW = layer.getWeightGradients();
   dW = dW.clip(-1.0, 1.0);  // If clip() exists

4. Check for bugs in loss function
```

### Problem: Vanishing Gradients
**Symptoms:** Loss doesn't decrease, weights barely change

**Solutions:**
```cpp
1. Use ReLU instead of Sigmoid/Tanh
   DenseLayer layer(100, 50, new ReLU());

2. Use He initialization with ReLU
   layer.initializeWeights("he");

3. Increase learning rate
   layer.updateParameters(0.01);  // Instead of 0.001

4. Add skip connections (ResNet-style)
```

### Problem: Symmetry Problem
**Symptoms:** All neurons output the same value

**Solutions:**
```cpp
1. Never initialize with zeros!
   layer.initializeWeights("zeros");  // ❌ WRONG

2. Use random initialization
   layer.initializeWeights("xavier");  // ✓ CORRECT

3. Check that weights are different
   Matrix W = layer.getWeights();
   // All rows should be different
```

### Problem: Overfitting
**Symptoms:** Training loss decreases but validation doesn't

**Solutions:**
```cpp
1. Use smaller network
   DenseLayer layer(100, 10);  // Instead of (100, 1000)

2. Add dropout (if implemented)
   DenseLayer layer(100, 50, new ReLU(), 0.5);

3. Reduce learning rate
   layer.updateParameters(0.001);

4. Use early stopping
```

### Problem: Wrong Output Shape
**Symptoms:** Matrix dimension mismatch errors

**Solutions:**
```cpp
1. Check input dimensions
   DenseLayer layer(784, 128);
   Matrix input(1, 784);  // Must match layer input size
   
2. Check chaining layers
   DenseLayer layer1(784, 128);
   DenseLayer layer2(128, 10);  // 128 must match previous output
   
3. Use getters to verify
   std::cout << layer.getInputSize() << "\n";
   std::cout << layer.getOutputSize() << "\n";
```

### Problem: Gradients Not Updating
**Symptoms:** Weights don't change during training

**Solutions:**
```cpp
1. Ensure backward is called
   Matrix output = layer.forward(input);
   Matrix grad = layer.backward(output_grad);  // Don't forget!
   
2. Check learning rate isn't zero
   layer.updateParameters(0.01);  // Not 0.0!
   
3. Verify gradients exist
   Matrix dW = layer.getWeightGradients();
   std::cout << "Gradient norm: " << dW.norm() << "\n";
   
4. Reset gradients between batches
   layer.resetGradients();
```

---

## Performance Tips

### Memory Usage
```cpp
// Parameters memory = 4 bytes × parameter_count
DenseLayer layer(1000, 1000);
// Memory: 4 × (1000×1000 + 1000) = ~4 MB

// Reduce size if memory limited
DenseLayer layer(1000, 100);  // ~400 KB
```

### Computational Cost
```cpp
// Forward pass: O(batch_size × input_size × output_size)
// Backward pass: O(batch_size × input_size × output_size)

// For layer (784 → 1000) with batch 32:
// Forward:  32 × 784 × 1000 = 25M operations
// Backward: 32 × 784 × 1000 = 25M operations
```

### Optimization Tips
```cpp
1. Batch inputs together
   Matrix batch(32, 784);  // 32 samples at once
   layer.forward(batch);   // More efficient than 32 individual calls

2. Reuse layer objects
   // Don't create/destroy layers in loop
   for (int i = 0; i < epochs; i++) {
       layer.forward(input);  // ✓
   }

3. Clear gradients properly
   layer.resetGradients();  // After each update
```

---

## Quick Formulas Reference

### Forward Pass
```
Z = X·W^T + b
A = σ(Z)
```

### Backward Pass
```
∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
∂L/∂W = (∂L/∂Z)^T · X
∂L/∂b = sum(∂L/∂Z, axis=0)
∂L/∂X = (∂L/∂Z) · W
```

### Parameter Update
```
W = W - α·∂L/∂W
b = b - α·∂L/∂b
```

### Initialization Variance
```
Xavier: Var(W) = 2 / (n_in + n_out)
He:     Var(W) = 2 / n_in
```

---

## Complete Example
```cpp
#include "../include/nn/layer.h"
#include "../include/nn/activation.h"

int main() {
    // Create layer: 784 inputs → 128 outputs with ReLU
    DenseLayer layer(784, 128, new ReLU());
    
    // Initialize with He initialization (good for ReLU)
    layer.initializeWeights("he");
    
    // Create input (28×28 image flattened)
    Matrix input(1, 784);
    // ... fill input with image data ...
    
    // Forward pass
    Matrix output = layer.forward(input);
    // output shape: (1 × 128)
    
    // Backward pass (gradient from next layer)
    Matrix output_gradient(1, 128);
    // ... gradient from loss or next layer ...
    
    Matrix input_gradient = layer.backward(output_gradient);
    // input_gradient shape: (1 × 784)
    
    // Update parameters
    double learning_rate = 0.01;
    layer.updateParameters(learning_rate);
    
    // Clear gradients for next batch
    layer.resetGradients();
    
    return 0;
}
```

---

## See Also
- **LAYER_COMPLETE_GUIDE.md** - Detailed explanations with examples
- **layer_example.cpp** - 7 complete working examples
- **activation.h** - Available activation functions
- **optimizer.h** - Advanced optimization algorithms

---

**Quick Tips:**
- ✓ Always initialize weights randomly (Xavier or He)
- ✓ Match layer dimensions: layer1.output = layer2.input
- ✓ Call backward before updateParameters
- ✓ Reset gradients after updating
- ✗ Never initialize with zeros
- ✗ Never skip the backward pass
- ✗ Never forget to update parameters

**Happy Learning! 🚀**
