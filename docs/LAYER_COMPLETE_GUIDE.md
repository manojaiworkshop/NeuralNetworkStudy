# Neural Network Layer Implementation - Complete Guide

## Table of Contents
1. [What Are Layers?](#what-are-layers)
2. [layer.h - Header Explanation](#layerh---header-explanation)
3. [layer.cpp - Implementation Explanation](#layercpp---implementation-explanation)
4. [Mathematical Foundation](#mathematical-foundation)
5. [How Forward/Backward Pass Works](#how-forwardbackward-pass-works)
6. [Weight Initialization Strategies](#weight-initialization-strategies)
7. [Practical Examples](#practical-examples)

---

## What Are Layers?

### The Building Blocks

Neural networks are built by stacking layers:

```
Input → Layer 1 → Layer 2 → Layer 3 → Output
         ↓         ↓         ↓
      Weights   Weights   Weights
      Biases    Biases    Biases
      Activation Activation Activation
```

### What Does a Layer Do?

**Forward Pass** (Prediction):
```
Input → [Layer Computation] → Output
         • Matrix multiply
         • Add bias
         • Apply activation
```

**Backward Pass** (Learning):
```
Output ← [Gradient Computation] ← Error
         • Compute parameter gradients
         • Propagate error backwards
```

### Dense Layer (Fully Connected)

```
Each neuron connects to ALL inputs:

Input Layer (3 neurons)    Dense Layer (4 neurons)
       ○                          ○
       ○ ─────────────────────────○
       ○ ─────────────────────────○
                                  ○

Total connections: 3 × 4 = 12 weights
Plus 4 biases = 16 parameters
```

---

## layer.h - Header Explanation

### Include Guard and Dependencies

```cpp
#ifndef LAYER_H
#define LAYER_H
```
**Purpose**: Prevent multiple inclusion
- Standard C++ header guard
- Ensures file included only once

---

```cpp
#include "matrix.h"
#include "activation.h"
#include <memory>
#include <string>
```
**Dependencies**:
- `matrix.h` - Matrix operations (weights, inputs, outputs)
- `activation.h` - Activation functions (ReLU, Sigmoid, etc.)
- `<memory>` - Smart pointers (std::unique_ptr)
- `<string>` - Layer naming

---

### Base Layer Class

```cpp
class Layer {
public:
    virtual ~Layer() = default;
```
**Abstract Base Class**:
- `virtual ~Layer()` - Virtual destructor for proper cleanup
- `= default` - Use compiler-generated destructor
- All methods are pure virtual (= 0)

**Why virtual destructor?**
```cpp
Layer* layer = new DenseLayer(10, 5);
delete layer;  // Calls DenseLayer destructor (because virtual)
```

---

```cpp
    virtual Matrix forward(const Matrix& input) = 0;
```
**Forward Pass** - Pure virtual method

**Purpose**: Transform input to output
```
Example:
  Input: [1, 2, 3]  (3 features)
  Output: [4, 5]     (2 neurons)
```

**What happens inside?**
```
1. Z = X·W^T + b     (linear transformation)
2. A = activation(Z) (non-linearity)
```

**Return**: Output matrix after transformation

---

```cpp
    virtual Matrix backward(const Matrix& output_gradient) = 0;
```
**Backward Pass** - Compute gradients

**Purpose**: Propagate error backwards for learning

**Parameters**:
- `output_gradient` - Error signal from next layer (∂Loss/∂Output)

**Return**: 
- Gradient with respect to input (∂Loss/∂Input)

**What happens inside?**
```
1. Compute ∂Loss/∂Weights
2. Compute ∂Loss/∂Biases  
3. Compute ∂Loss/∂Input (for previous layer)
```

**Example**:
```
Layer 2 output gradient: [0.1, 0.2]
         ↓
Layer 2 backward computes:
  • Weight gradients (to update Layer 2)
  • Input gradient (to pass to Layer 1)
         ↓
Layer 1 receives: [0.05, 0.03, 0.08]
```

---

```cpp
    virtual void updateParameters(double learning_rate) = 0;
```
**Update Weights and Biases**

**Purpose**: Apply gradients to learn
```
Formula: θ_new = θ_old - learning_rate × gradient
```

**Example**:
```cpp
// After backward() computed gradients
layer->updateParameters(0.01);  // learning_rate = 0.01

// Weights updated:
W_new = W_old - 0.01 × ∂Loss/∂W
b_new = b_old - 0.01 × ∂Loss/∂b
```

---

```cpp
    virtual std::string getName() const = 0;
```
**Get Layer Type**: Returns "Dense", "Conv", "LSTM", etc.

**Usage**:
```cpp
std::cout << "Layer type: " << layer->getName() << "\n";
// Output: Layer type: Dense
```

---

```cpp
    virtual size_t getInputSize() const = 0;
    virtual size_t getOutputSize() const = 0;
```
**Dimension Information**:
- `getInputSize()` - Number of input features
- `getOutputSize()` - Number of output neurons

**Example**:
```cpp
DenseLayer layer(784, 128);  // MNIST: 28×28 = 784 pixels
std::cout << "Input: " << layer.getInputSize() << "\n";   // 784
std::cout << "Output: " << layer.getOutputSize() << "\n"; // 128
```

---

```cpp
    virtual int getParameterCount() const = 0;
```
**Count Trainable Parameters**

**For Dense Layer**:
```
Parameters = (input_size × output_size) + output_size
           = weights + biases

Example:
  input = 784, output = 128
  weights = 784 × 128 = 100,352
  biases = 128
  total = 100,480 parameters
```

**Why important?**
- Memory usage estimation
- Model complexity analysis
- Debugging

---

```cpp
    virtual Matrix getWeights() const { return Matrix(); }
    virtual Matrix getBiases() const { return Matrix(); }
```
**Get Parameters** - Optional (default returns empty)

**Usage**:
```cpp
Matrix W = layer.getWeights();
Matrix b = layer.getBiases();

// Inspect learned values
std::cout << "First weight: " << W.get(0, 0) << "\n";
```

---

```cpp
    virtual void setWeights(const Matrix& weights) {}
    virtual void setBiases(const Matrix& biases) {}
```
**Set Parameters** - For transfer learning or loading saved models

**Example**:
```cpp
// Load pre-trained weights
Matrix pretrained_weights = loadFromFile("weights.txt");
layer.setWeights(pretrained_weights);

// Fine-tune on new data
train(layer, new_data);
```

---

### DenseLayer Class

```cpp
class DenseLayer : public Layer {
private:
    size_t input_size;
    size_t output_size;
```
**Dimensions**:
- `input_size` - Number of input features
- `output_size` - Number of neurons in this layer

**Example**:
```cpp
DenseLayer layer(10, 5);
// 10 inputs → 5 neurons
```

---

```cpp
    Matrix weights;           // (output_size x input_size)
    Matrix biases;            // (output_size x 1)
```
**Trainable Parameters**:

**Weights Matrix**:
```
Shape: (output_size × input_size)

Example: 3 inputs → 2 outputs
weights = [[w11, w12, w13],  ← Neuron 1 weights
           [w21, w22, w23]]  ← Neuron 2 weights
```

**Biases Vector**:
```
Shape: (output_size × 1)

biases = [[b1],  ← Neuron 1 bias
          [b2]]  ← Neuron 2 bias
```

**Why these shapes?**
```
Forward pass: Z = X·W^T + b
  X: (batch × input_size)
  W^T: (input_size × output_size)
  Z: (batch × output_size)
```

---

```cpp
    Matrix weight_gradients;  // Accumulated gradients for weights
    Matrix bias_gradients;    // Accumulated gradients for biases
```
**Gradient Storage**:

**Purpose**: Store ∂Loss/∂Weights and ∂Loss/∂Biases

**Shape**: Same as parameters
- `weight_gradients`: (output_size × input_size)
- `bias_gradients`: (output_size × 1)

**Lifecycle**:
```
1. Initialize to zero
2. backward() computes gradients
3. updateParameters() uses gradients
4. Reset to zero
```

---

```cpp
    Matrix cached_input;      // Cached for backward pass
```
**Input Caching**:

**Why cache input?**
```
Forward pass:  Z = X·W^T + b
                  ↑
                  Need X for backward pass!

Backward pass: ∂L/∂W = ∂L/∂Z × X
                              ↑
                        Use cached input
```

**Example**:
```cpp
// Forward
cached_input = input;  // Store: [[1, 2, 3]]
output = compute(input);

// Backward (later)
gradient = delta.transpose() * cached_input;  // Use cached value
```

---

```cpp
    std::unique_ptr<Activation> activation;
    Matrix cached_z;          // Pre-activation values
```
**Activation Function**:

**activation**: Smart pointer to activation function
- Can be ReLU, Sigmoid, Tanh, etc.
- `nullptr` means linear (no activation)

**cached_z**: Pre-activation values
```
Forward:  Z = X·W^T + b   ← Cache this!
          A = activation(Z)

Backward: Need Z for activation gradient
          ∂A/∂Z depends on Z
```

**Example**:
```cpp
DenseLayer layer(10, 5, new ReLU());
                         └─ Activation

Forward:
  Z = X·W^T + b           // cached_z = Z
  A = ReLU(Z)             // return A

Backward:
  gradient = ReLU'(cached_z) ⊙ output_gradient
```

---

### Constructor

```cpp
    DenseLayer(size_t input_size, size_t output_size, Activation* activation = nullptr);
```
**Create Layer**:

**Parameters**:
1. `input_size` - Number of input features
2. `output_size` - Number of neurons
3. `activation` - Activation function (optional)

**Example**:
```cpp
// Linear layer (no activation)
DenseLayer layer1(784, 128);

// ReLU activation
DenseLayer layer2(128, 64, new ReLU());

// Sigmoid output
DenseLayer layer3(64, 10, new Sigmoid());
```

---

### Weight Initialization

```cpp
    void initializeWeights(const std::string& strategy = "xavier");
```
**Initialize Weights**:

**Strategies**:
1. **"xavier"** (default) - Xavier/Glorot initialization
2. **"he"** - He initialization (for ReLU)
3. **"random"** - Random in [-0.5, 0.5]
4. **"zeros"** - All zeros (bad for training!)

**Why important?**
```
Bad initialization:
  • Too large → Exploding gradients
  • Too small → Vanishing gradients
  • All same → Neurons learn same thing

Good initialization:
  • Balanced variance
  • Breaks symmetry
  • Faster convergence
```

**Example**:
```cpp
DenseLayer layer(784, 128);
layer.initializeWeights("he");  // Good for ReLU
```

---

### Method Overrides

```cpp
    std::string getName() const override { return "Dense"; }
    size_t getInputSize() const override { return input_size; }
    size_t getOutputSize() const override { return output_size; }
```
**Simple Getters**: Return stored values

---

```cpp
    int getParameterCount() const override { 
        return (input_size * output_size) + output_size; 
    }
```
**Parameter Count Formula**:
```
Total = Weights + Biases
      = (input × output) + output
      = output × (input + 1)

Example: 784 input, 128 output
  = 784 × 128 + 128
  = 100,352 + 128
  = 100,480 parameters
```

---

### Gradient Access

```cpp
    Matrix getWeightGradients() const { return weight_gradients; }
    Matrix getBiasGradients() const { return bias_gradients; }
```
**Get Gradients**: For custom optimizers

**Usage**:
```cpp
// After backward pass
Matrix dW = layer.getWeightGradients();
Matrix db = layer.getBiasGradients();

// Use with custom optimizer
optimizer.update(layer.getWeights(), dW, "layer1_weights");
optimizer.update(layer.getBiases(), db, "layer1_biases");
```

---

```cpp
    void resetGradients();
```
**Reset to Zero**: Clear accumulated gradients

**When to use?**
```cpp
// After parameter update
layer.updateParameters(learning_rate);
layer.resetGradients();  // Prepare for next batch

// Or explicitly between batches
for (auto& batch : dataset) {
    layer.forward(batch);
    layer.backward(gradient);
    layer.resetGradients();  // Clear before next batch
}
```

---

## layer.cpp - Implementation Explanation

### Constructor Implementation

```cpp
DenseLayer::DenseLayer(size_t input_size, size_t output_size, Activation* activation)
    : input_size(input_size), output_size(output_size),
```
**Member Initializer List**:
- Initializes `input_size` and `output_size`
- Happens before constructor body

---

```cpp
      weights(output_size, input_size),
      biases(output_size, 1),
```
**Create Parameter Matrices**:
```
weights: (output_size × input_size)
  Example: (128 × 784) for 784→128 layer

biases: (output_size × 1)
  Example: (128 × 1)
```

---

```cpp
      weight_gradients(output_size, input_size),
      bias_gradients(output_size, 1),
```
**Create Gradient Matrices**:
- Same shape as parameters
- Will store ∂Loss/∂θ

---

```cpp
      activation(activation) {
```
**Store Activation**:
- `std::unique_ptr` takes ownership
- Automatically deletes when layer destroyed

---

```cpp
    initializeWeights("xavier");
    biases.zeros();
```
**Initialize Parameters**:
1. Weights: Xavier initialization
2. Biases: All zeros (common practice)

**Why zero biases?**
```
Initial forward pass should be roughly:
  Z = X·W + 0  (b=0)
  
If we initialize biases randomly:
  • Breaks symmetry already broken by weights
  • Adds unnecessary randomness
  • Can lead to unstable training
```

---

```cpp
    weight_gradients.zeros();
    bias_gradients.zeros();
}
```
**Initialize Gradients to Zero**:
- Ready for first backward pass

---

### Weight Initialization Implementation

```cpp
void DenseLayer::initializeWeights(const std::string& strategy) {
    if (strategy == "xavier" || strategy == "glorot") {
        weights.xavierInit(input_size, output_size);
```
**Xavier/Glorot Initialization**:
```
Formula: W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Where:
  n_in = input_size
  n_out = output_size

Purpose: Maintain variance through layers
  Var(input) ≈ Var(output)
```

**Example**:
```
Layer: 100 inputs → 50 outputs
limit = √(6/(100+50)) = √0.04 = 0.2
W ~ U(-0.2, 0.2)
```

---

```cpp
    } else if (strategy == "he") {
        weights.heInit(input_size);
```
**He Initialization**:
```
Formula: W ~ N(0, √(2/n_in))

Where:
  n_in = input_size

Purpose: Designed for ReLU activation
  Accounts for ReLU killing half the neurons
```

**When to use?**
```cpp
// ReLU → Use He
DenseLayer layer(100, 50, new ReLU());
layer.initializeWeights("he");

// Sigmoid/Tanh → Use Xavier
DenseLayer layer(100, 50, new Sigmoid());
layer.initializeWeights("xavier");
```

---

```cpp
    } else if (strategy == "random") {
        weights.randomize(-0.5, 0.5);
```
**Random Uniform**:
```
W ~ U(-0.5, 0.5)

Simple but not optimal:
  • Doesn't consider layer size
  • May cause vanishing/exploding gradients
```

---

```cpp
    } else if (strategy == "zeros") {
        weights.zeros();
```
**All Zeros**:
```
W = 0

⚠ BAD FOR TRAINING!
  • All neurons compute same output
  • All neurons receive same gradient
  • Neurons never differentiate (symmetry problem)

Only use for testing/debugging
```

---

```cpp
    } else {
        throw std::invalid_argument("Unknown weight initialization strategy: " + strategy);
    }
}
```
**Error Handling**: Invalid strategy throws exception

---

### Forward Pass Implementation

```cpp
Matrix DenseLayer::forward(const Matrix& input) {
    if (input.getCols() != input_size) {
        throw std::invalid_argument("Input size mismatch in DenseLayer::forward");
    }
```
**Dimension Check**:
```
Expected: (batch_size × input_size)
Example: (32 × 784) for batch of 32 MNIST images

If mismatch → throw error before computation
```

---

```cpp
    cached_input = input;
```
**Cache Input**:
```
Store for backward pass!

Forward:  Need input for computation
Backward: Need input for gradient computation
```

---

```cpp
    Matrix z = input * weights.transpose();
```
**Linear Transformation** - The core computation!

**Mathematics**:
```
Z = X · W^T

Dimensions:
  X: (batch × input_size)
  W: (output_size × input_size)
  W^T: (input_size × output_size)
  Z: (batch × output_size)

Example:
  X: (32 × 784)  - 32 images, 784 pixels each
  W^T: (784 × 128)
  Z: (32 × 128)  - 32 samples, 128 features each
```

**Why transpose W?**
```
Storage: W is (output_size × input_size)
  Each row = one neuron's weights

Computation: Need (input_size × output_size)
  Each column = one neuron's weights

Example:
W = [[w11, w12, w13],   W^T = [[w11, w21],
     [w21, w22, w23]]          [w12, w22],
                               [w13, w23]]
```

**What each element means**:
```
Z[i,j] = sum over k: X[i,k] * W[j,k]
       = dot product of sample i with neuron j's weights
```

---

```cpp
    for (size_t i = 0; i < z.getRows(); ++i) {
        for (size_t j = 0; j < z.getCols(); ++j) {
            z.set(i, j, z.get(i, j) + biases.get(j, 0));
        }
    }
```
**Add Bias** - Broadcasting across batch

**Mathematics**:
```
Z = Z + b

Shape:
  Z: (batch × output_size)
  b: (output_size × 1)
  
Broadcast b to each sample in batch
```

**Example**:
```
Z = [[1, 2],      b = [[0.5],
     [3, 4],           [1.0]]
     [5, 6]]

After adding bias:
Z = [[1.5, 3.0],
     [3.5, 5.0],
     [5.5, 7.0]]
```

**Implementation note**:
```cpp
// For each sample i
for i in batch:
    // For each neuron j
    for j in neurons:
        Z[i,j] = Z[i,j] + biases[j]
        
This adds bias[j] to all samples' j-th output
```

---

```cpp
    cached_z = z;
```
**Cache Pre-Activation**:
```
Store Z before activation!

Forward:  A = activation(Z)
Backward: Need Z for activation gradient
```

**Example with ReLU**:
```
Z = [-1, 2, -3, 4]  ← Cache this
A = ReLU(Z) = [0, 2, 0, 4]

Backward:
  ReLU'(Z) = [0, 1, 0, 1]  ← Need Z to compute!
```

---

```cpp
    if (activation) {
        return activation->forward(z);
    }
    
    return z;
}
```
**Apply Activation** (if present):
```
With activation:
  return activation(Z)

Without activation (linear):
  return Z
```

**Example**:
```cpp
// ReLU layer
DenseLayer layer(10, 5, new ReLU());
output = layer.forward(input);
// output = ReLU(X·W^T + b)

// Linear layer
DenseLayer layer(10, 5, nullptr);
output = layer.forward(input);
// output = X·W^T + b
```

---

### Backward Pass Implementation

```cpp
Matrix DenseLayer::backward(const Matrix& output_gradient) {
    Matrix delta = output_gradient;
    if (activation) {
        delta = activation->backward(cached_z, output_gradient);
    }
```
**Activation Gradient**:

**With activation**:
```
∂L/∂Z = ∂L/∂A ⊙ ∂A/∂Z
      = output_gradient ⊙ activation'(Z)

Where:
  ⊙ = element-wise multiplication (Hadamard product)
  Z = cached pre-activation values
```

**Example with ReLU**:
```
Z = [-1, 2, -3, 4]
A = ReLU(Z) = [0, 2, 0, 4]
∂L/∂A = [0.1, 0.2, 0.3, 0.4]  (from next layer)

ReLU'(Z) = [0, 1, 0, 1]  (gradient is 1 if Z>0, else 0)

delta = ∂L/∂A ⊙ ReLU'(Z)
      = [0.1, 0.2, 0.3, 0.4] ⊙ [0, 1, 0, 1]
      = [0, 0.2, 0, 0.4]
```

**Without activation** (linear):
```
delta = output_gradient  (no change)
```

---

```cpp
    weight_gradients = delta.transpose() * cached_input;
```
**Weight Gradients** - THE KEY COMPUTATION!

**Mathematics**:
```
∂L/∂W = delta^T · X

Dimensions:
  delta: (batch × output_size)
  delta^T: (output_size × batch)
  X: (batch × input_size)
  ∂L/∂W: (output_size × input_size)  ← Same as W!
```

**Why this formula?**
```
Forward:  Z = X · W^T + b
Backward: ∂L/∂W = ?

Chain rule:
  ∂L/∂W[j,k] = sum over i: ∂L/∂Z[i,j] · ∂Z[i,j]/∂W[j,k]
              = sum over i: delta[i,j] · X[i,k]
              = (delta^T · X)[j,k]
```

**Intuition**:
```
Weight gradient = 
  "How much does loss change with this weight?"

For weight W[j,k] (neuron j, input k):
  • Look at all samples in batch
  • For each sample:
      - How wrong was neuron j? (delta[i,j])
      - What was input k? (X[i,k])
  • Multiply and sum → gradient
```

**Example**:
```
delta = [[0.1, 0.2],    X = [[1, 2, 3],
         [0.3, 0.4]]         [4, 5, 6]]

delta^T = [[0.1, 0.3],
           [0.2, 0.4]]

∂L/∂W = delta^T · X
      = [[0.1, 0.3],  [[1, 2, 3],
         [0.2, 0.4]]   [4, 5, 6]]
         
      = [[0.1×1 + 0.3×4, 0.1×2 + 0.3×5, 0.1×3 + 0.3×6],
         [0.2×1 + 0.4×4, 0.2×2 + 0.4×5, 0.2×3 + 0.4×6]]
         
      = [[1.3, 1.7, 2.1],
         [1.8, 2.4, 3.0]]
```

---

```cpp
    bias_gradients = Matrix(output_size, 1);
    for (size_t j = 0; j < output_size; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < delta.getRows(); ++i) {
            sum += delta.get(i, j);
        }
        bias_gradients.set(j, 0, sum);
    }
```
**Bias Gradients** - Sum across batch

**Mathematics**:
```
∂L/∂b[j] = sum over batch: ∂L/∂Z[i,j]
         = sum over batch: delta[i,j]

For each neuron j:
  Sum the errors from all samples
```

**Why sum?**
```
Forward: Z[i,j] = (X·W^T)[i,j] + b[j]
                                  ↑
                    Same bias for all samples!

Backward: Each sample contributes to bias gradient
         Need to sum contributions
```

**Example**:
```
delta = [[0.1, 0.2],
         [0.3, 0.4],
         [0.5, 0.6]]

∂L/∂b = [[0.1 + 0.3 + 0.5],  = [[0.9],
         [0.2 + 0.4 + 0.6]]     [1.2]]
```

---

```cpp
    Matrix input_gradient = delta * weights;
    
    return input_gradient;
}
```
**Input Gradient** - Pass to previous layer

**Mathematics**:
```
∂L/∂X = delta · W

Dimensions:
  delta: (batch × output_size)
  W: (output_size × input_size)
  ∂L/∂X: (batch × input_size)  ← Same as X!
```

**Why this formula?**
```
Forward:  Z = X · W^T
Backward: ∂L/∂X = ?

Chain rule:
  ∂L/∂X[i,k] = sum over j: ∂L/∂Z[i,j] · ∂Z[i,j]/∂X[i,k]
              = sum over j: delta[i,j] · W[j,k]
              = (delta · W)[i,k]
```

**Intuition**:
```
"How much does loss change with this input?"

For input X[i,k] (sample i, feature k):
  • Look at all neurons j that used this input
  • For each neuron:
      - How wrong was it? (delta[i,j])
      - How much did it use this input? (W[j,k])
  • Multiply and sum → gradient
```

**Example**:
```
delta = [[0.1, 0.2]]  (1 sample, 2 neurons)
W = [[1, 2, 3],       (2 neurons, 3 inputs)
     [4, 5, 6]]

∂L/∂X = delta · W
      = [[0.1, 0.2]] · [[1, 2, 3],
                        [4, 5, 6]]
      = [[0.1×1 + 0.2×4, 0.1×2 + 0.2×5, 0.1×3 + 0.2×6]]
      = [[0.9, 1.2, 1.5]]
```

---

### Update Parameters Implementation

```cpp
void DenseLayer::updateParameters(double learning_rate) {
    weights = weights - weight_gradients * learning_rate;
    biases = biases - bias_gradients * learning_rate;
```
**Gradient Descent Update**:
```
Formula: θ_new = θ_old - α × ∇θ

Where:
  θ = parameters (weights or biases)
  α = learning_rate
  ∇θ = gradients
```

**Example**:
```
W_old = [[1.0, 2.0],
         [3.0, 4.0]]

∂L/∂W = [[0.1, 0.2],
         [0.3, 0.4]]

learning_rate = 0.1

W_new = W_old - 0.1 × ∂L/∂W
      = [[1.0, 2.0],     [[0.01, 0.02],
         [3.0, 4.0]]  -   [0.03, 0.04]]
      = [[0.99, 1.98],
         [2.97, 3.96]]
```

**Why subtract?**
```
Gradient points uphill (direction of increase)
We want to go downhill (minimize loss)
→ Move opposite to gradient
```

---

```cpp
    resetGradients();
}
```
**Reset After Update**: Clear gradients for next iteration

---

```cpp
void DenseLayer::resetGradients() {
    weight_gradients.zeros();
    bias_gradients.zeros();
}
```
**Clear Gradients**:
```
Set all gradient values to zero

Why?
  • Prevent accumulation from previous batches
  • Start fresh for next forward/backward pass
```

---

## Mathematical Foundation

### Forward Pass Mathematics

```
Complete forward pass for one layer:

1. Linear Transformation:
   Z = X·W^T + b
   
   X: (batch × input_size)     - Input data
   W: (output_size × input_size) - Weights
   b: (output_size × 1)         - Biases
   Z: (batch × output_size)     - Pre-activation

2. Activation:
   A = σ(Z)
   
   σ = activation function (ReLU, Sigmoid, etc.)
   A: (batch × output_size)     - Output

Example with numbers:
  X = [[1, 2],      W = [[0.5, 0.3],     b = [[0.1],
       [3, 4]]           [0.2, 0.4]]          [0.2]]
  
  Step 1: Z = X·W^T + b
    W^T = [[0.5, 0.2],
           [0.3, 0.4]]
    
    X·W^T = [[1×0.5+2×0.3, 1×0.2+2×0.4],
             [3×0.5+4×0.3, 3×0.2+4×0.4]]
          = [[1.1, 1.0],
             [2.7, 2.2]]
    
    Z = [[1.1+0.1, 1.0+0.2],
         [2.7+0.1, 2.2+0.2]]
      = [[1.2, 1.2],
         [2.8, 2.4]]
  
  Step 2: A = ReLU(Z)
    A = [[1.2, 1.2],
         [2.8, 2.4]]  (all positive, no change)
```

---

### Backward Pass Mathematics

```
Complete backward pass for one layer:

Given: ∂L/∂A (gradient from next layer)

1. Activation Gradient:
   ∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
   
   ⊙ = element-wise multiplication

2. Weight Gradient:
   ∂L/∂W = (∂L/∂Z)^T · X
   
3. Bias Gradient:
   ∂L/∂b = sum(∂L/∂Z, axis=0)
   
4. Input Gradient (for previous layer):
   ∂L/∂X = ∂L/∂Z · W

Example with numbers:
  Forward saved:
    Z = [[1.2, 1.2],
         [2.8, 2.4]]
    X = [[1, 2],
         [3, 4]]
    W = [[0.5, 0.3],
         [0.2, 0.4]]
  
  Backward receives:
    ∂L/∂A = [[0.1, 0.2],
             [0.3, 0.4]]
  
  Step 1: ∂L/∂Z = ∂L/∂A ⊙ ReLU'(Z)
    ReLU'(Z) = [[1, 1],  (all positive)
                [1, 1]]
    ∂L/∂Z = [[0.1, 0.2],
             [0.3, 0.4]]
  
  Step 2: ∂L/∂W = (∂L/∂Z)^T · X
    (∂L/∂Z)^T = [[0.1, 0.3],
                 [0.2, 0.4]]
    ∂L/∂W = [[0.1×1+0.3×3, 0.1×2+0.3×4],
             [0.2×1+0.4×3, 0.2×2+0.4×4]]
          = [[1.0, 1.4],
             [1.4, 2.0]]
  
  Step 3: ∂L/∂b = sum(∂L/∂Z, axis=0)
    ∂L/∂b = [[0.1+0.3],
             [0.2+0.4]]
          = [[0.4],
             [0.6]]
  
  Step 4: ∂L/∂X = ∂L/∂Z · W
    ∂L/∂X = [[0.1×0.5+0.2×0.2, 0.1×0.3+0.2×0.4],
             [0.3×0.5+0.4×0.2, 0.3×0.3+0.4×0.4]]
          = [[0.09, 0.11],
             [0.23, 0.25]]
```

---

## Weight Initialization Strategies

### Xavier (Glorot) Initialization

```
Formula: W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

Purpose: Maintain equal variance of activations and gradients

Derivation:
  Assume: Input variance = 1
  Want: Output variance = 1
  
  Z = X·W (ignore bias)
  Var(Z) = n_in × Var(W)
  
  For Var(Z) = 1:
    Var(W) = 1/n_in
    
  Xavier considers both forward (n_in) and backward (n_out):
    Var(W) = 2/(n_in + n_out)
    
  For uniform distribution:
    W ~ U(-a, a) has Var = a²/3
    a²/3 = 2/(n_in + n_out)
    a = √(6/(n_in + n_out))

Best for: Sigmoid, Tanh activations
```

### He Initialization

```
Formula: W ~ N(0, √(2/n_in))

Purpose: Account for ReLU killing half the neurons

Derivation:
  ReLU zeros out half the values
  → Effective fan-in is n_in/2
  → Need 2× larger variance
  
  Var(W) = 2/n_in

Best for: ReLU, Leaky ReLU activations
```

### Comparison

```
Layer: 100 inputs → 50 outputs

Xavier:
  limit = √(6/150) ≈ 0.200
  W ~ U(-0.200, 0.200)

He:
  std = √(2/100) ≈ 0.141
  W ~ N(0, 0.141)

Random:
  W ~ U(-0.5, 0.5)
  → Too large! Likely exploding gradients

Zeros:
  W = 0
  → Symmetry problem, neurons don't differentiate
```

---

## How Forward/Backward Pass Works

### Complete Example: Two-Layer Network

```
Architecture:
  Input (3) → Dense(3→4, ReLU) → Dense(4→2, Sigmoid) → Output

Step-by-step:
```

#### Forward Pass

```
Input: X = [[1, 2, 3]]  (1 sample, 3 features)

Layer 1: Dense(3→4, ReLU)
  W1 = [[0.5, 0.2, 0.1],
        [0.3, 0.4, 0.2],
        [0.1, 0.3, 0.5],
        [0.2, 0.1, 0.4]]
  b1 = [[0.1], [0.2], [0.1], [0.3]]
  
  Z1 = X·W1^T + b1
     = [[1, 2, 3]] · [[0.5, 0.3, 0.1, 0.2],
                      [0.2, 0.4, 0.3, 0.1],
                      [0.1, 0.2, 0.5, 0.4]] + [[0.1], [0.2], [0.1], [0.3]]
     = [[1.2, 1.9, 2.8, 2.2]]
  
  A1 = ReLU(Z1)
     = [[1.2, 1.9, 2.8, 2.2]]  (all positive)

Layer 2: Dense(4→2, Sigmoid)
  W2 = [[0.3, 0.4, 0.2, 0.1],
        [0.2, 0.3, 0.4, 0.5]]
  b2 = [[0.1], [0.2]]
  
  Z2 = A1·W2^T + b2
     = [[1.2, 1.9, 2.8, 2.2]] · [[0.3, 0.2],
                                  [0.4, 0.3],
                                  [0.2, 0.4],
                                  [0.1, 0.5]] + [[0.1], [0.2]]
     = [[2.04, 2.81]]
  
  A2 = Sigmoid(Z2)
     = [[0.885, 0.943]]  (output)
```

#### Backward Pass

```
Given: Target = [[1, 0]]
       Loss = MSE

∂L/∂A2 = 2(A2 - Target)
       = 2([[0.885, 0.943]] - [[1, 0]])
       = [[-0.230, 1.886]]

Layer 2 Backward:
  ∂L/∂Z2 = ∂L/∂A2 ⊙ Sigmoid'(Z2)
         = [[-0.230, 1.886]] ⊙ [[0.102, 0.054]]
         = [[-0.023, 0.102]]
  
  ∂L/∂W2 = (∂L/∂Z2)^T · A1
         = [[-0.023], [0.102]]^T · [[1.2, 1.9, 2.8, 2.2]]
         = [[-0.028, -0.044, -0.064, -0.051],
            [0.122, 0.194, 0.286, 0.224]]
  
  ∂L/∂b2 = [[-0.023], [0.102]]
  
  ∂L/∂A1 = ∂L/∂Z2 · W2
         = [[-0.023, 0.102]] · [[0.3, 0.4, 0.2, 0.1],
                                [0.2, 0.3, 0.4, 0.5]]
         = [[0.013, 0.021, 0.036, 0.049]]

Layer 1 Backward:
  ∂L/∂Z1 = ∂L/∂A1 ⊙ ReLU'(Z1)
         = [[0.013, 0.021, 0.036, 0.049]] ⊙ [[1, 1, 1, 1]]
         = [[0.013, 0.021, 0.036, 0.049]]
  
  ∂L/∂W1 = (∂L/∂Z1)^T · X
         = [[0.013], [0.021], [0.036], [0.049]] · [[1, 2, 3]]
         = [[0.013, 0.026, 0.039],
            [0.021, 0.042, 0.063],
            [0.036, 0.072, 0.108],
            [0.049, 0.098, 0.147]]
  
  ∂L/∂b1 = [[0.013], [0.021], [0.036], [0.049]]
```

#### Parameter Update

```
learning_rate = 0.1

Layer 2:
  W2_new = W2 - 0.1 × ∂L/∂W2
  b2_new = b2 - 0.1 × ∂L/∂b2

Layer 1:
  W1_new = W1 - 0.1 × ∂L/∂W1
  b1_new = b1 - 0.1 × ∂L/∂b1
```

---

## Practical Examples

See [layer_example.cpp](../example/layer_example.cpp) for:
1. Creating and initializing layers
2. Forward and backward pass
3. Complete training loop
4. Multi-layer networks
5. Visualization of learning

### Quick Example

```cpp
#include "nn/layer.h"
#include "nn/activation.h"

// Create a layer: 784 inputs → 128 outputs with ReLU
DenseLayer layer(784, 128, new ReLU());

// Initialize weights
layer.initializeWeights("he");  // He init for ReLU

// Forward pass
Matrix input(1, 784);  // 1 MNIST image
Matrix output = layer.forward(input);  // (1 × 128)

// Backward pass (training)
Matrix gradient(1, 128);  // Gradient from loss
Matrix input_grad = layer.backward(gradient);

// Update parameters
layer.updateParameters(0.01);  // learning_rate = 0.01
```

---

## Summary

### Key Concepts

1. **Dense Layer** = Fully connected layer
   - Every input connects to every output
   - Most parameters in typical networks

2. **Forward Pass** = Prediction
   - Z = X·W^T + b (linear)
   - A = activation(Z) (non-linear)

3. **Backward Pass** = Learning
   - Compute gradients for weights, biases, inputs
   - Use chain rule

4. **Weight Update** = Gradient descent
   - θ_new = θ_old - α × ∇θ

5. **Initialization** = Critical for training
   - Xavier for Sigmoid/Tanh
   - He for ReLU

### Implementation Highlights

- ✅ Polymorphic base class design
- ✅ Supports any activation function
- ✅ Caches values for backward pass
- ✅ Efficient matrix operations
- ✅ Dimension checking
- ✅ Multiple initialization strategies

### Common Patterns

```cpp
// Build network
DenseLayer layer1(784, 256, new ReLU());
DenseLayer layer2(256, 128, new ReLU());
DenseLayer layer3(128, 10, new Softmax());

// Training loop
for (epoch in range(100)) {
    // Forward
    auto h1 = layer1.forward(input);
    auto h2 = layer2.forward(h1);
    auto output = layer3.forward(h2);
    
    // Compute loss
    auto loss = computeLoss(output, target);
    
    // Backward
    auto grad3 = layer3.backward(loss_gradient);
    auto grad2 = layer2.backward(grad3);
    auto grad1 = layer1.backward(grad2);
    
    // Update
    layer3.updateParameters(lr);
    layer2.updateParameters(lr);
    layer1.updateParameters(lr);
}
```

---

**Next**: See practical examples in [layer_example.cpp](../example/layer_example.cpp)!
