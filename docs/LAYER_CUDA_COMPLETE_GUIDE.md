# GPU-Accelerated Neural Network Layers - Complete Guide

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Header File Explanation](#header-file-explanation)
3. [Implementation Explanation](#implementation-explanation)
4. [Mathematical Foundation](#mathematical-foundation)
5. [GPU Architecture](#gpu-architecture)
6. [Performance Optimization](#performance-optimization)
7. [Complete Examples](#complete-examples)

---

## Introduction

### What are CUDA Layers?

CUDA layers are GPU-accelerated implementations of neural network layers that leverage NVIDIA's CUDA platform for massive parallelization. Instead of computing operations sequentially on the CPU, CUDA layers distribute computations across thousands of GPU cores simultaneously.

### Why Use GPU Acceleration?

**Speed Advantage:**
- **CPU**: Sequential processing, 4-16 cores typically
- **GPU**: Parallel processing, thousands of CUDA cores
- **Speedup**: 10-100x for large matrices and deep networks

**Memory Bandwidth:**
- GPU memory bandwidth: ~500-900 GB/s
- CPU memory bandwidth: ~50-100 GB/s
- Critical for large matrix operations

**Real-World Impact:**
```
Training ImageNet (14M images) on ResNet-50:
- CPU: ~2 weeks
- Single GPU: ~1-2 days
- Multiple GPUs: Hours
```

### When to Use CUDA Layers?

✅ **Use GPU when:**
- Large models (>1M parameters)
- Large datasets (>10K samples)
- Large batch sizes (>32)
- Real-time inference needed
- Training deep networks

❌ **Use CPU when:**
- Small models (<10K parameters)
- Small datasets (<1K samples)
- Batch size = 1
- Prototyping/debugging
- No GPU available

---

## Header File Explanation

### File: `include/nn/layer_cuda.h`

Let's analyze every line of the header file:

```cpp
#ifndef LAYER_CUDA_H
#define LAYER_CUDA_H
```
**Purpose**: Header guard prevents multiple inclusions
- First inclusion: LAYER_CUDA_H is undefined, so content is included
- Subsequent inclusions: LAYER_CUDA_H is defined, content is skipped
- Prevents redefinition errors

```cpp
#include "matrix_cuda.h"
#include "activation_cuda.h"
#include <memory>
#include <string>
```
**Dependencies:**
- `matrix_cuda.h`: GPU matrix operations (cuBLAS, CUDA kernels)
- `activation_cuda.h`: GPU activation functions (ReLU, Sigmoid, etc.)
- `<memory>`: Smart pointers (std::unique_ptr) for activation
- `<string>`: Layer names and strategies

#### Base Class: LayerCUDA

```cpp
class LayerCUDA {
public:
    virtual ~LayerCUDA() = default;
```
**Abstract base class** for all CUDA layer types
- `virtual ~LayerCUDA() = default`: Virtual destructor for proper cleanup
- When derived class is deleted through base pointer, derived destructor is called
- Prevents memory leaks in polymorphic hierarchies

```cpp
    virtual MatrixCUDA forward(const MatrixCUDA& input) = 0;
```
**Forward propagation** (pure virtual = must be implemented by derived classes)

**What it does:**
- Takes input matrix (batch_size × input_features)
- Applies layer transformation
- Returns output matrix (batch_size × output_features)

**GPU Execution:**
- Input already on GPU (MatrixCUDA)
- Computation happens on GPU (cuBLAS operations)
- Output stays on GPU (no CPU transfer)

**Mathematical Operation:**
```
Z = X·W^T + b    (Linear transformation)
A = σ(Z)         (Activation function)
```

**Example:**
```cpp
// 32 images, 784 pixels each → 128 features
MatrixCUDA input(32, 784);  // On GPU
MatrixCUDA output = layer.forward(input);  // GPU computation
// output: (32 × 128) still on GPU
```

```cpp
    virtual MatrixCUDA backward(const MatrixCUDA& output_gradient) = 0;
```
**Backward propagation** (gradient computation)

**What it does:**
- Takes gradient from next layer (∂L/∂output)
- Computes three gradients:
  1. ∂L/∂W (weight gradients) - stored internally
  2. ∂L/∂b (bias gradients) - stored internally
  3. ∂L/∂X (input gradients) - returned to previous layer

**GPU Execution:**
- All matrix multiplications use cuBLAS
- Element-wise operations use CUDA kernels
- Thousands of gradients computed in parallel

**Chain Rule Application:**
```
Given: ∂L/∂A
Compute: ∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
Then:    ∂L/∂W = (∂L/∂Z)^T · X
         ∂L/∂b = sum(∂L/∂Z)
         ∂L/∂X = (∂L/∂Z) · W
```

```cpp
    virtual void updateParameters(double learning_rate) = 0;
```
**Parameter update** (gradient descent)

**What it does:**
- Applies computed gradients to parameters
- Uses simple SGD formula: θ = θ - α·∇θ

**GPU Execution:**
- Element-wise subtraction on GPU
- Thousands of parameters updated in parallel
- No CPU involvement

**Update Formula:**
```
W_new = W_old - learning_rate × ∂L/∂W
b_new = b_old - learning_rate × ∂L/∂b
```

**Example:**
```cpp
layer.forward(input);          // Compute output
layer.backward(output_grad);   // Compute gradients
layer.updateParameters(0.01);  // Apply updates (α = 0.01)
```

#### Derived Class: DenseLayerCUDA

```cpp
class DenseLayerCUDA : public LayerCUDA {
private:
    size_t input_size;   // Number of input features
    size_t output_size;  // Number of output neurons
```
**Layer dimensions:**
- `input_size`: How many features each input sample has
- `output_size`: How many neurons in this layer
- Example: (784, 128) means 784 pixels → 128 neurons

```cpp
    MatrixCUDA weights;           // (output × input)
    MatrixCUDA biases;            // (output × 1)
```
**Trainable parameters** (stored on GPU)

**Weight Matrix Shape:**
- Dimensions: (output_size × input_size)
- Why transposed? Efficient GPU matrix multiplication
- Example: (128 × 784) for layer with 128 neurons, 784 inputs

**Memory Usage:**
```
weights: output × input × 4 bytes (float32)
biases:  output × 4 bytes
Total:   (output × input + output) × 4 bytes
```

**Example:**
```
Layer (784 → 128):
weights = 128 × 784 = 100,352 floats = 401 KB
biases  = 128 floats = 512 bytes
Total = ~402 KB on GPU
```

```cpp
    MatrixCUDA weight_gradients;  // Same shape as weights
    MatrixCUDA bias_gradients;    // Same shape as biases
```
**Gradient storage** (accumulated during backward pass)

**Why separate from parameters?**
- Allows gradient accumulation across mini-batches
- Enables advanced optimizers (momentum, Adam)
- Clean separation of concerns

**Accumulation:**
```cpp
// Batch 1
layer.backward(grad1);  // gradients += new_gradients

// Batch 2
layer.backward(grad2);  // gradients += more_gradients

// Update with accumulated gradients
layer.updateParameters(lr);
layer.resetGradients();  // Clear for next iteration
```

```cpp
    MatrixCUDA cached_input;  // Input from forward pass
    MatrixCUDA cached_z;      // Pre-activation values
```
**Cached values for backward pass** (stored on GPU)

**Why cache input?**
- Needed to compute ∂L/∂W = (∂L/∂Z)^T · X
- Caching avoids recomputation
- Stays on GPU (no transfer overhead)

**Why cache Z?**
- Needed for activation derivative: σ'(Z)
- Example: ReLU'(z) = 1 if z > 0, else 0
- Used in chain rule: ∂L/∂Z = ∂L/∂A ⊙ σ'(Z)

**Memory Trade-off:**
```
For batch_size=32, layer(784→128):
cached_input: 32 × 784 × 4 = 100 KB
cached_z:     32 × 128 × 4 = 16 KB
Total:        ~116 KB

Trade: Extra 116 KB GPU memory for fast backward pass
```

```cpp
    std::unique_ptr<ActivationCUDA> activation;
```
**Activation function** (GPU version)

**Smart Pointer Benefits:**
- Automatic memory management (no manual delete)
- Ownership semantics (layer owns activation)
- nullptr if no activation (linear layer)

**Polymorphism:**
```cpp
activation = new ReLUCUDA();      // GPU ReLU
activation = new SigmoidCUDA();   // GPU Sigmoid
activation = new TanhCUDA();      // GPU Tanh
```

#### Constructor

```cpp
DenseLayerCUDA(size_t input_size, size_t output_size, 
               ActivationCUDA* act = nullptr);
```
**Layer creation**

**What it does:**
1. Allocates GPU memory for weights, biases, gradients
2. Initializes weights using Xavier initialization
3. Initializes biases to zeros
4. Stores activation function

**GPU Memory Allocation:**
```
MatrixCUDA weights(output_size, input_size);
- Allocates: output × input × sizeof(float) bytes on GPU
- Uses cudaMalloc internally
- Throws exception if GPU out of memory
```

**Example Usage:**
```cpp
// 784 inputs → 128 outputs with ReLU
DenseLayerCUDA layer(784, 128, new ReLUCUDA());

// GPU memory allocated:
// weights: 784 × 128 × 4 = 401 KB
// biases: 128 × 4 = 512 bytes
// gradients: 402 KB (same as parameters)
// Total: ~803 KB
```

#### Weight Initialization

```cpp
void initializeWeights(const std::string& strategy = "xavier");
```
**Initialize layer weights with various strategies**

**Available Strategies:**

**1. Xavier (Glorot) Initialization:**
```cpp
layer.initializeWeights("xavier");

Formula: W ~ N(0, σ²) where σ² = 2/(n_in + n_out)
```
**Derivation:**
- Goal: Var(output) = Var(input) for stable gradients
- Forward: Var(output) = n_in × Var(W) × Var(input)
- For Var(output) = Var(input): Var(W) = 1/n_in
- Considering backward pass: Var(W) = 2/(n_in + n_out)

**Best for:** Sigmoid, Tanh (symmetric activations)

**Example:**
```python
Layer (100 → 50):
σ² = 2 / (100 + 50) = 0.0133
σ = 0.115
```

**2. He Initialization:**
```cpp
layer.initializeWeights("he");

Formula: W ~ N(0, σ²) where σ² = 2/n_in
```
**Derivation:**
- ReLU kills half the neurons (outputs 0 for negative)
- Effective neurons = n_in/2
- To maintain variance: double Xavier initialization
- Result: Var(W) = 2/n_in

**Best for:** ReLU, LeakyReLU

**Example:**
```python
Layer (100 → 50):
σ² = 2 / 100 = 0.02
σ = 0.141  (larger than Xavier)
```

**3. Random Initialization:**
```cpp
layer.initializeWeights("random");

Formula: W ~ U(-1, 1)  (uniform distribution)
```
- Simple uniform random
- No theoretical guarantees
- Can work for small networks

**4. Zero Initialization:**
```cpp
layer.initializeWeights("zeros");

Formula: W = 0  (all zeros)
```
⚠️ **WARNING: Symmetry Problem!**
- All neurons compute same output
- All neurons get same gradient
- All neurons learn the same thing
- Network effectively has 1 neuron
- **Only use for debugging, never for training!**

**Implementation Detail:**
```cpp
// Initialization happens on CPU, then transferred to GPU
Matrix temp_weights(output, input);
// ... fill with random values ...
weights = MatrixCUDA(temp_weights);  // CPU → GPU transfer
```

**Why on CPU?**
- Random number generation simpler on CPU
- Initialization happens once (not performance critical)
- Could be optimized with cuRAND for very large models

---

## Implementation Explanation

### File: `src/layer_cuda.cu`

#### Constructor Implementation

```cpp
DenseLayerCUDA::DenseLayerCUDA(size_t input_size, size_t output_size, 
                               ActivationCUDA* act)
    : input_size(input_size),
      output_size(output_size),
      weights(output_size, input_size),       // GPU allocation
      biases(output_size, 1),                 // GPU allocation
      weight_gradients(output_size, input_size),
      bias_gradients(output_size, 1),
      activation(act)
{
    initializeWeights("xavier");
    biases.zeros();
    weight_gradients.zeros();
    bias_gradients.zeros();
}
```

**Step-by-Step Execution:**

**Line 1-2: Member initialization list**
```cpp
input_size(input_size),
output_size(output_size),
```
- Stores layer dimensions
- Used for validation and getters

**Line 3-4: GPU memory allocation for parameters**
```cpp
weights(output_size, input_size),       // GPU allocation
biases(output_size, 1),                 // GPU allocation
```
**What happens:**
1. MatrixCUDA constructor called
2. cudaMalloc() allocates GPU memory
3. Memory initialized on GPU

**GPU Operation:**
```cuda
// Inside MatrixCUDA constructor
cudaMalloc(&d_data, rows * cols * sizeof(float));
// d_data now points to GPU memory
```

**Line 5-6: Gradient storage allocation**
```cpp
weight_gradients(output_size, input_size),
bias_gradients(output_size, 1),
```
- Same shapes as parameters
- Initialized to zeros
- Accumulate gradients during backward pass

**Line 7: Store activation function**
```cpp
activation(act)
```
- Smart pointer takes ownership
- Will call activation->forward() and activation->backward()
- Can be nullptr for linear layer

**Line 9-12: Initialization**
```cpp
initializeWeights("xavier");
biases.zeros();
weight_gradients.zeros();
bias_gradients.zeros();
```
- Weights: Xavier random initialization
- Biases: All zeros (standard practice)
- Gradients: All zeros (ready to accumulate)

#### Forward Pass Implementation

```cpp
MatrixCUDA DenseLayerCUDA::forward(const MatrixCUDA& input) {
    cached_input = input;  // Cache for backward pass
    
    MatrixCUDA weights_T = weights.transpose();  // GPU operation
    MatrixCUDA z = input.multiplyGPU(weights_T);  // cuBLAS
    
    // Add biases (broadcasted)
    for (size_t i = 0; i < z.getRows(); i++) {
        for (size_t j = 0; j < z.getCols(); j++) {
            double val = z.get(i, j) + biases.get(j, 0);
            z.set(i, j, val);
        }
    }
    
    cached_z = z;  // Cache for backward pass
    
    if (activation) {
        return activation->forward(z);  // GPU activation
    }
    
    return z;  // Linear layer
}
```

**Step-by-Step with GPU Operations:**

**Step 1: Cache input**
```cpp
cached_input = input;
```
- Shallow copy (both point to same GPU memory)
- Needed for backward pass: ∂L/∂W = (∂L/∂Z)^T · X
- No CPU-GPU transfer

**Step 2: Transpose weights**
```cpp
MatrixCUDA weights_T = weights.transpose();
```
**GPU Operation:**
- Not actual data movement, just layout change
- cuBLAS can work with transposed layout
- Very fast (just metadata update)

**Step 3: Matrix multiplication**
```cpp
MatrixCUDA z = input.multiplyGPU(weights_T);
```
**cuBLAS Operation:**
```cuda
// Simplified cuBLAS call
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    input_d, lda,
    weights_T_d, ldb,
    &beta,
    z_d, ldc);
```

**Parallelization:**
- Each output element computed by different GPU thread
- For (32 × 784) * (784 × 128):
  - Output: 32 × 128 = 4,096 elements
  - 4,096 threads launched
  - Each thread computes dot product of 784 elements
  - All threads execute simultaneously

**Performance:**
```
CPU: Sequential, ~10 GFLOPs
GPU: Parallel, ~10,000 GFLOPs (1000x)
```

**Step 4: Add biases**
```cpp
for (size_t i = 0; i < z.getRows(); i++) {
    for (size_t j = 0; j < z.getCols(); j++) {
        double val = z.get(i, j) + biases.get(j, 0);
        z.set(i, j, val);
    }
}
```
**Broadcasting:**
- Same bias added to all samples in batch
- bias[j] added to all elements in column j
- GPU: All additions happen in parallel

**CUDA Kernel (conceptual):**
```cuda
__global__ void addBias(float* z, float* biases, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        z[i * cols + j] += biases[j];
    }
}
```

**Step 5: Apply activation**
```cpp
if (activation) {
    return activation->forward(z);
}
```
**GPU Activation:**
- Element-wise operation
- Each thread processes one element
- Examples:
  - ReLU: `z[i] = max(0, z[i])`
  - Sigmoid: `z[i] = 1 / (1 + exp(-z[i]))`
  - Tanh: `z[i] = tanh(z[i])`

#### Backward Pass Implementation

```cpp
MatrixCUDA DenseLayerCUDA::backward(const MatrixCUDA& output_gradient) {
    MatrixCUDA delta = output_gradient;
    
    if (activation) {
        delta = activation->backward(cached_z, delta);
    }
    
    MatrixCUDA delta_T = delta.transpose();
    weight_gradients = delta_T.multiplyGPU(cached_input);
    
    bias_gradients.zeros();
    for (size_t i = 0; i < delta.getRows(); i++) {
        for (size_t j = 0; j < delta.getCols(); j++) {
            double current = bias_gradients.get(j, 0);
            bias_gradients.set(j, 0, current + delta.get(i, j));
        }
    }
    
    MatrixCUDA input_gradient = delta.multiplyGPU(weights);
    return input_gradient;
}
```

**Step-by-Step with Chain Rule:**

**Step 1: Activation gradient**
```cpp
if (activation) {
    delta = activation->backward(cached_z, delta);
}
```
**Chain Rule:**
```
∂L/∂Z = ∂L/∂A ⊙ ∂A/∂Z
      = output_gradient ⊙ activation'(cached_z)
```

**GPU Execution:**
- Element-wise multiplication
- Each thread processes one element
- Example (ReLU):
```cuda
__global__ void reluBackward(float* grad, float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad[i] *= (z[i] > 0) ? 1.0f : 0.0f;
    }
}
```

**Step 2: Weight gradients**
```cpp
MatrixCUDA delta_T = delta.transpose();
weight_gradients = delta_T.multiplyGPU(cached_input);
```
**Formula:**
```
∂L/∂W = (∂L/∂Z)^T · X
```

**Dimensions:**
```
delta_T:      (output × batch)
cached_input: (batch × input)
Result:       (output × input)  [same as weights]
```

**cuBLAS Operation:**
- Matrix multiplication on GPU
- Thousands of gradient elements computed in parallel
- Example: For layer (784→128), batch=32:
  - 128 × 784 = 100,352 gradient values
  - All computed simultaneously on GPU

**Step 3: Bias gradients**
```cpp
bias_gradients.zeros();
for (size_t i = 0; i < delta.getRows(); i++) {
    for (size_t j = 0; j < delta.getCols(); j++) {
        double current = bias_gradients.get(j, 0);
        bias_gradients.set(j, 0, current + delta.get(i, j));
    }
}
```
**Formula:**
```
∂L/∂b_j = Σ_i ∂L/∂Z_ij  (sum over batch)
```

**Why sum over batch?**
- Same bias used for all samples
- Gradient contributions from all samples must be summed
- Total gradient = average effect on all samples

**GPU Optimization (conceptual):**
```cuda
// Parallel reduction
__global__ void sumColumns(float* delta, float* bias_grad, 
                          int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            sum += delta[i * cols + j];
        }
        bias_grad[j] = sum;
    }
}
```

**Step 4: Input gradients**
```cpp
MatrixCUDA input_gradient = delta.multiplyGPU(weights);
return input_gradient;
```
**Formula:**
```
∂L/∂X = ∂L/∂Z · W
```

**Dimensions:**
```
delta:   (batch × output)
weights: (output × input)
Result:  (batch × input)  [same as input]
```

**Purpose:**
- Passes gradient to previous layer
- Enables multi-layer backpropagation
- Chain rule continues: previous layer uses this as output_gradient

---

## Mathematical Foundation

### Forward Pass Mathematics

**Complete Formulation:**

**Input:**
- X: (batch_size × input_size)
- W: (output_size × input_size)  
- b: (output_size × 1)

**Step 1: Linear Transformation**
```
Z = X·W^T + b
```

**Matrix Multiplication Details:**
```
For element Z[i,j]:
Z[i,j] = Σ_k X[i,k] × W[j,k] + b[j]
       = (dot product of row i of X and row j of W) + bias j
```

**Step 2: Activation**
```
A = σ(Z)
```

**Element-wise Application:**
```
A[i,j] = σ(Z[i,j])
```

**Common Activations:**
```
ReLU:    σ(z) = max(0, z)
Sigmoid: σ(z) = 1 / (1 + e^(-z))
Tanh:    σ(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

### Backward Pass Mathematics

**Given:**
- ∂L/∂A: Gradient from next layer (batch_size × output_size)

**Step 1: Compute ∂L/∂Z**
```
∂L/∂Z = ∂L/∂A ⊙ σ'(Z)
```
where ⊙ denotes element-wise multiplication (Hadamard product)

**Activation Derivatives:**
```
ReLU:    σ'(z) = 1 if z > 0, else 0
Sigmoid: σ'(z) = σ(z) × (1 - σ(z))
Tanh:    σ'(z) = 1 - tanh²(z)
```

**Step 2: Compute ∂L/∂W**

**Derivation:**
```
Z[i,j] = Σ_k X[i,k] × W[j,k] + b[j]

∂Z[i,j]/∂W[j,k] = X[i,k]

∂L/∂W[j,k] = Σ_i (∂L/∂Z[i,j] × ∂Z[i,j]/∂W[j,k])
           = Σ_i (∂L/∂Z[i,j] × X[i,k])
```

**Matrix Form:**
```
∂L/∂W = (∂L/∂Z)^T · X
```

**Dimensions:**
```
(∂L/∂Z)^T: (output × batch)
X:         (batch × input)
∂L/∂W:     (output × input)  ✓ matches W shape
```

**Step 3: Compute ∂L/∂b**

**Derivation:**
```
Z[i,j] = ... + b[j]  (same bias for all samples)

∂Z[i,j]/∂b[j] = 1

∂L/∂b[j] = Σ_i ∂L/∂Z[i,j]
```

**Matrix Form:**
```
∂L/∂b = sum(∂L/∂Z, axis=0)  (sum over batch dimension)
```

**Step 4: Compute ∂L/∂X**

**Derivation:**
```
Z[i,j] = Σ_k X[i,k] × W[j,k]

∂Z[i,j]/∂X[i,m] = W[j,m]

∂L/∂X[i,m] = Σ_j (∂L/∂Z[i,j] × ∂Z[i,j]/∂X[i,m])
           = Σ_j (∂L/∂Z[i,j] × W[j,m])
```

**Matrix Form:**
```
∂L/∂X = ∂L/∂Z · W
```

**Dimensions:**
```
∂L/∂Z: (batch × output)
W:     (output × input)
∂L/∂X: (batch × input)  ✓ matches X shape
```

### Numerical Example

**Setup:**
```
Layer: 3 inputs → 2 outputs (ReLU)
Batch size: 1
```

**Parameters:**
```
W = [[0.5, 0.3, 0.2],
     [0.4, 0.6, 0.1]]

b = [[0.1],
     [0.2]]
```

**Input:**
```
X = [[1.0, 2.0, 3.0]]
```

**Forward Pass:**

**Step 1: Compute Z**
```
Z = X·W^T + b

Z[0,0] = 1.0×0.5 + 2.0×0.3 + 3.0×0.2 + 0.1
       = 0.5 + 0.6 + 0.6 + 0.1
       = 1.8

Z[0,1] = 1.0×0.4 + 2.0×0.6 + 3.0×0.1 + 0.2
       = 0.4 + 1.2 + 0.3 + 0.2
       = 2.1

Z = [[1.8, 2.1]]
```

**Step 2: Compute A (ReLU)**
```
A[0,0] = max(0, 1.8) = 1.8
A[0,1] = max(0, 2.1) = 2.1

A = [[1.8, 2.1]]
```

**Backward Pass:**

**Given gradient from next layer:**
```
∂L/∂A = [[0.1, 0.2]]
```

**Step 1: Compute ∂L/∂Z**
```
ReLU'(1.8) = 1 (since 1.8 > 0)
ReLU'(2.1) = 1 (since 2.1 > 0)

∂L/∂Z = ∂L/∂A ⊙ ReLU'(Z)
      = [[0.1, 0.2]] ⊙ [[1, 1]]
      = [[0.1, 0.2]]
```

**Step 2: Compute ∂L/∂W**
```
∂L/∂W = (∂L/∂Z)^T · X

(∂L/∂Z)^T = [[0.1],
             [0.2]]

X = [[1.0, 2.0, 3.0]]

∂L/∂W = [[0.1],  · [[1.0, 2.0, 3.0]]
         [0.2]]

     = [[0.1×1.0, 0.1×2.0, 0.1×3.0],
        [0.2×1.0, 0.2×2.0, 0.2×3.0]]
     
     = [[0.1, 0.2, 0.3],
        [0.2, 0.4, 0.6]]
```

**Step 3: Compute ∂L/∂b**
```
∂L/∂b = sum(∂L/∂Z, axis=0)
      = [[0.1],
         [0.2]]
(only one sample in batch, so no summing needed)
```

**Step 4: Compute ∂L/∂X**
```
∂L/∂X = ∂L/∂Z · W

∂L/∂Z = [[0.1, 0.2]]
W = [[0.5, 0.3, 0.2],
     [0.4, 0.6, 0.1]]

∂L/∂X[0,0] = 0.1×0.5 + 0.2×0.4 = 0.05 + 0.08 = 0.13
∂L/∂X[0,1] = 0.1×0.3 + 0.2×0.6 = 0.03 + 0.12 = 0.15
∂L/∂X[0,2] = 0.1×0.2 + 0.2×0.1 = 0.02 + 0.02 = 0.04

∂L/∂X = [[0.13, 0.15, 0.04]]
```

**Parameter Update:**
```
learning_rate = 0.1

W_new = W - learning_rate × ∂L/∂W
      = [[0.5, 0.3, 0.2],  - 0.1 × [[0.1, 0.2, 0.3],
         [0.4, 0.6, 0.1]]            [0.2, 0.4, 0.6]]
      
      = [[0.49, 0.28, 0.17],
         [0.38, 0.56, 0.04]]

b_new = b - learning_rate × ∂L/∂b
      = [[0.1],  - 0.1 × [[0.1],
         [0.2]]            [0.2]]
      
      = [[0.09],
         [0.18]]
```

---

## GPU Architecture

### CUDA Execution Model

**Thread Hierarchy:**
```
Grid (entire computation)
  └─ Blocks (groups of threads)
       └─ Threads (individual computations)
```

**Example: Matrix Addition (1000×1000)**
```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads per block
dim3 numBlocks(
    (1000 + 15) / 16,  // 63 blocks in x
    (1000 + 15) / 16   // 63 blocks in y
);

addMatrices<<<numBlocks, threadsPerBlock>>>(A, B, C, 1000, 1000);
```

**Result:**
- Total blocks: 63 × 63 = 3,969
- Total threads: 3,969 × 256 = 1,016,064
- All threads execute in parallel

### Memory Hierarchy

**1. Global Memory (GPU DRAM)**
```
Capacity: 4-24 GB (typical)
Bandwidth: 500-900 GB/s
Latency: ~500 cycles
Use: Main storage for matrices
```

**2. Shared Memory (On-chip)**
```
Capacity: 48-96 KB per block
Bandwidth: ~15 TB/s
Latency: ~5 cycles
Use: Temporary data shared within block
```

**3. Registers**
```
Capacity: 64 KB per SM
Bandwidth: Highest
Latency: 1 cycle
Use: Thread-local variables
```

**Optimization Strategy:**
```
1. Minimize global memory accesses
2. Maximize shared memory usage
3. Coalesce global memory accesses
4. Avoid bank conflicts in shared memory
```

### cuBLAS Library

**What is cuBLAS?**
- CUDA Basic Linear Algebra Subprograms
- Highly optimized GPU implementations
- NVIDIA engineers spent years optimizing
- Used by all major deep learning frameworks

**Key Operations:**

**1. Matrix Multiplication (GEMM)**
```cuda
cublasSgemm(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    A, lda,      // Matrix A (m × k)
    B, ldb,      // Matrix B (k × n)
    &beta,
    C, ldc       // Matrix C (m × n) = α(A×B) + βC
);
```

**Performance:**
- Achieves >90% of GPU peak performance
- Uses tensor cores on modern GPUs
- Automatic tiling and optimization

**2. Why So Fast?**

**Naive Implementation:**
```cuda
// Simple but slow
for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*n + j];
        }
        C[i*n + j] = sum;
    }
}
// Many global memory accesses, poor cache usage
```

**cuBLAS Optimizations:**
1. **Tiling**: Break matrices into tiles that fit in shared memory
2. **Coalescing**: Access memory in patterns that maximize bandwidth
3. **Register Blocking**: Keep intermediate results in registers
4. **Tensor Cores**: Use specialized hardware (on Turing+)

**Result:**
- 10-100x faster than naive implementation
- Approaches theoretical peak performance

---

## Performance Optimization

### Batch Size Effects

**Small Batch (batch_size = 1):**
```
GPU Utilization: ~5-10%
CPU Faster: Yes (due to overhead)
Use case: Real-time inference
```

**Medium Batch (batch_size = 32):**
```
GPU Utilization: ~40-60%
GPU Faster: Yes (2-5x)
Use case: Standard training
```

**Large Batch (batch_size = 256):**
```
GPU Utilization: ~80-95%
GPU Faster: Yes (10-50x)
Use case: Large-scale training
```

**Trade-offs:**
```
Larger batch:
  ✓ Better GPU utilization
  ✓ Faster training per sample
  ✗ More GPU memory needed
  ✗ May affect convergence
```

### Memory Management

**GPU Memory Breakdown:**
```
Model parameters:    Fixed (e.g., 100 MB)
Gradients:          Fixed (same as parameters)
Activations:        Batch-dependent
Optimizer state:    Fixed or 2-3x parameters
```

**Example:**
```
ResNet-50:
  Parameters: 25M × 4 bytes = 100 MB
  Gradients: 100 MB
  Activations (batch=32): ~400 MB
  Adam optimizer: 200 MB (momentum + variance)
  Total: ~800 MB
```

**Out of Memory? Solutions:**
1. **Reduce batch size**: Most effective
2. **Gradient checkpointing**: Trade compute for memory
3. **Mixed precision**: Use float16 instead of float32
4. **Model parallelism**: Split model across GPUs

### Transfer Overhead

**CPU ↔ GPU Transfer Costs:**
```
Matrix (1000×1000):
  Size: 4 MB
  Transfer time: ~0.5 ms
  
If computation takes <0.5ms:
  Transfer overhead dominates
  CPU might be faster!
```

**Minimizing Transfers:**
```cpp
// Bad: Multiple small transfers
for (int i = 0; i < 100; i++) {
    MatrixCUDA x(data[i]);  // CPU → GPU
    MatrixCUDA y = layer.forward(x);
    result[i] = y.toCPU();  // GPU → CPU
}

// Good: Batch processing
MatrixCUDA X(100, features);  // One CPU → GPU
MatrixCUDA Y = layer.forward(X);
Y.toCPU();  // One GPU → CPU
```

### When to Use CPU vs GPU

**Use CPU when:**
- Batch size < 8
- Model has < 100K parameters
- Rapid prototyping/debugging
- Production inference (single sample)

**Use GPU when:**
- Batch size ≥ 32
- Model has > 1M parameters
- Training phase
- High-throughput inference

---

## Complete Examples

### Example 1: Simple Forward/Backward

```cpp
#include "layer_cuda.h"

int main() {
    // Create layer: 784 → 128 with ReLU
    DenseLayerCUDA layer(784, 128, new ReLUCUDA());
    
    // Initialize weights
    layer.initializeWeights("he");
    
    // Create input (32 samples, 784 features)
    Matrix X_cpu(32, 784);
    // ... fill X_cpu with data ...
    MatrixCUDA X(X_cpu);  // Transfer to GPU
    
    // Forward pass (on GPU)
    MatrixCUDA output = layer.forward(X);
    
    // Create gradient (from next layer)
    Matrix grad_cpu(32, 128);
    // ... fill grad_cpu ...
    MatrixCUDA grad(grad_cpu);  // Transfer to GPU
    
    // Backward pass (on GPU)
    MatrixCUDA input_grad = layer.backward(grad);
    
    // Update parameters (on GPU)
    layer.updateParameters(0.01);
    
    // Clear gradients for next batch
    layer.resetGradients();
    
    return 0;
}
```

### Example 2: Multi-Layer Network

```cpp
int main() {
    // Build network: 784 → 256 → 128 → 10
    DenseLayerCUDA hidden1(784, 256, new ReLUCUDA());
    DenseLayerCUDA hidden2(256, 128, new ReLUCUDA());
    DenseLayerCUDA output(128, 10, new SigmoidCUDA());
    
    // Initialize
    hidden1.initializeWeights("he");
    hidden2.initializeWeights("he");
    output.initializeWeights("xavier");
    
    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        for (auto& [X_cpu, Y_cpu] : dataset) {
            // Transfer to GPU
            MatrixCUDA X(X_cpu);
            MatrixCUDA Y(Y_cpu);
            
            // Forward pass (all on GPU)
            MatrixCUDA h1 = hidden1.forward(X);
            MatrixCUDA h2 = hidden2.forward(h1);
            MatrixCUDA pred = output.forward(h2);
            
            // Compute loss gradient
            MSELossCUDA loss_fn;
            MatrixCUDA loss_grad = loss_fn.gradient(pred, Y);
            
            // Backward pass (all on GPU)
            MatrixCUDA grad3 = output.backward(loss_grad);
            MatrixCUDA grad2 = hidden2.backward(grad3);
            MatrixCUDA grad1 = hidden1.backward(grad2);
            
            // Update all layers (on GPU)
            output.updateParameters(0.01);
            hidden2.updateParameters(0.01);
            hidden1.updateParameters(0.01);
        }
    }
    
    return 0;
}
```

### Example 3: Performance Comparison

```cpp
void benchmark(int input_size, int output_size, int batch_size) {
    // Create CPU and GPU layers
    DenseLayer cpu_layer(input_size, output_size, new ReLU());
    DenseLayerCUDA gpu_layer(input_size, output_size, new ReLUCUDA());
    
    // Create data
    Matrix X_cpu(batch_size, input_size);
    // ... fill with random data ...
    MatrixCUDA X_gpu(X_cpu);
    
    // Benchmark CPU
    auto start = std::chrono::high_resolution_clock::now();
    Matrix cpu_out = cpu_layer.forward(X_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start);
    
    // Benchmark GPU
    start = std::chrono::high_resolution_clock::now();
    MatrixCUDA gpu_out = gpu_layer.forward(X_gpu);
    end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start);
    
    // Results
    std::cout << "CPU: " << cpu_time.count() << " μs\n";
    std::cout << "GPU: " << gpu_time.count() << " μs\n";
    std::cout << "Speedup: " << (double)cpu_time.count() / gpu_time.count() 
              << "x\n";
}
```

---

## Summary

**Key Takeaways:**

1. **GPU Acceleration**: 10-100x speedup for large models/batches
2. **cuBLAS**: Highly optimized matrix operations
3. **Parallel Execution**: Thousands of threads simultaneously
4. **Memory Hierarchy**: Global → Shared → Registers
5. **Batch Processing**: Critical for GPU utilization
6. **Transfer Overhead**: Minimize CPU↔GPU transfers

**When to Use CUDA Layers:**
- Training large models
- Large batch sizes
- Production inference (batch mode)
- Real-time constraints

**Best Practices:**
- Keep data on GPU during training
- Use appropriate batch sizes
- Choose initialization based on activation
- Monitor GPU memory usage
- Profile to identify bottlenecks

For more examples, see [layer_cuda_example.cpp](../example/layer_cuda_example.cpp)!
