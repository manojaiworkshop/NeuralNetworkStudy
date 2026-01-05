/**
 * @file layer_cuda.cu
 * @brief CUDA implementation of neural network layers
 * 
 * This file implements GPU-accelerated layer operations using MatrixCUDA.
 * All computations leverage CUDA kernels for maximum performance.
 */

#include "../include/nn/layer_cuda.h"
#include "../include/nn/matrix.h"  // For CPU matrix operations during init
#include <random>
#include <cmath>
#include <stdexcept>

/**
 * DenseLayerCUDA Constructor
 * 
 * Initializes a fully connected layer with GPU memory allocation.
 * 
 * Steps:
 * 1. Store layer dimensions
 * 2. Allocate GPU memory for weights (output × input)
 * 3. Allocate GPU memory for biases (output × 1)
 * 4. Allocate GPU memory for gradients
 * 5. Initialize weights using Xavier method
 * 6. Initialize biases to zeros
 * 7. Store activation function
 * 
 * Memory Allocation:
 * - weights: output_size × input_size elements (4 bytes each)
 * - biases: output_size elements
 * - gradients: Same as parameters
 * Total GPU memory = 2 × (output × input + output) × 4 bytes
 * 
 * Example:
 *   Layer (784 → 128):
 *   Memory = 2 × (784×128 + 128) × 4 = ~800 KB
 */
DenseLayerCUDA::DenseLayerCUDA(size_t input_size, size_t output_size, 
                               ActivationCUDA* act)
    : input_size(input_size),
      output_size(output_size),
      weights(output_size, input_size),       // GPU allocation
      biases(output_size, 1),                 // GPU allocation
      weight_gradients(output_size, input_size),  // GPU allocation
      bias_gradients(output_size, 1),        // GPU allocation
      activation(act)
{
    // Initialize weights using Xavier initialization
    // This happens on CPU, then transfers to GPU
    initializeWeights("xavier");
    
    // Initialize biases to zeros on GPU
    biases.zeros();
    
    // Initialize gradients to zeros on GPU
    weight_gradients.zeros();
    bias_gradients.zeros();
}

/**
 * Initialize Weights
 * 
 * Implements various initialization strategies. Process:
 * 1. Create temporary CPU matrix
 * 2. Fill with random values according to strategy
 * 3. Transfer to GPU memory
 * 
 * Why initialize on CPU?
 * - Random number generation is simpler on CPU
 * - Initialization happens once (not performance critical)
 * - Could be optimized with cuRAND for very large models
 * 
 * Strategy Details:
 * 
 * 1. Xavier (Glorot) Initialization:
 *    W ~ N(0, σ²) where σ² = 2/(n_in + n_out)
 *    
 *    Derivation:
 *    - Assume: Var(input) = Var(output) for stable gradients
 *    - Forward: Var(output) = n_in × Var(W) × Var(input)
 *    - For Var(output) = Var(input): Var(W) = 1/n_in
 *    - Considering backward pass: Var(W) = 2/(n_in + n_out)
 *    
 *    Best for: Sigmoid, Tanh (symmetric activations)
 * 
 * 2. He Initialization:
 *    W ~ N(0, σ²) where σ² = 2/n_in
 *    
 *    Derivation:
 *    - ReLU kills half the neurons (outputs 0 for negative)
 *    - Effective neurons = n_in/2
 *    - Compensate: Var(W) = 2/n_in (double Xavier)
 *    
 *    Best for: ReLU, LeakyReLU
 * 
 * 3. Random: W ~ U(-1, 1)
 *    Simple uniform random, no specific guarantees
 * 
 * 4. Zeros: All zeros
 *    ⚠️ Symmetry problem: All neurons learn the same thing!
 */
void DenseLayerCUDA::initializeWeights(const std::string& strategy) {
    // Create temporary CPU matrix for initialization
    Matrix temp_weights(output_size, input_size);
    Matrix temp_biases(output_size, 1);
    
    if (strategy == "xavier") {
        // Xavier initialization: Var(W) = 2 / (n_in + n_out)
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Calculate standard deviation
        double stddev = std::sqrt(2.0 / (input_size + output_size));
        std::normal_distribution<double> dist(0.0, stddev);
        
        // Fill weights with random values from N(0, stddev²)
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < input_size; j++) {
                temp_weights.set(i, j, dist(gen));
            }
        }
        
    } else if (strategy == "he") {
        // He initialization: Var(W) = 2 / n_in
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Calculate standard deviation (larger than Xavier for ReLU)
        double stddev = std::sqrt(2.0 / input_size);
        std::normal_distribution<double> dist(0.0, stddev);
        
        // Fill weights
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < input_size; j++) {
                temp_weights.set(i, j, dist(gen));
            }
        }
        
    } else if (strategy == "random") {
        // Random uniform initialization: U(-1, 1)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        // Fill weights
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < input_size; j++) {
                temp_weights.set(i, j, dist(gen));
            }
        }
        
    } else if (strategy == "zeros") {
        // Zero initialization (bad for training!)
        temp_weights.zeros();
        
    } else {
        throw std::invalid_argument("Unknown initialization strategy: " + strategy);
    }
    
    // Initialize biases to zeros (standard practice)
    temp_biases.zeros();
    
    // Transfer from CPU to GPU
    weights = MatrixCUDA(temp_weights);
    biases = MatrixCUDA(temp_biases);
}

/**
 * Forward Pass
 * 
 * Computes layer output from input using GPU acceleration.
 * 
 * Mathematical Formula:
 *   Z = X·W^T + b    (Linear transformation)
 *   A = σ(Z)         (Activation function)
 * 
 * Where:
 *   X: input (batch × input_size)
 *   W: weights (output_size × input_size)
 *   b: biases (output_size × 1)
 *   Z: pre-activation (batch × output_size)
 *   A: output (batch × output_size)
 * 
 * GPU Operations:
 * 1. W^T: Transpose on GPU (layout change, no computation)
 * 2. X·W^T: cuBLAS matrix multiplication (highly optimized)
 * 3. +b: Broadcast addition using CUDA kernel
 * 4. σ(Z): Activation function CUDA kernel
 * 
 * Memory Efficiency:
 * - No CPU-GPU transfers during forward pass
 * - All intermediate results stay on GPU
 * - Cached values reused in backward pass
 * 
 * Example:
 *   Input: (32 × 784) - 32 images, 784 pixels each
 *   Weights: (128 × 784) - 128 neurons
 *   
 *   Step 1: W^T → (784 × 128)
 *   Step 2: X·W^T → (32 × 128)
 *   Step 3: +b → (32 × 128) [b broadcasted]
 *   Step 4: ReLU → (32 × 128)
 *   Output: (32 × 128)
 */
MatrixCUDA DenseLayerCUDA::forward(const MatrixCUDA& input) {
    // Cache input for backward pass (stays on GPU)
    cached_input = input;
    
    // Step 1: Transpose weights (output × input → input × output)
    MatrixCUDA weights_T = weights.transpose();
    
    // Step 2: Linear transformation Z = X·W^T
    // Uses cuBLAS for optimized matrix multiplication
    MatrixCUDA z = input.multiplyGPU(weights_T);
    
    // Step 3: Add biases (broadcasted across batch)
    // For each sample in batch: z[i] = z[i] + b
    for (size_t i = 0; i < z.getRows(); i++) {
        for (size_t j = 0; j < z.getCols(); j++) {
            double val = z.get(i, j) + biases.get(j, 0);
            z.set(i, j, val);
        }
    }
    
    // Cache pre-activation values for backward pass
    cached_z = z;
    
    // Step 4: Apply activation function (if exists)
    if (activation) {
        // Activation computed on GPU
        return activation->forward(z);
    }
    
    // Linear layer (no activation)
    return z;
}

/**
 * Backward Pass
 * 
 * Computes gradients for learning using the chain rule.
 * 
 * Given: ∂L/∂A (gradient from next layer or loss function)
 * Compute:
 *   1. ∂L/∂W (weight gradients)
 *   2. ∂L/∂b (bias gradients)
 *   3. ∂L/∂X (input gradients for previous layer)
 * 
 * Chain Rule Derivation:
 * 
 * Forward pass: A = σ(Z), Z = X·W^T + b
 * 
 * Step 1: Compute ∂L/∂Z
 *   ∂L/∂Z = ∂L/∂A ⊙ ∂A/∂Z
 *         = ∂L/∂A ⊙ σ'(Z)
 *   
 *   Element-wise multiplication of gradients and activation derivative
 * 
 * Step 2: Compute ∂L/∂W
 *   Z_ij = Σ_k X_ik W_jk + b_j
 *   ∂Z_ij/∂W_jk = X_ik
 *   
 *   ∂L/∂W_jk = Σ_i ∂L/∂Z_ij × ∂Z_ij/∂W_jk
 *            = Σ_i ∂L/∂Z_ij × X_ik
 *   
 *   In matrix form: ∂L/∂W = (∂L/∂Z)^T · X
 *   Shape: (output × batch) · (batch × input) = (output × input)
 * 
 * Step 3: Compute ∂L/∂b
 *   Z_ij = ... + b_j (same bias for all samples)
 *   ∂Z_ij/∂b_j = 1
 *   
 *   ∂L/∂b_j = Σ_i ∂L/∂Z_ij
 *   
 *   Sum gradients across batch dimension
 *   Shape: (batch × output) → (output × 1)
 * 
 * Step 4: Compute ∂L/∂X (for previous layer)
 *   Z_ij = Σ_k X_ik W_jk
 *   ∂Z_ij/∂X_im = W_jm
 *   
 *   ∂L/∂X_im = Σ_j ∂L/∂Z_ij × ∂Z_ij/∂X_im
 *            = Σ_j ∂L/∂Z_ij × W_jm
 *   
 *   In matrix form: ∂L/∂X = ∂L/∂Z · W
 *   Shape: (batch × output) · (output × input) = (batch × input)
 * 
 * GPU Operations:
 * - All matrix operations use cuBLAS
 * - Element-wise operations use CUDA kernels
 * - No CPU-GPU transfers needed
 */
MatrixCUDA DenseLayerCUDA::backward(const MatrixCUDA& output_gradient) {
    // Step 1: Apply activation derivative (if activation exists)
    MatrixCUDA delta = output_gradient;
    
    if (activation) {
        // ∂L/∂Z = activation->backward computes gradient
        // backward(input, output_gradient) returns input_gradient
        delta = activation->backward(cached_z, delta);
    }
    
    // Step 2: Compute weight gradients ∂L/∂W = δ^T · X
    // delta: (batch × output), cached_input: (batch × input)
    // Result: (output × input)
    MatrixCUDA delta_T = delta.transpose();
    weight_gradients = delta_T.multiplyGPU(cached_input);
    
    // Step 3: Compute bias gradients ∂L/∂b = sum(δ, axis=0)
    // Sum across batch dimension
    bias_gradients.zeros();
    for (size_t i = 0; i < delta.getRows(); i++) {
        for (size_t j = 0; j < delta.getCols(); j++) {
            double current = bias_gradients.get(j, 0);
            bias_gradients.set(j, 0, current + delta.get(i, j));
        }
    }
    
    // Step 4: Compute input gradients ∂L/∂X = δ · W
    // delta: (batch × output), weights: (output × input)
    // Result: (batch × input)
    MatrixCUDA input_gradient = delta.multiplyGPU(weights);
    
    return input_gradient;
}

/**
 * Update Parameters
 * 
 * Applies gradient descent to update layer parameters.
 * 
 * Formula:
 *   θ_new = θ_old - α × ∂L/∂θ
 * 
 * Where:
 *   θ: parameters (weights or biases)
 *   α: learning rate
 *   ∂L/∂θ: gradient computed in backward pass
 * 
 * Steps:
 * 1. W = W - learning_rate × ∂L/∂W
 * 2. b = b - learning_rate × ∂L/∂b
 * 
 * GPU Execution:
 * - Operations performed element-wise on GPU
 * - No CPU involvement
 * - Very fast for large parameter counts
 * 
 * Note: This is simple SGD. For advanced optimizers (Adam, RMSprop),
 * use OptimizerCUDA classes which maintain momentum, adaptive rates, etc.
 * 
 * Example:
 *   layer.forward(input);
 *   layer.backward(gradient);
 *   layer.updateParameters(0.01);  // learning_rate = 0.01
 *   layer.resetGradients();        // Clear for next batch
 */
void DenseLayerCUDA::updateParameters(double learning_rate) {
    // Update weights: W = W - α × ∂L/∂W
    for (size_t i = 0; i < weights.getRows(); i++) {
        for (size_t j = 0; j < weights.getCols(); j++) {
            double w = weights.get(i, j);
            double dw = weight_gradients.get(i, j);
            weights.set(i, j, w - learning_rate * dw);
        }
    }
    
    // Update biases: b = b - α × ∂L/∂b
    for (size_t i = 0; i < biases.getRows(); i++) {
        double b = biases.get(i, 0);
        double db = bias_gradients.get(i, 0);
        biases.set(i, 0, b - learning_rate * db);
    }
}

/**
 * Reset Gradients
 * 
 * Clears accumulated gradients by setting them to zero on GPU.
 * 
 * When to use:
 * - After updating parameters
 * - Before starting a new batch
 * - When switching between train/eval modes
 * 
 * Why necessary:
 * - backward() accumulates gradients (doesn't overwrite)
 * - Allows gradient accumulation across mini-batches
 * - Prevents stale gradients from affecting updates
 * 
 * GPU Operation:
 * - CUDA kernel sets all elements to 0
 * - Fast parallel operation
 * - No CPU involvement
 */
void DenseLayerCUDA::resetGradients() {
    weight_gradients.zeros();
    bias_gradients.zeros();
}
