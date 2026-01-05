# ACTIVATION FUNCTIONS - LINE BY LINE CODE EXPLANATION

## TABLE OF CONTENTS
1. [What Are Activation Functions?](#what-are-activation-functions)
2. [Header File (activation.h) - Line by Line](#header-file-line-by-line)
3. [Implementation File (activation.cpp) - Line by Line](#implementation-file-line-by-line)
4. [Matrix Operations in Activations](#matrix-operations-in-activations)
5. [Practical Examples](#practical-examples)

---

## WHAT ARE ACTIVATION FUNCTIONS?

### The Problem Without Activation Functions

```
Neural Network WITHOUT activation:
Input → [W₁·x + b₁] → [W₂·h + b₂] → [W₃·z + b₃] → Output

This simplifies to:
Output = W₃(W₂(W₁·x + b₁) + b₂) + b₃
       = (W₃·W₂·W₁)·x + (combined biases)
       = W_combined·x + b_combined

Result: Just a single linear transformation! 
        Can only learn LINEAR relationships (straight lines, planes)
```

### The Solution: Add Non-Linearity

```
Neural Network WITH activation (σ):
Input → [W₁·x + b₁] → σ(z₁) → [W₂·h + b₂] → σ(z₂) → Output
                       ↑                       ↑
                   NON-LINEAR             NON-LINEAR

Now the network can learn:
- Curves, circles, complex shapes
- XOR problem (impossible without activation)
- Image recognition, language models, etc.
```

### ASCII Diagram of Data Flow

```
┌─────────┐
│ Input X │  Shape: [batch_size × input_features]
│ [2×3]   │  Example: 2 samples, 3 features each
└────┬────┘
     │
     ▼
┌─────────────────────┐
│ Linear Transform    │  z = X·W + b
│ Matrix Multiply     │  W shape: [3×5]
│ z = X × W + b       │  z shape: [2×5]
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Activation Function │  a = σ(z)
│ Applied ELEMENT-WISE│  Each element transformed independently
│ a = σ(z)            │  a shape: [2×5] (same as z)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Next Layer Input    │  This 'a' becomes input to next layer
│ Shape preserved     │  Shape: [2×5]
└─────────────────────┘
```

---

## HEADER FILE LINE BY LINE

### File: `include/nn/activation.h`

```cpp
#ifndef ACTIVATION_H
#define ACTIVATION_H
```
**Explanation:**
- **Include guards**: Prevent multiple inclusion of this header
- If `ACTIVATION_H` is not defined, define it and include the content
- Prevents "redefinition" errors when header is included multiple times

---

```cpp
#include "matrix.h"
#include <memory>
#include <string>
```
**Explanation:**
- `matrix.h`: We need Matrix class since activations operate on matrices
- `<memory>`: For `std::unique_ptr` (smart pointer for memory management)
- `<string>`: For returning activation function names

---

### Base Class: Activation (Abstract Interface)

```cpp
class Activation {
public:
    virtual ~Activation() = default;
```
**Explanation:**
- **Abstract base class**: Defines interface that all activations must implement
- `virtual ~Activation() = default;`: Virtual destructor
  - **Virtual**: Ensures correct destructor is called when deleting through base pointer
  - `= default`: Use compiler-generated destructor
  - **Why needed?**: 
    ```cpp
    Activation* act = new ReLU();  // Base pointer, derived object
    delete act;  // Without virtual destructor, only ~Activation() called!
                 // With virtual, ~ReLU() → ~Activation() called correctly
    ```

---

```cpp
    virtual Matrix forward(const Matrix& input) const = 0;
```
**Explanation:**
- **Forward pass**: Apply activation function to input matrix
- `virtual`: Can be overridden by derived classes
- `= 0`: **Pure virtual function** (abstract method)
  - MUST be implemented by derived classes
  - Makes this class abstract (cannot instantiate Activation directly)
- `const Matrix& input`: Pass by **const reference**
  - No copying (efficient for large matrices)
  - Cannot modify input (const)
- `const` at end: Method doesn't modify the object itself

**Example Usage:**
```cpp
Matrix z(2, 3);  // Some input
z.fill(5.0);
ReLU relu;
Matrix a = relu.forward(z);  // Calls ReLU's forward implementation
```

---

```cpp
    virtual Matrix backward(const Matrix& input, 
                           const Matrix& output_gradient) const = 0;
```
**Explanation:**
- **Backward pass**: Compute gradient for backpropagation
- Used during training to update weights
- **Parameters:**
  - `input`: Original input that was passed to forward()
  - `output_gradient`: Gradient flowing back from next layer (∂L/∂output)
- **Returns:** Gradient with respect to input (∂L/∂input)
- **Chain rule:** 
  ```
  ∂L/∂input = ∂L/∂output ⊙ ∂output/∂input
            = output_gradient ⊙ derivative_of_activation(input)
  ```
  (⊙ is element-wise multiplication, Hadamard product)

**Visual Example:**
```
Forward:  input → activation → output → loss
            x   →     σ(x)   →   y    →  L

Backward:  ∂L/∂x  ←  ∂L/∂y ⊙ σ'(x)
```

---

```cpp
    virtual std::string getName() const = 0;
    virtual std::unique_ptr<Activation> clone() const = 0;
};
```
**Explanation:**
- `getName()`: Returns name of activation ("ReLU", "Sigmoid", etc.)
  - Useful for debugging, logging, model inspection
- `clone()`: Creates a deep copy of the activation function
  - Returns `unique_ptr`: Smart pointer that manages memory automatically
  - **Why unique_ptr?** Automatic memory management, clear ownership
  - **Polymorphic cloning**: Can clone through base pointer

---

### Sigmoid Class

```cpp
class Sigmoid : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Sigmoid"; }
    std::unique_ptr<Activation> clone() const override;
};
```
**Explanation:**
- `public Activation`: **Inherits** from Activation class
- `override`: Keyword that ensures we're actually overriding a virtual function
  - Compiler error if no matching virtual function in base class
  - Helps catch typos and signature mismatches
- `getName()` implemented **inline** in header (simple one-liner)
- Other methods declared here, defined in .cpp file

**Sigmoid Formula:** σ(x) = 1 / (1 + e^(-x))

**Properties:**
- Output range: (0, 1)
- S-shaped curve
- Smooth, differentiable everywhere
- **Used for:** Binary classification output layer

**Graph:**
```
1.0 ┤         ╭─────────  Saturates at 1
    │       ╭─╯
0.5 ┤     ╭─╯            σ(0) = 0.5
    │   ╭─╯
0.0 ┤───╯                Saturates at 0
    └─────────────────► x
     -∞    0    +∞
```

---

### ReLU Class

```cpp
class ReLU : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "ReLU"; }
    std::unique_ptr<Activation> clone() const override;
};
```
**Explanation:**
- **ReLU** = Rectified Linear Unit
- **Formula:** ReLU(x) = max(0, x)
- **Most popular** activation for hidden layers!

**Properties:**
- Output range: [0, ∞)
- Dead simple: negative → 0, positive → unchanged
- **Advantages:**
  - Fast to compute
  - No vanishing gradient for positive values
  - Sparse activation (many zeros)
- **Disadvantage:**
  - "Dying ReLU": neurons can get stuck at 0

**Graph:**
```
 y
 5┤            ╱  Slope = 1
 4┤          ╱
 3┤        ╱
 2┤      ╱
 1┤    ╱
 0┤────╯  Slope = 0 (dead zone)
  └─────────────► x
  -3  -1  0  1  3
```

---

### Tanh Class

```cpp
class Tanh : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Tanh"; }
    std::unique_ptr<Activation> clone() const override;
};
```
**Explanation:**
- **Hyperbolic tangent**
- **Formula:** tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **Properties:**
  - Output range: (-1, 1)
  - **Zero-centered** (unlike sigmoid)
  - Stronger gradients than sigmoid
  - Used in RNNs, LSTMs

**Comparison with Sigmoid:**
```
Sigmoid:        Tanh:
(0, 1)          (-1, 1)  ← Zero-centered is better!
     ╭───            ╭───
   ╭─╯            ╭─╯
 ──╯          ────╯────  ← Passes through origin
              ╯
```

---

### LeakyReLU Class

```cpp
class LeakyReLU : public Activation {
private:
    double alpha;
    
public:
    explicit LeakyReLU(double alpha = 0.01) : alpha(alpha) {}
```
**Explanation:**
- **LeakyReLU** fixes "dying ReLU" problem
- **Formula:** LeakyReLU(x) = x if x > 0, else α·x
- `private: double alpha;`: Stores the leak coefficient
- `explicit`: Prevents implicit type conversions
  ```cpp
  LeakyReLU a(0.01);  // OK
  LeakyReLU b = 0.01; // ERROR without explicit (prevents confusion)
  ```
- `alpha = 0.01`: **Default parameter** (typical value)
- `: alpha(alpha)`: **Member initializer list** (initializes member before constructor body)

**Graph:**
```
     ╱  Slope = 1 (positive region)
   ╱
 ╱────  Slope = α (e.g., 0.01)
       (small but non-zero!)
```

**Advantage over ReLU:**
- Negative inputs still have small gradient (α)
- Neurons can recover from "dead" state

---

### Softmax Class

```cpp
class Softmax : public Activation {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& output_gradient) const override;
    std::string getName() const override { return "Softmax"; }
    std::unique_ptr<Activation> clone() const override;
};
```
**Explanation:**
- **Softmax**: Converts raw scores to probability distribution
- **Formula:** softmax(x_i) = e^(x_i) / Σⱼ e^(x_j)
- **Properties:**
  - Output range: (0, 1)
  - **Sum of outputs = 1** (probability distribution!)
  - Multi-class generalization of sigmoid
- **Used ONLY for:** Output layer in multi-class classification

**Example:**
```
Input (raw scores):  [2.0, 1.0, 0.1]  ← Network outputs
                          ↓
                      Softmax
                          ↓
Output (probabilities): [0.659, 0.242, 0.099]  ← Sum = 1.0
                         
Interpretation: 65.9% chance of class 0
                24.2% chance of class 1
                 9.9% chance of class 2
```

---

## IMPLEMENTATION FILE LINE BY LINE

### File: `src/activation.cpp`

### Sigmoid Implementation

```cpp
Matrix Sigmoid::forward(const Matrix& input) const {
    return input.apply([](double x) {
        return 1.0 / (1.0 + std::exp(-x));
    });
}
```
**Line-by-line Explanation:**

1. **Function signature:** `Matrix Sigmoid::forward(const Matrix& input) const`
   - Member function of `Sigmoid` class
   - Takes input matrix by const reference
   - `const` at end: doesn't modify object state
   - Returns new Matrix (by value)

2. **`input.apply(...)`**: Matrix method that applies function to each element
   - Loops through every element in matrix
   - Applies given function to each element
   - Returns new matrix with transformed values

3. **Lambda function:** `[](double x) { ... }`
   - `[]`: **Capture clause** (empty = captures nothing from outer scope)
   - `(double x)`: Parameter (each matrix element)
   - `{ return 1.0 / (1.0 + std::exp(-x)); }`: Function body
   - **Anonymous function** passed to `apply()`

4. **Sigmoid calculation:** `1.0 / (1.0 + std::exp(-x))`
   - `std::exp(-x)`: Calculates e^(-x)
   - `1.0 + exp(-)`: Denominator
   - `1.0 / (...)`: Division

**Example transformation:**
```
Input matrix:           Output matrix:
[-2.0,  0.0,  2.0]      [0.119, 0.500, 0.881]
[-1.0,  1.0,  3.0]  →   [0.269, 0.731, 0.953]

Element-wise:
  σ(-2.0) = 1/(1+e^2.0)  = 1/8.389 = 0.119
  σ(0.0)  = 1/(1+e^0)    = 1/2     = 0.500
  σ(2.0)  = 1/(1+e^-2.0) = 1/1.135 = 0.881
```

---

```cpp
Matrix Sigmoid::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix activated = forward(input);
```
**Explanation:**
- **Backward pass**: Compute gradient for backpropagation
- `Matrix activated = forward(input);`: First compute σ(x)
  - **Why?** Sigmoid derivative uses σ(x) itself!
  - σ'(x) = σ(x) × (1 - σ(x))

---

```cpp
    Matrix derivative = activated.hadamard(activated.apply([](double x) {
        return 1.0 - x;
    }));
```
**Explanation:**
- **Derivative formula:** σ'(x) = σ(x) × (1 - σ(x))
- **Step 1:** `activated.apply([](double x) { return 1.0 - x; })`
  - Creates matrix: (1 - σ(x))
  - Element-wise: each σ(x) becomes 1 - σ(x)
- **Step 2:** `.hadamard(...)`: Element-wise multiplication
  - Multiplies σ(x) with (1 - σ(x))
  - Result: σ(x) × (1 - σ(x)) for each element

**Example:**
```
activated (σ(x)):    [0.119, 0.500, 0.881]
1 - σ(x):            [0.881, 0.500, 0.119]
σ'(x) = σ(x)×(1-σ):  [0.105, 0.250, 0.105]  ← Derivative
```

---

```cpp
    return derivative.hadamard(output_gradient);
}
```
**Explanation:**
- **Chain rule:** ∂L/∂input = ∂L/∂output ⊙ ∂output/∂input
- `derivative`: ∂output/∂input (local gradient)
- `output_gradient`: ∂L/∂output (gradient from next layer)
- `.hadamard(...)`: Element-wise multiply (⊙)

**Visualization:**
```
Forward flow:
  input (x) → Sigmoid → output (y) → Loss (L)

Backward flow (chain rule):
  ∂L/∂x = ∂L/∂y ⊙ ∂y/∂x
        = output_gradient ⊙ derivative

Example:
  derivative:       [0.105, 0.250, 0.105]  ← Local gradient
  output_gradient:  [1.000, 0.500, 2.000]  ← From next layer
  result:           [0.105, 0.125, 0.210]  ← Gradient to pass back
```

---

```cpp
std::unique_ptr<Activation> Sigmoid::clone() const {
    return std::make_unique<Sigmoid>();
}
```
**Explanation:**
- **Clone pattern**: Create a copy of the activation
- `std::make_unique<Sigmoid>()`: Creates new Sigmoid on heap
  - Returns `unique_ptr` (smart pointer)
  - **Automatic memory management** (no need for delete)
- **Use case:** 
  ```cpp
  std::unique_ptr<Activation> act1 = std::make_unique<Sigmoid>();
  std::unique_ptr<Activation> act2 = act1->clone();  // Deep copy
  ```

---

### ReLU Implementation

```cpp
Matrix ReLU::forward(const Matrix& input) const {
    return input.apply([](double x) {
        return std::max(0.0, x);
    });
}
```
**Explanation:**
- **ReLU formula:** max(0, x)
- `std::max(0.0, x)`: Built-in function returns larger of two values
- **Logic:**
  - If x > 0: returns x
  - If x ≤ 0: returns 0.0

**Example:**
```
Input:   [-2.0, -0.5,  0.0,  1.5,  3.0]
Output:  [ 0.0,  0.0,  0.0,  1.5,  3.0]
          ↑     ↑     ↑     ↑     ↑
         neg   neg   zero   pos   pos
         →0    →0    →0     →same →same
```

---

```cpp
Matrix ReLU::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix derivative = input.apply([](double x) {
        return (x > 0.0) ? 1.0 : 0.0;
    });
    return derivative.hadamard(output_gradient);
}
```
**Explanation:**
- **ReLU derivative:**
  - 1 if x > 0 (gradient flows through)
  - 0 if x ≤ 0 (gradient blocked, "dead neuron")
- `(x > 0.0) ? 1.0 : 0.0`: **Ternary operator**
  - Condition ? value_if_true : value_if_false
- **Effect:** Only positive inputs receive gradients

**Example:**
```
input:            [-2.0,  1.5,  3.0]
derivative:       [ 0.0,  1.0,  1.0]  ← ReLU derivative
output_gradient:  [ 0.5,  0.8,  1.2]  ← From next layer
result:           [ 0.0,  0.8,  1.2]  ← Gradient passed back
                    ↑
                 Blocked! (dying ReLU problem)
```

---

### LeakyReLU Implementation

```cpp
Matrix LeakyReLU::forward(const Matrix& input) const {
    return input.apply([this](double x) {
        return (x > 0.0) ? x : alpha * x;
    });
}
```
**Explanation:**
- **LeakyReLU formula:** x if x > 0, else α·x
- `[this]`: **Capture clause** captures `this` pointer
  - Allows accessing member variable `alpha` inside lambda
  - Without `[this]`, cannot access `alpha`
- `alpha * x`: Small gradient for negative values (typically α = 0.01)

**Example with α = 0.01:**
```
Input:   [-2.0, -1.0,  0.0,  1.0,  2.0]
Output:  [-0.02,-0.01, 0.0,  1.0,  2.0]
          ↑     ↑      ↑     ↑     ↑
         ×0.01 ×0.01  0     same  same
         (leak) (leak)
```

---

```cpp
Matrix LeakyReLU::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix derivative = input.apply([this](double x) {
        return (x > 0.0) ? 1.0 : alpha;
    });
    return derivative.hadamard(output_gradient);
}
```
**Explanation:**
- **LeakyReLU derivative:**
  - 1 if x > 0
  - α if x ≤ 0 (small but non-zero!)
- **Advantage:** Gradient still flows for negative inputs
  - Neurons can recover from "dead" state
  - Better than standard ReLU

---

### Softmax Implementation

```cpp
Matrix Softmax::forward(const Matrix& input) const {
    Matrix result(input.getRows(), input.getCols());
    
    // Process each row independently (each sample)
    for (size_t i = 0; i < input.getRows(); ++i) {
```
**Explanation:**
- `Matrix result(...)`: Create output matrix (same dimensions as input)
- **Row-wise processing**: Each row is one sample
  - For batch of data: each row is independent
  - Softmax applied separately to each sample
- Loop through rows (samples)

---

```cpp
        // Find max for numerical stability
        double max_val = input.get(i, 0);
        for (size_t j = 1; j < input.getCols(); ++j) {
            max_val = std::max(max_val, input.get(i, j));
        }
```
**Explanation:**
- **Numerical stability trick**: Subtract max before exp
- **Problem:** e^(large number) can overflow
  ```
  e^1000 = ∞ (overflow!)
  ```
- **Solution:** 
  ```
  softmax(x) = e^x / Σe^x
             = e^(x - max) / Σe^(x - max)  ← Subtract max
  ```
- **Why it works:** Mathematically equivalent, but prevents overflow

---

```cpp
        // Compute exp(x - max) and sum
        double sum = 0.0;
        std::vector<double> exp_vals(input.getCols());
        for (size_t j = 0; j < input.getCols(); ++j) {
            exp_vals[j] = std::exp(input.get(i, j) - max_val);
            sum += exp_vals[j];
        }
```
**Explanation:**
- `exp_vals`: Temporary storage for e^(x - max)
- Loop through columns (classes):
  1. Compute e^(x - max) for each element
  2. Store in `exp_vals`
  3. Accumulate sum: Σe^(x - max)

**Example:**
```
Input row:    [2.0, 1.0, 0.5]
max_val:      2.0

After subtract: [0.0, -1.0, -1.5]
After exp:      [1.0, 0.368, 0.223]
sum:            1.591
```

---

```cpp
        // Normalize
        for (size_t j = 0; j < input.getCols(); ++j) {
            result.set(i, j, exp_vals[j] / sum);
        }
    }
    return result;
}
```
**Explanation:**
- **Normalization**: Divide each exp by sum
- Result: Each value in (0, 1), sum = 1.0 (probability distribution)

**Example continued:**
```
exp_vals:       [1.0,   0.368, 0.223]
sum:            1.591
After divide:   [0.629, 0.231, 0.140]  ← Probabilities!
                 ↑      ↑      ↑
                62.9%  23.1%  14.0%   (sums to 100%)
```

---

### Softmax Backward Pass

```cpp
Matrix Softmax::backward(const Matrix& input, const Matrix& output_gradient) const {
    Matrix output = forward(input);
    Matrix result(input.getRows(), input.getCols());
    
    // For each sample in the batch
    for (size_t i = 0; i < input.getRows(); ++i) {
```
**Explanation:**
- Softmax backward is **complex** because outputs depend on ALL inputs
- Not element-wise like ReLU!
- Must compute full **Jacobian matrix** for each sample

---

```cpp
        for (size_t j = 0; j < input.getCols(); ++j) {
            double grad = 0.0;
            for (size_t k = 0; k < input.getCols(); ++k) {
                double s_i = output.get(i, j);
                double s_k = output.get(i, k);
                double kronecker = (j == k) ? 1.0 : 0.0;
                grad += output_gradient.get(i, k) * s_i * (kronecker - s_k);
            }
            result.set(i, j, grad);
        }
    }
    return result;
}
```
**Explanation:**
- **Jacobian formula:** ∂softmax_i/∂x_j = softmax_i × (δᵢⱼ - softmax_j)
  - δᵢⱼ (Kronecker delta): 1 if i==j, else 0
- **Three nested loops:**
  1. i: samples
  2. j: output dimensions
  3. k: input dimensions (summing over)
- **Computation:**
  - `s_i`: softmax output at position j
  - `s_k`: softmax output at position k
  - `kronecker`: 1 if j==k (diagonal), else 0
  - Formula: grad += ∂L/∂outputₖ × softmaxᵢ × (δⱼₖ - softmaxₖ)

**Why so complex?** 
Each softmax output depends on ALL inputs (normalization by sum)!

---

## MATRIX OPERATIONS IN ACTIVATIONS

### How Matrices Flow Through Activations

```
EXAMPLE: Batch of 3 samples, 4 features each

Input Matrix (3×4):
  ┌                    ┐
  │ 0.5  -1.2  2.0  0.8 │ ← Sample 1
  │-0.3   1.5 -2.1  0.0 │ ← Sample 2
  │ 1.0   0.5  0.2 -1.0 │ ← Sample 3
  └                    ┘

Apply ReLU (element-wise):
  ┌                    ┐
  │ 0.5   0.0  2.0  0.8 │ ← max(0, each element)
  │ 0.0   1.5  0.0  0.0 │
  │ 1.0   0.5  0.2  0.0 │
  └                    ┘

Result: Same shape (3×4), each element transformed independently
```

### Memory Layout

```
Matrix internal representation (std::vector<std::vector<double>>):

Conceptual:              Actual Memory (heap):
┌───┬───┬───┐           ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │   Row 0  │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │...│...│...│
├───┼───┼───┤    ↓      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
│ 4 │ 5 │ 6 │   Row 1   ↑       ↑       ↑
├───┼───┼───┤    ↓      Row 0   Row 1   Row 2
│ 7 │ 8 │ 9 │   Row 2
└───┴───┴───┘

When apply() is called:
1. Loop through each row
2. Loop through each element in row
3. Apply function to each element
4. Store result in new matrix
```

### Parallelization Potential

```
Element-wise operations (ReLU, Sigmoid, Tanh):
  • Each element independent
  • Perfect for GPU parallelization!
  • Can process millions of elements simultaneously

GPU Implementation:
  ┌─────────────────────────────────┐
  │ Thread 1: Process element[0,0]  │ ← Parallel
  │ Thread 2: Process element[0,1]  │ ← Parallel
  │ Thread 3: Process element[0,2]  │ ← Parallel
  │ ...                             │
  │ Thread N: Process element[M,N]  │ ← Parallel
  └─────────────────────────────────┘
  All threads execute simultaneously!

Softmax:
  • Each row depends on sum of ALL elements in that row
  • Still parallelizable per row
  • More complex than element-wise
```

---

## PRACTICAL EXAMPLES

### Example 1: Binary Classification

```cpp
// Input: [batch=2, features=3]
Matrix x(2, 3);
x.set(0, 0, 0.5); x.set(0, 1, 0.8); x.set(0, 2, 0.3);
x.set(1, 0, 0.2); x.set(1, 1, 0.6); x.set(1, 2, 0.9);

// Weights: [features=3, hidden=4]
Matrix W1(3, 4);
W1.randomize(-1, 1);

// Layer 1: Linear + ReLU
Matrix z1 = x * W1;           // [2×4]
ReLU relu;
Matrix a1 = relu.forward(z1); // [2×4], activated

// Weights: [hidden=4, output=1]
Matrix W2(4, 1);
W2.randomize(-1, 1);

// Layer 2: Linear + Sigmoid
Matrix z2 = a1 * W2;           // [2×1]
Sigmoid sigmoid;
Matrix output = sigmoid.forward(z2);  // [2×1], probability

// output[0,0] = probability sample 0 is positive class
// output[1,0] = probability sample 1 is positive class
```

### Example 2: Multi-class Classification

```cpp
// 3 samples, 3 classes
Matrix logits(3, 3);
logits.set(0, 0, 2.0); logits.set(0, 1, 1.0); logits.set(0, 2, 0.1);  // Sample 0
logits.set(1, 0, 0.5); logits.set(1, 1, 2.5); logits.set(1, 2, 0.3);  // Sample 1
logits.set(2, 0, 1.5); logits.set(2, 1, 0.8); logits.set(2, 2, 3.0);  // Sample 2

Softmax softmax;
Matrix probs = softmax.forward(logits);  // [3×3]

// probs[0,0], probs[0,1], probs[0,2] sum to 1.0 (sample 0)
// probs[1,0], probs[1,1], probs[1,2] sum to 1.0 (sample 1)
// probs[2,0], probs[2,1], probs[2,2] sum to 1.0 (sample 2)

// Predict class with highest probability:
int predicted_class_0 = argmax(probs, row=0);  // Class with max prob
```

### Example 3: Gradient Flow

```cpp
// Forward pass
Matrix x(1, 3);
x.fill(2.0);  // [2.0, 2.0, 2.0]

ReLU relu;
Matrix y = relu.forward(x);  // [2.0, 2.0, 2.0] (all positive, unchanged)

// Backward pass (simulated gradient)
Matrix grad_y(1, 3);
grad_y.fill(1.0);  // Gradient from loss

Matrix grad_x = relu.backward(x, grad_y);
// grad_x = [1.0, 1.0, 1.0] (all passed through)

// If x was negative:
x.fill(-2.0);  // [-2.0, -2.0, -2.0]
y = relu.forward(x);  // [0.0, 0.0, 0.0]
grad_x = relu.backward(x, grad_y);  // [0.0, 0.0, 0.0] (blocked!)
```

---

## SUMMARY

### Key Concepts

1. **Activation functions add non-linearity** to neural networks
2. **Element-wise operations** preserve matrix shape
3. **Forward pass** transforms data through network
4. **Backward pass** computes gradients for learning
5. **Polymorphism** allows using different activations through base pointer

### Design Patterns

1. **Abstract base class** defines interface
2. **Virtual functions** enable polymorphism
3. **Lambda functions** for concise element-wise operations
4. **Smart pointers** for automatic memory management
5. **Const correctness** prevents accidental modifications

### Performance Considerations

1. **Element-wise activations** (ReLU, Sigmoid) are highly parallelizable
2. **Matrix operations** use std::vector (dynamic heap allocation)
3. **GPU acceleration** can speed up by 10-100x (see matrix_cuda)
4. **Memory layout** matters for cache efficiency

---

## BUILD AND RUN THE EXAMPLE

```bash
# Add to CMakeLists.txt
add_executable(activation_example
    example/activation_detailed_example.cpp
    src/activation.cpp
)
target_link_libraries(activation_example matrix_lib)

# Build
cd build
cmake ..
make activation_example

# Run
./activation_example
```

This will demonstrate all activation functions with detailed explanations!
