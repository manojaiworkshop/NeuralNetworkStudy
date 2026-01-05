# Loss Functions - Complete Line-by-Line Explanation

## Table of Contents
1. [What Are Loss Functions?](#what-are-loss-functions)
2. [loss.h - Header File Explained](#lossh---header-file-explained)
3. [loss.cpp - Implementation Explained](#losscpp---implementation-explained)
4. [Mathematical Derivations](#mathematical-derivations)
5. [When to Use Which Loss](#when-to-use-which-loss)
6. [Practical Examples](#practical-examples)

---

## What Are Loss Functions?

### The Problem
In machine learning, we need a way to measure **how wrong** our predictions are. This measurement is called the **loss** (or cost/error).

```
Problem: Model predicts house price = $280,000
         Actual house price = $250,000
         
Question: How do we quantify this error?
Answer:   Loss Function!
```

### The Solution
A loss function takes two inputs:
1. **Predictions** (ŷ) - what the model thinks
2. **Targets** (y) - the true values

And outputs a single number:
- **High loss** = Bad predictions
- **Low loss** = Good predictions

### Why We Need Loss
```
┌─────────────────┐
│  Training Loop  │
└─────────────────┘
         │
         ├─► 1. Forward Pass: Input → Network → Prediction
         │
         ├─► 2. Compute Loss: How wrong is the prediction?
         │
         ├─► 3. Backward Pass: Calculate gradients
         │
         └─► 4. Update Weights: W = W - α·∇Loss
                                    └─ Learning rate
```

**Without loss**, we wouldn't know:
- How well the model is doing
- Which direction to adjust weights
- When to stop training

---

## loss.h - Header File Explained

Let's analyze every single line of the header file:

### File Location
```
include/nn/loss.h
```

### Include Guard and Headers
```cpp
#ifndef NN_LOSS_H
#define NN_LOSS_H
```
**Purpose**: Prevent multiple inclusion of this header file
- If `NN_LOSS_H` not defined → Define it and include the contents
- If already defined → Skip this file (already included)

```cpp
#include "matrix.h"
#include <memory>
#include <string>
```
**Dependencies**:
- `matrix.h` - We need Matrix class to work with predictions/targets
- `<memory>` - For `std::unique_ptr` (smart pointers for memory safety)
- `<string>` - For loss function names (`getName()` method)

### Base Loss Class

```cpp
class Loss {
public:
    virtual ~Loss() = default;
```
**Purpose**: Virtual destructor for polymorphism
- `virtual` - Allows derived classes to override
- `= default` - Use compiler-generated destructor
- **Why?** When deleting through base pointer, calls correct destructor

**Example**:
```cpp
Loss* loss = new MSELoss();
delete loss;  // Calls MSELoss destructor (because virtual)
```

---

```cpp
    virtual double calculate(const Matrix& predictions, 
                            const Matrix& targets) const = 0;
```
**Purpose**: Compute the loss value
- `virtual ... = 0` - Pure virtual function (must be overridden)
- `const Matrix&` - Pass by reference (no copy), can't modify
- Returns `double` - The loss value (single number)

**What it does**:
```
Input:  predictions = [2.5, 3.8, 1.2]
        targets     = [2.0, 4.0, 1.5]
        
Output: loss = 0.1267  (for MSE)
```

---

```cpp
    virtual Matrix gradient(const Matrix& predictions, 
                           const Matrix& targets) const = 0;
```
**Purpose**: Compute the gradient of loss w.r.t. predictions
- Returns `Matrix` - Same shape as predictions
- Each element = ∂Loss/∂prediction[i]

**Why we need it**:
```
Training: Need to know how to adjust each prediction
          
          prediction[i] = 2.5
          gradient[i] = 0.333  ← Positive means decrease it
          
Backprop: Gradient flows backward through network
          ∂Loss/∂weights = ∂Loss/∂predictions × ∂predictions/∂weights
                           └─ From this function
```

---

```cpp
    virtual std::string getName() const = 0;
```
**Purpose**: Return human-readable name
- Used for logging/debugging
- Example: `"MSELoss"`, `"BinaryCrossEntropyLoss"`

---

```cpp
    virtual std::unique_ptr<Loss> clone() const = 0;
```
**Purpose**: Create a deep copy of the loss object
- `unique_ptr` - Smart pointer (automatic memory management)
- Used when you need to copy loss function

**Example**:
```cpp
MSELoss mse;
auto copy = mse.clone();  // Creates new MSELoss object
// No memory leaks - unique_ptr deletes automatically
```

---

### MSELoss Class

```cpp
class MSELoss : public Loss {
public:
    double calculate(const Matrix& predictions, 
                    const Matrix& targets) const override;
```

**MSE Formula**: 
```
MSE = (1/n) × Σ(y - ŷ)²
```

**When to use**:
- Regression problems (continuous output)
- Examples: house prices, temperature prediction

**Why squared?**
- Penalizes large errors exponentially
- Always positive
- Smooth gradient (easy to optimize)

**Example**:
```
predictions = [2, 5, 4]
targets     = [1, 4, 6]
differences = [1, 1, -2]
squared     = [1, 1, 4]
MSE = (1+1+4)/3 = 2.0
```

---

```cpp
    Matrix gradient(const Matrix& predictions, 
                   const Matrix& targets) const override;
```

**Gradient Formula**:
```
∂MSE/∂ŷ = (2/n) × (ŷ - y)
```

**Example**:
```
predictions = [3]
targets     = [5]
n = 1

gradient = (2/1) × (3 - 5) = 2 × (-2) = -4

Interpretation: Prediction too low, increase it!
                (negative gradient → increase prediction)
```

---

```cpp
    std::string getName() const override { return "MSELoss"; }
    std::unique_ptr<Loss> clone() const override {
        return std::make_unique<MSELoss>(*this);
    }
};
```

**Inline implementations**:
- `getName()` - Returns string name
- `clone()` - Creates new MSELoss via copy constructor
- `std::make_unique` - Creates smart pointer (C++14)

---

### BinaryCrossEntropyLoss Class

```cpp
class BinaryCrossEntropyLoss : public Loss {
private:
    double epsilon;
```

**Epsilon**: Small value to prevent log(0)
- Default: 1e-7 (0.0000001)
- Clips predictions: [epsilon, 1-epsilon]

**Why?**
```
log(0) = -∞  ← Not good!
log(1e-7) = -16.1  ← Still large but finite
```

---

```cpp
public:
    explicit BinaryCrossEntropyLoss(double eps = 1e-7) : epsilon(eps) {}
```

**Constructor**:
- `explicit` - Prevents implicit conversions
- `eps = 1e-7` - Default parameter
- `: epsilon(eps)` - Initializer list (sets member variable)

**Usage**:
```cpp
BinaryCrossEntropyLoss bce1;        // epsilon = 1e-7
BinaryCrossEntropyLoss bce2(1e-5);  // epsilon = 1e-5 (custom)
```

---

```cpp
    double calculate(const Matrix& predictions, 
                    const Matrix& targets) const override;
```

**BCE Formula**:
```
BCE = -(1/n) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**When to use**:
- Binary classification (2 classes)
- Output: Probability in [0, 1]
- Use with: Sigmoid activation

**Intuition**:
```
If y = 1 (true class):
  ŷ = 0.9 → loss = -log(0.9) = 0.105  (good!)
  ŷ = 0.1 → loss = -log(0.1) = 2.303  (bad!)

If y = 0 (false class):
  ŷ = 0.1 → loss = -log(0.9) = 0.105  (good!)
  ŷ = 0.9 → loss = -log(0.1) = 2.303  (bad!)
```

---

```cpp
    Matrix gradient(const Matrix& predictions, 
                   const Matrix& targets) const override;
```

**Gradient Formula**:
```
∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / n
```

**Simplifies to**:
```
∂BCE/∂ŷ = (ŷ - y) / (ŷ(1-ŷ)n)
```

**Example**:
```
y = 1, ŷ = 0.8, n = 1
gradient = -(1/0.8 - 0/0.2) = -1.25

Interpretation: Increase prediction (it's too low)
```

---

### CategoricalCrossEntropyLoss Class

```cpp
class CategoricalCrossEntropyLoss : public Loss {
private:
    double epsilon;

public:
    explicit CategoricalCrossEntropyLoss(double eps = 1e-7) 
        : epsilon(eps) {}
```

**Similar structure to BCE** but for multiple classes.

---

```cpp
    double calculate(const Matrix& predictions, 
                    const Matrix& targets) const override;
```

**CCE Formula**:
```
CCE = -(1/n) × ΣΣ y_ij · log(ŷ_ij)
      └─ i: samples
         └─ j: classes
```

**When to use**:
- Multi-class classification (>2 classes)
- Targets: One-hot encoded
- Use with: Softmax activation

**Example**:
```
Classes: [cat, dog, bird]

Sample 1:
  predictions = [0.7, 0.2, 0.1]  (70% cat, 20% dog, 10% bird)
  targets     = [1.0, 0.0, 0.0]  (true class: cat)
  
  loss = -[1·log(0.7) + 0·log(0.2) + 0·log(0.1)]
       = -log(0.7) = 0.357
       
Only the probability of the TRUE class matters!
```

---

```cpp
    Matrix gradient(const Matrix& predictions, 
                   const Matrix& targets) const override;
```

**Gradient Formula**:
```
∂CCE/∂ŷ_ij = -y_ij / (ŷ_ij × n)
```

**Example**:
```
y_ij = 1, ŷ_ij = 0.7, n = 1
gradient = -1/(0.7×1) = -1.43

Interpretation: Increase probability of true class
```

---

### MAELoss Class

```cpp
class MAELoss : public Loss {
public:
    double calculate(const Matrix& predictions, 
                    const Matrix& targets) const override;
```

**MAE Formula**:
```
MAE = (1/n) × Σ|y - ŷ|
```

**When to use**:
- Regression problems
- Robust to outliers
- Linear penalty (vs quadratic for MSE)

**MSE vs MAE**:
```
Error = 0.5
  MSE: 0.5² = 0.25
  MAE: |0.5| = 0.5

Error = 5.0
  MSE: 5² = 25.0  ← Heavily penalizes!
  MAE: |5| = 5.0  ← Linear
```

---

```cpp
    Matrix gradient(const Matrix& predictions, 
                   const Matrix& targets) const override;
```

**Gradient Formula**:
```
∂MAE/∂ŷ = sign(ŷ - y) / n

where sign(x) = { +1  if x > 0
                {  0  if x = 0
                { -1  if x < 0
```

**Example**:
```
predictions = [3]
targets     = [5]
n = 1

gradient = sign(3-5)/1 = sign(-2)/1 = -1

Interpretation: Prediction too low, increase it!
```

---

## loss.cpp - Implementation Explained

Now let's analyze the implementation file line by line:

### MSELoss Implementation

```cpp
double MSELoss::calculate(const Matrix& predictions, 
                         const Matrix& targets) const {
```

**Function signature**:
- `MSELoss::` - This is a method of MSELoss class
- `const Matrix&` - Read-only reference (efficient, no copy)
- `const` at end - This method doesn't modify object state

---

```cpp
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "MSELoss: predictions and targets must have same dimensions");
    }
```

**Shape validation**:
- Ensures predictions and targets have same dimensions
- Example:
  ```
  predictions: 2×3 matrix
  targets:     2×3 matrix  ✓ OK
  
  predictions: 2×3 matrix
  targets:     3×2 matrix  ✗ ERROR
  ```

**Why throw exception?**
- Catches bugs early
- Better than silent wrong results
- Standard C++ error handling

---

```cpp
    double sum = 0.0;
    int n = predictions.rows() * predictions.cols();
```

**Initialize variables**:
- `sum` - Accumulator for squared errors
- `n` - Total number of elements
  - For 2×3 matrix: n = 2×3 = 6 elements

---

```cpp
    for (int i = 0; i < predictions.rows(); i++) {
        for (int j = 0; j < predictions.cols(); j++) {
```

**Nested loops**: Iterate over all elements
```
Matrix (2×3):
  Row 0: [a b c]
  Row 1: [d e f]

Loop order:
  i=0, j=0 → a
  i=0, j=1 → b
  i=0, j=2 → c
  i=1, j=0 → d
  i=1, j=1 → e
  i=1, j=2 → f
```

---

```cpp
            double diff = targets.get(i, j) - predictions.get(i, j);
            sum += diff * diff;
```

**Core MSE calculation**:
```
Example:
  target = 5.0
  prediction = 3.0
  
  diff = 5.0 - 3.0 = 2.0
  diff² = 2.0 × 2.0 = 4.0
  sum += 4.0
```

**Why diff×diff instead of pow(diff, 2)?**
- Multiplication is faster
- No need for expensive pow() function

---

```cpp
        }
    }
    return sum / n;
}
```

**Return average**: Divide total by number of elements

**Complete example**:
```
predictions = [2, 5, 4]
targets     = [1, 4, 6]

Iteration 1: diff = 1-2 = -1, sum = 1
Iteration 2: diff = 4-5 = -1, sum = 2
Iteration 3: diff = 6-4 = 2,  sum = 6

return 6/3 = 2.0
```

---

### MSELoss Gradient

```cpp
Matrix MSELoss::gradient(const Matrix& predictions, 
                        const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "MSELoss: predictions and targets must have same dimensions");
    }
```

**Same validation** as `calculate()`.

---

```cpp
    int rows = predictions.rows();
    int cols = predictions.cols();
    int n = rows * cols;
    
    Matrix grad(rows, cols);
```

**Setup**:
- Get dimensions
- Calculate total elements
- Create gradient matrix (same shape as input)

---

```cpp
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            grad.set(i, j, 
                    (2.0 / n) * (predictions.get(i, j) - targets.get(i, j)));
        }
    }
    return grad;
}
```

**Gradient calculation**:
```
Formula: ∂MSE/∂ŷ = (2/n) × (ŷ - y)

Example:
  prediction = 3, target = 5, n = 3
  gradient = (2/3) × (3 - 5) = 0.667 × (-2) = -1.333
  
  Interpretation:
    Negative gradient → Increase prediction
    Magnitude 1.333 → How much to adjust
```

---

### BinaryCrossEntropyLoss Implementation

```cpp
double BinaryCrossEntropyLoss::calculate(const Matrix& predictions, 
                                        const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "BinaryCrossEntropyLoss: predictions and targets must have same dimensions");
    }
```

**Same validation**.

---

```cpp
    double sum = 0.0;
    int n = predictions.rows() * predictions.cols();
    
    for (int i = 0; i < predictions.rows(); i++) {
        for (int j = 0; j < predictions.cols(); j++) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
```

**Setup and get values**.

---

```cpp
            // Clip predictions to avoid log(0)
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
```

**Epsilon clipping** - CRITICAL LINE!

**Purpose**: Prevent `log(0) = -∞`

**How it works**:
```cpp
std::max(epsilon, std::min(1.0 - epsilon, pred))
         └─ Lower bound
                    └─ Upper bound
```

**Example with epsilon = 1e-7**:
```
pred = 0.0     → clipped to 1e-7
pred = 0.5     → stays 0.5
pred = 1.0     → clipped to 1 - 1e-7 = 0.9999999
pred = 0.00001 → stays 0.00001
```

**Why clip both ends?**
```
log(0) = -∞     ← Disaster!
log(1-1) = log(0) = -∞  ← Also disaster!

log(1e-7) ≈ -16  ← Large but finite
```

---

```cpp
            sum += target * std::log(pred) + 
                   (1.0 - target) * std::log(1.0 - pred);
```

**BCE formula implementation**:
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
       └─ Will be negated below
```

**Example**:
```
Case 1: target = 1, pred = 0.9
  sum += 1·log(0.9) + 0·log(0.1)
       = log(0.9) = -0.105
       
Case 2: target = 0, pred = 0.2
  sum += 0·log(0.2) + 1·log(0.8)
       = log(0.8) = -0.223
```

---

```cpp
        }
    }
    return -sum / n;
}
```

**Return negative average**:
```
sum = -1.5  (accumulated log values)
n = 3
result = -(-1.5)/3 = 0.5
```

**Why negative?**
- Log values are negative
- We want loss to be positive
- Negative of negative = positive!

---

### BinaryCrossEntropyLoss Gradient

```cpp
Matrix BinaryCrossEntropyLoss::gradient(const Matrix& predictions, 
                                       const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "BinaryCrossEntropyLoss: predictions and targets must have same dimensions");
    }
    
    int rows = predictions.rows();
    int cols = predictions.cols();
    int n = rows * cols;
    
    Matrix grad(rows, cols);
```

**Standard setup**.

---

```cpp
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            // Clip predictions to avoid division by zero
            pred = std::max(epsilon, std::min(1.0 - epsilon, pred));
```

**Same epsilon clipping** to avoid division by zero.

---

```cpp
            grad.set(i, j, -(target / pred - (1.0 - target) / (1.0 - pred)) / n);
```

**Gradient formula**:
```
∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / n
```

**Derivation**:
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

∂BCE/∂ŷ = -[y·(1/ŷ) + (1-y)·(-1/(1-ŷ))]
        = -[y/ŷ - (1-y)/(1-ŷ)]
```

**Example**:
```
target = 1, pred = 0.8, n = 1
gradient = -(1/0.8 - 0/0.2) / 1
         = -1.25
         
Interpretation: Prediction too low (0.8 < 1.0), increase it!
```

---

### CategoricalCrossEntropyLoss Implementation

```cpp
double CategoricalCrossEntropyLoss::calculate(const Matrix& predictions, 
                                             const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "CategoricalCrossEntropyLoss: predictions and targets must have same dimensions");
    }
    
    double sum = 0.0;
    int n = predictions.rows();  // Number of samples
```

**Note**: `n = rows` (samples), not `rows×cols`
- Each row is one sample
- Each column is one class

---

```cpp
    for (int i = 0; i < predictions.rows(); i++) {
        for (int j = 0; j < predictions.cols(); j++) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            if (target > 0) {  // Only compute for non-zero targets
```

**Optimization**: Skip when target = 0
```
CCE = -Σ y·log(ŷ)

If y = 0:
  0·log(ŷ) = 0  ← Doesn't contribute to sum
  
So we skip these terms!
```

---

```cpp
                pred = std::max(epsilon, pred);  // Clip to avoid log(0)
                sum += target * std::log(pred);
```

**Accumulate**:
```
Only clip lower bound (not upper)
Why? Targets are one-hot (0 or 1), so we only care when target=1
```

---

```cpp
            }
        }
    }
    return -sum / n;
}
```

**Return negative average** (per sample).

**Example**:
```
2 samples, 3 classes

Sample 1: targets = [1, 0, 0], predictions = [0.7, 0.2, 0.1]
  sum += 1·log(0.7) = -0.357

Sample 2: targets = [0, 0, 1], predictions = [0.1, 0.3, 0.6]
  sum += 1·log(0.6) = -0.511

result = -(-0.357-0.511)/2 = 0.434
```

---

### CategoricalCrossEntropyLoss Gradient

```cpp
Matrix CategoricalCrossEntropyLoss::gradient(const Matrix& predictions, 
                                            const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "CategoricalCrossEntropyLoss: predictions and targets must have same dimensions");
    }
    
    int rows = predictions.rows();
    int cols = predictions.cols();
    int n = rows;  // Number of samples
    
    Matrix grad(rows, cols);
```

**Setup**.

---

```cpp
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pred = predictions.get(i, j);
            double target = targets.get(i, j);
            
            pred = std::max(epsilon, pred);  // Avoid division by zero
            
            grad.set(i, j, -target / (pred * n));
        }
    }
    return grad;
}
```

**Gradient formula**:
```
∂CCE/∂ŷ_ij = -y_ij / (ŷ_ij × n)

Example:
  target = 1, pred = 0.7, n = 2
  gradient = -1/(0.7×2) = -0.714
  
  Interpretation: Increase probability of true class
```

---

### MAELoss Implementation

```cpp
double MAELoss::calculate(const Matrix& predictions, 
                         const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "MAELoss: predictions and targets must have same dimensions");
    }
    
    double sum = 0.0;
    int n = predictions.rows() * predictions.cols();
```

**Standard setup**.

---

```cpp
    for (int i = 0; i < predictions.rows(); i++) {
        for (int j = 0; j < predictions.cols(); j++) {
            double diff = targets.get(i, j) - predictions.get(i, j);
            sum += std::abs(diff);
```

**Absolute difference**:
```
std::abs() - Returns absolute value

Example:
  target = 5, prediction = 3
  diff = 5 - 3 = 2
  sum += |2| = 2
  
  target = 2, prediction = 5
  diff = 2 - 5 = -3
  sum += |-3| = 3
```

**Why absolute?**
- Don't want positive/negative errors to cancel out
- Treats over/under predictions equally

---

```cpp
        }
    }
    return sum / n;
}
```

**Return average**.

---

### MAELoss Gradient

```cpp
Matrix MAELoss::gradient(const Matrix& predictions, 
                        const Matrix& targets) const {
    if (predictions.rows() != targets.rows() || 
        predictions.cols() != targets.cols()) {
        throw std::invalid_argument(
            "MAELoss: predictions and targets must have same dimensions");
    }
    
    int rows = predictions.rows();
    int cols = predictions.cols();
    int n = rows * cols;
    
    Matrix grad(rows, cols);
```

**Setup**.

---

```cpp
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = predictions.get(i, j) - targets.get(i, j);
            
            // Gradient is sign of difference
            double sign = (diff > 0) ? 1.0 : ((diff < 0) ? -1.0 : 0.0);
            grad.set(i, j, sign / n);
```

**Sign function**:
```
sign(x) = { +1  if x > 0
          {  0  if x = 0
          { -1  if x < 0

Example:
  prediction = 3, target = 5
  diff = 3 - 5 = -2
  sign = -1
  gradient = -1/n
  
  Interpretation: Prediction too low, increase it!
```

**Ternary operator breakdown**:
```cpp
(condition) ? value_if_true : value_if_false

diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0)
└─ If positive: +1
                └─ If negative: -1
                                └─ Otherwise: 0
```

---

```cpp
        }
    }
    return grad;
}
```

**Return gradient matrix**.

---

## Mathematical Derivations

### MSE Gradient Derivation

**Forward**:
```
MSE = (1/n) Σ(y - ŷ)²

Let E = y - ŷ  (error)

MSE = (1/n) Σ E²
```

**Backward** (Chain Rule):
```
∂MSE/∂ŷ = ∂/∂ŷ [(1/n) Σ(y - ŷ)²]

Step 1: Bring constant out
= (1/n) ∂/∂ŷ [Σ(y - ŷ)²]

Step 2: Derivative of squared term
= (1/n) Σ ∂/∂ŷ [(y - ŷ)²]

Step 3: Chain rule: d(f²) = 2f·df
= (1/n) Σ [2(y - ŷ) · ∂/∂ŷ(y - ŷ)]

Step 4: ∂/∂ŷ(y - ŷ) = -1
= (1/n) Σ [2(y - ŷ) · (-1)]

Step 5: Simplify
= -(2/n) Σ(y - ŷ)

Step 6: Multiply out negative
= (2/n) Σ(ŷ - y)

Final: ∂MSE/∂ŷ = (2/n)(ŷ - y)
```

---

### BCE Gradient Derivation

**Forward**:
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Backward**:
```
∂BCE/∂ŷ = -[y · ∂/∂ŷ(log ŷ) + (1-y) · ∂/∂ŷ(log(1-ŷ))]

Step 1: Derivative of log
d(log x)/dx = 1/x

∂BCE/∂ŷ = -[y · (1/ŷ) + (1-y) · 1/(1-ŷ) · (-1)]
                                           └─ Chain rule

Step 2: Simplify
= -[y/ŷ - (1-y)/(1-ŷ)]

Step 3: Common denominator
= -[(y(1-ŷ) - (1-y)ŷ) / (ŷ(1-ŷ))]

Step 4: Expand numerator
= -[(y - yŷ - ŷ + yŷ) / (ŷ(1-ŷ))]

Step 5: Simplify
= -[(y - ŷ) / (ŷ(1-ŷ))]

Final: ∂BCE/∂ŷ = (ŷ - y) / (ŷ(1-ŷ))
```

---

### CCE Gradient Derivation

**Forward**:
```
CCE = -Σ y_j · log(ŷ_j)
```

**Backward**:
```
∂CCE/∂ŷ_i = ∂/∂ŷ_i [-Σ y_j · log(ŷ_j)]

Only the i-th term depends on ŷ_i:

∂CCE/∂ŷ_i = -y_i · ∂/∂ŷ_i [log(ŷ_i)]

= -y_i · (1/ŷ_i)

Final: ∂CCE/∂ŷ_i = -y_i/ŷ_i
```

---

### MAE Gradient Derivation

**Forward**:
```
MAE = (1/n) Σ|y - ŷ|
```

**Backward**:
```
∂MAE/∂ŷ = (1/n) Σ ∂/∂ŷ [|y - ŷ|]

Derivative of |x| = sign(x)

∂MAE/∂ŷ = (1/n) Σ sign(y - ŷ) · ∂/∂ŷ(y - ŷ)

= (1/n) Σ sign(y - ŷ) · (-1)

= -(1/n) Σ sign(y - ŷ)

= (1/n) Σ sign(ŷ - y)

Final: ∂MAE/∂ŷ = (1/n) · sign(ŷ - y)
```

---

## When to Use Which Loss

### Decision Tree

```
What's your problem type?
├─ REGRESSION (continuous output)
│  ├─ Care more about large errors?
│  │  └─ Use MSE
│  │     • Penalizes outliers heavily
│  │     • Smooth gradients
│  │     • Examples: house prices, stock prediction
│  │
│  └─ Want robustness to outliers?
│     └─ Use MAE
│        • Linear penalty
│        • Robust to anomalies
│        • Examples: median estimation, noisy data
│
└─ CLASSIFICATION (discrete output)
   ├─ Binary (2 classes)
   │  └─ Use Binary Cross-Entropy
   │     • Output: probability [0,1]
   │     • Use with Sigmoid activation
   │     • Examples: spam/not, yes/no, positive/negative
   │
   └─ Multi-class (>2 classes)
      └─ Use Categorical Cross-Entropy
         • Output: probability distribution
         • Use with Softmax activation
         • Examples: digit recognition, image classification
```

### Summary Table

| Problem | Loss | Output | Activation | Example |
|---------|------|--------|------------|---------|
| Regression | MSE | Real number | Linear | House prices |
| Regression (robust) | MAE | Real number | Linear | Median estimation |
| Binary Classification | BCE | [0,1] | Sigmoid | Spam detection |
| Multi-class Classification | CCE | [0,1]ⁿ | Softmax | Image classification |

---

## Practical Examples

### Example 1: House Price Prediction

**Problem**: Predict house price from features

```cpp
// Setup
MSELoss mse;

// True prices
Matrix targets(3, 1);
targets.set(0, 0, 250000);  // $250k
targets.set(1, 0, 180000);  // $180k
targets.set(2, 0, 320000);  // $320k

// Model predictions
Matrix predictions(3, 1);
predictions.set(0, 0, 245000);  // Off by $5k
predictions.set(1, 0, 200000);  // Off by $20k
predictions.set(2, 0, 315000);  // Off by $5k

// Calculate loss
double loss = mse.calculate(predictions, targets);
// loss = [(5000)² + (20000)² + (5000)²] / 3
//      = [25M + 400M + 25M] / 3
//      = 150M

double rmse = sqrt(loss);
// rmse = $12,247 (typical error per house)
```

**Interpretation**: On average, predictions are off by ~$12k.

---

### Example 2: Email Spam Detection

**Problem**: Classify email as spam (1) or not spam (0)

```cpp
BinaryCrossEntropyLoss bce;

// Email 1: Actually spam
Matrix pred1(1, 1);
pred1.set(0, 0, 0.95);  // 95% confidence spam

Matrix target1(1, 1);
target1.set(0, 0, 1.0);  // True: spam

double loss1 = bce.calculate(pred1, target1);
// loss1 = -log(0.95) = 0.051  ← Low loss (good!)

// Email 2: Actually spam but predicted not spam
Matrix pred2(1, 1);
pred2.set(0, 0, 0.1);  // 10% confidence spam

Matrix target2(1, 1);
target2.set(0, 0, 1.0);  // True: spam

double loss2 = bce.calculate(pred2, target2);
// loss2 = -log(0.1) = 2.303  ← High loss (bad!)
```

**Interpretation**: 
- Model confident and correct → Low loss
- Model confident and wrong → High loss

---

### Example 3: Digit Recognition

**Problem**: Recognize handwritten digits (0-9)

```cpp
CategoricalCrossEntropyLoss cce;

// Image is digit "3"
Matrix targets(1, 10);
for (int i = 0; i < 10; i++) {
    targets.set(0, i, (i == 3) ? 1.0 : 0.0);
}
// targets = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
//           └─ Position 3 is "hot"

// Model predictions (probabilities)
Matrix predictions(1, 10);
predictions.set(0, 0, 0.01);  // 1% it's a 0
predictions.set(0, 1, 0.02);  // 2% it's a 1
predictions.set(0, 2, 0.05);  // 5% it's a 2
predictions.set(0, 3, 0.80);  // 80% it's a 3 ✓
predictions.set(0, 4, 0.03);  // ...
// ... rest sum to 0.09

double loss = cce.calculate(predictions, targets);
// loss = -log(0.80) = 0.223

// If model was uncertain:
predictions.set(0, 3, 0.30);  // Only 30%
double loss2 = cce.calculate(predictions, targets);
// loss2 = -log(0.30) = 1.204  ← Much higher!
```

---

## Key Takeaways

1. **Loss measures error**
   - How wrong are our predictions?
   - Always non-negative
   - Lower is better

2. **Different problems need different losses**
   - Regression → MSE or MAE
   - Binary classification → BCE
   - Multi-class → CCE

3. **Loss gradient guides training**
   - Tells us which direction to adjust weights
   - Magnitude tells us how much to adjust
   - Backpropagation uses these gradients

4. **Loss connects to optimization**
   - Training = minimizing loss
   - Gradient descent: W = W - α·∇Loss
   - Stop when loss is small enough

5. **Implementation details matter**
   - Epsilon clipping prevents log(0)
   - Shape validation catches bugs
   - Efficient computation (no unnecessary copies)

---

## Complete Training Example

```cpp
// Initialize network components
Matrix weights(10, 784);  // Random initialization
MSELoss loss;
double learning_rate = 0.01;

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    double total_loss = 0.0;
    
    for (auto& batch : training_data) {
        // 1. FORWARD PASS
        Matrix predictions = network.forward(batch.inputs, weights);
        
        // 2. COMPUTE LOSS
        double batch_loss = loss.calculate(predictions, batch.targets);
        total_loss += batch_loss;
        
        // 3. BACKWARD PASS
        Matrix loss_grad = loss.gradient(predictions, batch.targets);
        Matrix weight_grad = network.backward(loss_grad);
        
        // 4. UPDATE WEIGHTS
        weights = weights - learning_rate * weight_grad;
    }
    
    std::cout << "Epoch " << epoch 
              << ", Loss: " << total_loss/training_data.size() << "\n";
}
```

---

## Conclusion

Loss functions are the **heart of machine learning**:
- They measure how well (or poorly) a model performs
- They provide gradients for optimization
- Different tasks require different loss functions
- Understanding them deeply is crucial for training neural networks

You now understand:
- ✅ What loss functions do and why we need them
- ✅ Every line of code in loss.h and loss.cpp
- ✅ Mathematical derivations of gradients
- ✅ When to use which loss function
- ✅ How loss fits into the training loop
- ✅ Practical implementation details

**Next Steps**:
1. Compile and run the example: `./loss_detailed_example`
2. Try modifying the examples with your own data
3. Implement training loops using these losses
4. Combine with activation functions for full networks

Happy learning! 🎓
