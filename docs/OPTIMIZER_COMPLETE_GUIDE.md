# Optimizer Classes - Complete Line-by-Line Explanation

## Table of Contents
1. [What Are Optimizers?](#what-are-optimizers)
2. [optimizer.h - Header File Explained](#optimizerh---header-file-explained)
3. [optimizer.cpp - Implementation Explained](#optimizercpp---implementation-explained)
4. [When to Use Which Optimizer](#when-to-use-which-optimizer)
5. [Mathematical Details](#mathematical-details)
6. [Practical Examples](#practical-examples)

---

## What Are Optimizers?

### The Problem
In machine learning, we need to minimize the loss function by adjusting weights:

```
Goal: Find weights W that minimize Loss(W)

How? Update weights iteratively:
  W_new = W_old - ?? (some update rule)
```

### The Solution
**Optimizers** determine HOW to update weights based on gradients.

```
Training Loop:
┌─────────────────────────┐
│ 1. Forward Pass         │ → Get predictions
│ 2. Compute Loss         │ → Measure error
│ 3. Backward Pass        │ → Get gradients ∇L/∂W
│ 4. OPTIMIZER STEP       │ → Update weights W
└─────────────────────────┘
         ↓
    Repeat until converged
```

### Why Different Optimizers?

Simple gradient descent has problems:
- ❌ Can get stuck in local minima
- ❌ Slow convergence
- ❌ Sensitive to learning rate
- ❌ Same update for all parameters

Advanced optimizers solve these issues!

---

## optimizer.h - Header File Explained

### Include Guard and Headers

```cpp
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
```
**Purpose**: Prevent multiple inclusion
- Standard C++ header guard
- Ensures file is only included once per compilation unit

---

```cpp
#include "matrix.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
```
**Dependencies**:
- `matrix.h` - We work with parameter matrices
- `<memory>` - Smart pointers (std::unique_ptr)
- `<string>` - Parameter identifiers ("layer1_weights", etc.)
- `<vector>` - Collections
- `<unordered_map>` - Store per-parameter state (velocities, moments)

**Why unordered_map?**
```cpp
// Each parameter (weights, biases) needs its own state
unordered_map<string, Matrix> velocity;
  └─ Key: "layer1_weights"
  └─ Value: Velocity matrix for those weights
```

---

### Base Optimizer Class

```cpp
class Optimizer {
protected:
    double learning_rate;
```
**Learning Rate**: How big are our update steps?
- `protected` - Derived classes can access
- Controls step size in weight updates

**Analogy**: Walking downhill
- Large learning_rate = Big steps (fast but might overshoot)
- Small learning_rate = Tiny steps (slow but precise)

**Example**:
```
Loss landscape:
     
    ╱╲
   ╱  ╲
  ╱    ╲___
 
α = 0.1 (good)    → Steps down slope smoothly
α = 1.0 (too big) → Jumps over minimum!
α = 0.001 (small) → Takes forever to reach bottom
```

---

```cpp
public:
    explicit Optimizer(double learning_rate = 0.01) : learning_rate(learning_rate) {}
```
**Constructor**:
- `explicit` - Prevents implicit conversions
- Default learning_rate = 0.01 (common starting value)
- `: learning_rate(learning_rate)` - Initializer list

**Usage**:
```cpp
Optimizer* opt1 = new SGD();        // lr = 0.01 (default)
Optimizer* opt2 = new SGD(0.001);   // lr = 0.001 (custom)
```

---

```cpp
    virtual ~Optimizer() = default;
```
**Virtual Destructor**:
- Allows proper cleanup through base pointer
- `= default` - Use compiler-generated version

**Why virtual?**
```cpp
Optimizer* opt = new Adam();
delete opt;  // Calls Adam's destructor (because virtual)
```

---

```cpp
    virtual Matrix update(const Matrix& parameters, const Matrix& gradients, 
                         const std::string& param_id) = 0;
```
**Core Method**: Update parameters using gradients

**Parameters**:
1. `parameters` - Current weight values (e.g., W = [[1, 2], [3, 4]])
2. `gradients` - ∂Loss/∂W (how loss changes with weights)
3. `param_id` - Identifier (e.g., "layer1_weights")

**Returns**: Updated parameters

**Why param_id?**
```cpp
// Different parameters need separate state
update(W1, ∇W1, "layer1_weights");  // Tracks velocity for layer 1
update(W2, ∇W2, "layer2_weights");  // Tracks velocity for layer 2
```

**`= 0`**: Pure virtual (must override in derived classes)

---

```cpp
    virtual std::string getName() const = 0;
```
**Get Optimizer Name**: For logging/debugging
- Returns: "SGD", "Adam", "RMSprop", etc.

**Example**:
```cpp
cout << "Using optimizer: " << optimizer->getName() << endl;
// Output: Using optimizer: Adam
```

---

```cpp
    virtual void reset() {}
```
**Reset State**: Clear accumulated history
- Default: Does nothing (stateless optimizers like SGD)
- Override: Stateful optimizers (Adam, Momentum) clear their caches

**When to use?**
```cpp
// Train on dataset 1
for (epoch in range(100)) {
    train();
}

optimizer->reset();  // Clear accumulated moments

// Train on completely different dataset 2
for (epoch in range(100)) {
    train();
}
```

---

```cpp
    void setLearningRate(double lr) { learning_rate = lr; }
    double getLearningRate() const { return learning_rate; }
```
**Learning Rate Access**:
- **Setter**: Change learning rate during training
- **Getter**: Query current learning rate

**Learning Rate Scheduling**:
```cpp
// Start with large learning rate, decrease over time
for (int epoch = 0; epoch < 100; epoch++) {
    if (epoch == 50) {
        optimizer->setLearningRate(0.001);  // Reduce after 50 epochs
    }
    train();
}
```

---

### SGD (Stochastic Gradient Descent)

```cpp
/**
 * @brief Stochastic Gradient Descent optimizer
 * Update: θ = θ - α * ∇θ
 */
class SGD : public Optimizer {
```
**Simplest Optimizer**:
- Formula: `W_new = W_old - learning_rate × gradient`
- No memory of past gradients
- Direct descent down gradient

**When to use?**
- Simple problems
- When you want baseline performance
- Fast computation (no state to maintain)

**ASCII Diagram**:
```
Current position: ●
Gradient points: ↓ (downhill)

Step: Move opposite to gradient
●  →  ○ (new position)
```

---

```cpp
public:
    explicit SGD(double learning_rate = 0.01) : Optimizer(learning_rate) {}
```
**Constructor**: Calls base Optimizer constructor

---

```cpp
    Matrix update(const Matrix& parameters, const Matrix& gradients, 
                 const std::string& param_id) override;
```
**Update Method**: 
- `override` - Implements pure virtual function from base
- Implementation in .cpp file

---

```cpp
    std::string getName() const override { return "SGD"; }
```
**Inline Implementation**: Simple enough to define in header

---

### Momentum Optimizer

```cpp
/**
 * @brief SGD with Momentum
 * v = β * v + ∇θ
 * θ = θ - α * v
 */
class Momentum : public Optimizer {
```
**Concept**: Accumulate velocity in descent direction

**Problem with SGD**:
```
Loss surface:
    ╱╲╱╲╱╲
   ╱      ╲
  ╱        ╲

SGD oscillates: ╱╲╱╲╱╲  (zigzags)
```

**Solution with Momentum**:
```
Momentum smooths out oscillations:
    ────────▼ (smooth descent)

Like a ball rolling downhill - builds up speed!
```

---

```cpp
private:
    double beta;  // Momentum coefficient (typically 0.9)
    std::unordered_map<std::string, Matrix> velocity;
```
**Private Members**:
1. **beta**: Momentum coefficient
   - Typical value: 0.9
   - Controls how much past gradients influence current update
   - Higher beta = More momentum (more smoothing)

2. **velocity**: Per-parameter velocity accumulation
   - `unordered_map` - Fast O(1) lookup
   - Key: parameter name
   - Value: accumulated velocity matrix

**Mathematical Meaning**:
```
β = 0.9 means:
  - 90% of previous velocity
  + 10% of current gradient

Example:
  t=0: v = 0
  t=1: v = 0.9×0 + 1×grad₁ = grad₁
  t=2: v = 0.9×grad₁ + 1×grad₂
  t=3: v = 0.9×(0.9×grad₁ + grad₂) + grad₃
  
Exponentially decaying average of past gradients!
```

---

```cpp
public:
    explicit Momentum(double learning_rate = 0.01, double beta = 0.9)
        : Optimizer(learning_rate), beta(beta) {}
```
**Constructor**:
- Default lr = 0.01
- Default beta = 0.9 (common choice)
- Initializes both base class and beta

---

```cpp
    void reset() override { velocity.clear(); }
```
**Reset Implementation**: Clear velocity cache
- Important when switching to new task
- Prevents old momentum from affecting new training

---

### RMSprop Optimizer

```cpp
/**
 * @brief RMSprop optimizer
 * v = β * v + (1 - β) * ∇θ²
 * θ = θ - α * ∇θ / (√v + ε)
 */
class RMSprop : public Optimizer {
```
**Concept**: Adapt learning rate per parameter

**Problem**: Some parameters need bigger steps, others need smaller
```
Parameter 1: Steep gradient  → needs small lr
Parameter 2: Gentle gradient → needs large lr

Solution: Divide gradient by its running average!
```

**Key Insight**: 
- Frequently updated parameters get smaller updates
- Rarely updated parameters get larger updates

---

```cpp
private:
    double beta;     // Decay rate (typically 0.9)
    double epsilon;  // Small constant for numerical stability
    std::unordered_map<std::string, Matrix> cache;
```
**Members**:
1. **beta**: Decay rate for running average
   - How quickly we forget old gradient information

2. **epsilon**: Prevents division by zero
   - Typical: 1e-8
   - Ensures denominator never exactly zero

3. **cache**: Running average of squared gradients
   - Tracks magnitude of gradients over time

**Why squared gradients?**
```
If gradient is consistently large:
  → gradient² is very large
  → √cache is large
  → division makes update smaller
  
If gradient is small:
  → gradient² is very small
  → √cache is small
  → division makes update larger
```

---

```cpp
public:
    explicit RMSprop(double learning_rate = 0.001, double beta = 0.9, double epsilon = 1e-8)
        : Optimizer(learning_rate), beta(beta), epsilon(epsilon) {}
```
**Constructor**:
- Default lr = 0.001 (smaller than SGD!)
- Default beta = 0.9
- Default epsilon = 1e-8

**Note**: RMSprop typically uses smaller learning rate than SGD

---

### Adam Optimizer

```cpp
/**
 * @brief Adam optimizer (Adaptive Moment Estimation)
 * Combines momentum and RMSprop
 * 
 * m = β₁ * m + (1 - β₁) * ∇θ
 * v = β₂ * v + (1 - β₂) * ∇θ²
 * m̂ = m / (1 - β₁^t)
 * v̂ = v / (1 - β₂^t)
 * θ = θ - α * m̂ / (√v̂ + ε)
 */
class Adam : public Optimizer {
```
**Concept**: Best of both worlds!
- **Momentum** - Smooths gradient direction
- **RMSprop** - Adapts per-parameter learning rates
- **Bias Correction** - Corrects initialization bias

**Why Adam is Popular**:
✓ Works well out-of-the-box
✓ Adaptive per-parameter learning rates
✓ Momentum helps escape local minima
✓ Bias correction ensures good early training

---

```cpp
private:
    double beta1;    // First moment decay rate (typically 0.9)
    double beta2;    // Second moment decay rate (typically 0.999)
    double epsilon;  // Small constant for numerical stability
    
    std::unordered_map<std::string, Matrix> m;  // First moment
    std::unordered_map<std::string, Matrix> v;  // Second moment
    std::unordered_map<std::string, int> t;     // Time step
```
**Members Explained**:

1. **beta1** (0.9): Controls momentum
   - First moment = running average of gradient
   - Like Momentum optimizer

2. **beta2** (0.999): Controls adaptive learning rate
   - Second moment = running average of squared gradient
   - Like RMSprop optimizer

3. **epsilon** (1e-8): Numerical stability

4. **m**: First moment estimates (momentum)
   ```
   m ≈ E[∇θ]  (expected value of gradient)
   ```

5. **v**: Second moment estimates (variance)
   ```
   v ≈ E[∇θ²]  (expected value of squared gradient)
   ```

6. **t**: Time step counter
   - Needed for bias correction
   - Each parameter tracked separately

**Why two betas?**
```
β₁ = 0.9   → Fast adaptation to direction (momentum)
β₂ = 0.999 → Slow adaptation to magnitude (stability)

Different time scales for different purposes!
```

---

```cpp
public:
    explicit Adam(double learning_rate = 0.001, double beta1 = 0.9, 
                 double beta2 = 0.999, double epsilon = 1e-8)
        : Optimizer(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}
```
**Constructor**: 
- Default lr = 0.001
- Defaults match original Adam paper
- These defaults work well for most problems!

---

```cpp
    void reset() override { 
        m.clear(); 
        v.clear(); 
        t.clear(); 
    }
```
**Reset**: Clear all accumulated state
- Three separate caches to clear
- Necessary for multi-task learning

---

### AdaGrad Optimizer

```cpp
/**
 * @brief AdaGrad optimizer
 * G = G + ∇θ²
 * θ = θ - α * ∇θ / (√G + ε)
 */
class AdaGrad : public Optimizer {
```
**Concept**: Adaptive learning rates, never forget

**Key Difference from RMSprop**:
- RMSprop: Running average (forgets old gradients)
- AdaGrad: Sum of ALL gradients (never forgets)

**Use Case**: Sparse data (NLP, recommendation systems)
- Frequent features → get small updates
- Rare features → get large updates

**Problem**: Learning rate monotonically decreases
```
As training progresses:
  G gets larger and larger
  → Updates get smaller and smaller
  → Eventually stops learning!
```

---

```cpp
private:
    double epsilon;  // Small constant for numerical stability
    std::unordered_map<std::string, Matrix> accumulated_gradients;
```
**Members**:
1. **epsilon**: Division safety
2. **accumulated_gradients**: Sum of squared gradients
   - Keeps growing throughout training
   - Never decays (unlike RMSprop)

---

```cpp
public:
    explicit AdaGrad(double learning_rate = 0.01, double epsilon = 1e-8)
        : Optimizer(learning_rate), epsilon(epsilon) {}
```
**Constructor**:
- Default lr = 0.01
- Only needs epsilon (no decay rates like RMSprop/Adam)

---

```cpp
    void reset() override { accumulated_gradients.clear(); }
```
**Reset**: Clear accumulated gradient history

---

## optimizer.cpp - Implementation Explained

### SGD Implementation

```cpp
Matrix SGD::update(const Matrix& parameters, const Matrix& gradients, 
                   const std::string& param_id) {
    // Simple gradient descent: θ = θ - α * ∇θ
    return parameters - gradients * learning_rate;
}
```
**Line-by-Line**:
1. `parameters - gradients * learning_rate`
   - Multiply gradient by learning rate
   - Subtract from current parameters
   - Move opposite to gradient (downhill)

**Example**:
```cpp
Matrix W(2, 2);  // Weights: [[1, 2], [3, 4]]
Matrix grad(2, 2);  // Gradients: [[0.1, 0.2], [0.3, 0.4]]
double lr = 0.1;

W_new = W - grad * lr
      = [[1, 2], [3, 4]] - [[0.1, 0.2], [0.3, 0.4]] * 0.1
      = [[1, 2], [3, 4]] - [[0.01, 0.02], [0.03, 0.04]]
      = [[0.99, 1.98], [2.97, 3.96]]
```

---

### Momentum Implementation

```cpp
Matrix Momentum::update(const Matrix& parameters, const Matrix& gradients, 
                       const std::string& param_id) {
```
**Function Start**: Same signature as base class

---

```cpp
    // Initialize velocity if not exists
    if (velocity.find(param_id) == velocity.end()) {
        velocity[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        velocity[param_id].zeros();
    }
```
**First-Time Initialization**:
1. Check if we've seen this parameter before
2. If not, create velocity matrix (same shape as parameters)
3. Initialize to zeros

**Why check?**
```cpp
First call: velocity["layer1"] doesn't exist → create it
Second call: velocity["layer1"] exists → use it
```

---

```cpp
    // Update velocity: v = β * v + ∇θ
    velocity[param_id] = velocity[param_id] * beta + gradients;
```
**Velocity Update**:
```
Formula: v_new = β × v_old + gradient

Example:
  β = 0.9
  v_old = [1, 2]
  gradient = [0.5, 0.5]
  
  v_new = 0.9 × [1, 2] + [0.5, 0.5]
        = [0.9, 1.8] + [0.5, 0.5]
        = [1.4, 2.3]
```

**Physical Analogy**:
```
Ball rolling downhill:
  - Previous velocity carries forward (β × v_old)
  - Current slope adds to it (+ gradient)
  - Builds up speed in consistent direction
```

---

```cpp
    // Update parameters: θ = θ - α * v
    return parameters - velocity[param_id] * learning_rate;
}
```
**Parameter Update**: Use velocity instead of raw gradient
```
SGD:      W_new = W - α × gradient
Momentum: W_new = W - α × velocity

Velocity smooths out noisy gradients!
```

---

### RMSprop Implementation

```cpp
Matrix RMSprop::update(const Matrix& parameters, const Matrix& gradients, 
                      const std::string& param_id) {
    // Initialize cache if not exists
    if (cache.find(param_id) == cache.end()) {
        cache[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        cache[param_id].zeros();
    }
```
**Initialization**: Same pattern as Momentum

---

```cpp
    // Update cache: v = β * v + (1 - β) * ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    cache[param_id] = cache[param_id] * beta + grad_squared * (1.0 - beta);
```
**Cache Update**:

1. **Square gradients**: Element-wise multiplication (Hadamard product)
   ```
   gradient = [1, -2, 3]
   grad² = [1, 4, 9]
   ```

2. **Exponential moving average**:
   ```
   Formula: cache = β × cache + (1-β) × gradient²
   
   Example (β=0.9):
     cache = 0.9 × cache_old + 0.1 × gradient²
   
   Interpretation:
     - 90% of history
     + 10% of current
   ```

**Why (1-β)?**
```
Without (1-β):
  cache = β × cache + gradient²
  
  After many steps, cache could explode!

With (1-β):
  cache stays bounded
  Properly normalized exponential average
```

---

```cpp
    // Compute update: θ = θ - α * ∇θ / (√v + ε)
    Matrix denominator = cache[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
```
**Compute Denominator**:
1. Take square root of cache: √v
2. Add epsilon for stability: √v + ε

**Lambda Function**:
```cpp
[this](double x) { return std::sqrt(x) + epsilon; }
  └─ Captures 'this' to access epsilon
     └─ Takes each element x
         └─ Returns √x + ε
```

**Example**:
```
cache = [0.01, 0.04, 0.09]
denominator = [√0.01 + 1e-8, √0.04 + 1e-8, √0.09 + 1e-8]
            = [0.1, 0.2, 0.3]
```

---

```cpp
    Matrix update = gradients.divide(denominator) * learning_rate;
    return parameters - update;
}
```
**Final Update**:
```
Formula: W_new = W - α × (gradient / √cache)

Adaptive per-parameter:
  - Large cache (frequent updates) → divide by large number → small update
  - Small cache (rare updates) → divide by small number → large update
```

---

### Adam Implementation

```cpp
Matrix Adam::update(const Matrix& parameters, const Matrix& gradients, 
                   const std::string& param_id) {
    // Initialize moments if not exists
    if (m.find(param_id) == m.end()) {
        m[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        m[param_id].zeros();
        v[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        v[param_id].zeros();
        t[param_id] = 0;
    }
```
**Initialization**: Three things to track
- m: first moment (momentum)
- v: second moment (variance)
- t: time step (for bias correction)

---

```cpp
    // Increment time step
    t[param_id]++;
    int time_step = t[param_id];
```
**Time Step**: Increments each update
```
First call:  t = 1
Second call: t = 2
Third call:  t = 3
...
```

---

```cpp
    // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * ∇θ
    m[param_id] = m[param_id] * beta1 + gradients * (1.0 - beta1);
```
**First Moment Update** (like Momentum):
```
Example (β₁=0.9):
  m = 0.9 × m_old + 0.1 × gradient
  
  Running average of gradients
```

---

```cpp
    // Update biased second raw moment estimate: v = β₂ * v + (1 - β₂) * ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    v[param_id] = v[param_id] * beta2 + grad_squared * (1.0 - beta2);
```
**Second Moment Update** (like RMSprop):
```
Example (β₂=0.999):
  v = 0.999 × v_old + 0.001 × gradient²
  
  Running average of squared gradients (variance)
```

---

```cpp
    // Compute bias-corrected first moment estimate: m̂ = m / (1 - β₁^t)
    double bias_correction1 = 1.0 - std::pow(beta1, time_step);
    Matrix m_hat = m[param_id] / bias_correction1;
```
**Bias Correction** - THE KEY INNOVATION!

**Why needed?**
```
At t=0: m = 0, v = 0 (initialized to zero)

At t=1:
  m = 0.9×0 + 0.1×grad = 0.1×grad
  
Problem: m is biased toward zero!

Solution: Divide by (1 - β^t)
  t=1: correction = 1 - 0.9¹ = 0.1   →  m_hat = 0.1×grad / 0.1 = grad
  t=2: correction = 1 - 0.9² = 0.19  →  m_hat = ... / 0.19
  t=∞: correction → 1                →  m_hat ≈ m
  
Corrects initialization bias!
```

**Example**:
```
β₁ = 0.9, t = 1:
  bias_correction1 = 1 - 0.9¹ = 0.1
  
β₁ = 0.9, t = 10:
  bias_correction1 = 1 - 0.9¹⁰ ≈ 0.65
  
β₁ = 0.9, t = 100:
  bias_correction1 = 1 - 0.9¹⁰⁰ ≈ 0.9999
```

---

```cpp
    // Compute bias-corrected second raw moment estimate: v̂ = v / (1 - β₂^t)
    double bias_correction2 = 1.0 - std::pow(beta2, time_step);
    Matrix v_hat = v[param_id] / bias_correction2;
```
**Same bias correction** for second moment
```
β₂ = 0.999 (closer to 1)
→ Takes longer to reach 1.0
→ Bias persists longer
```

---

```cpp
    // Compute update: θ = θ - α * m̂ / (√v̂ + ε)
    Matrix denominator = v_hat.apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    Matrix update = m_hat.divide(denominator) * learning_rate;
    return parameters - update;
}
```
**Final Update**:
```
Formula: W = W - α × m_hat / (√v_hat + ε)
              └─ lr    └─ momentum    └─ adaptive scaling

Combines:
  ✓ Momentum (m_hat) - direction
  ✓ RMSprop (v_hat) - per-parameter adaptation
  ✓ Bias correction - accurate early training
```

---

### AdaGrad Implementation

```cpp
Matrix AdaGrad::update(const Matrix& parameters, const Matrix& gradients, 
                      const std::string& param_id) {
    // Initialize accumulated gradients if not exists
    if (accumulated_gradients.find(param_id) == accumulated_gradients.end()) {
        accumulated_gradients[param_id] = Matrix(parameters.getRows(), parameters.getCols());
        accumulated_gradients[param_id].zeros();
    }
```
**Initialization**: Same pattern

---

```cpp
    // Accumulate squared gradients: G = G + ∇θ²
    Matrix grad_squared = gradients.hadamard(gradients);
    accumulated_gradients[param_id] = accumulated_gradients[param_id] + grad_squared;
```
**Accumulation** - KEY DIFFERENCE:
```
RMSprop: cache = β × cache + (1-β) × grad²  (exponential average, forgets)
AdaGrad: G = G + grad²                      (sum, never forgets)

Example over 3 steps:
  t=1: G = 0 + grad₁²
  t=2: G = grad₁² + grad₂²
  t=3: G = grad₁² + grad₂² + grad₃²
  
G keeps growing!
```

---

```cpp
    // Compute update: θ = θ - α * ∇θ / (√G + ε)
    Matrix denominator = accumulated_gradients[param_id].apply([this](double x) {
        return std::sqrt(x) + epsilon;
    });
    
    Matrix update = gradients.divide(denominator) * learning_rate;
    return parameters - update;
}
```
**Update**: Same as RMSprop but with ever-growing G
```
As training progresses:
  G gets larger → √G gets larger → updates get smaller

Eventually: Effective learning rate → 0
```

---

## When to Use Which Optimizer

### Decision Tree

```
What's your situation?
│
├─ Just starting / Simple problem?
│  └─ Use SGD
│     • Simple, easy to understand
│     • Good baseline
│     • Predictable behavior
│
├─ Default choice / Don't know what to use?
│  └─ Use Adam
│     • Works well out-of-the-box
│     • Adaptive + Momentum
│     • Most popular for deep learning
│
├─ Computer vision / Image problems?
│  └─ Try SGD with Momentum
│     • Often outperforms adaptive methods
│     • Better generalization on vision tasks
│     • Requires careful tuning
│
├─ Need adaptive rates but simpler than Adam?
│  └─ Use RMSprop
│     • Simpler than Adam
│     • Good for RNNs
│     • No bias correction needed
│
└─ Sparse data (NLP, embeddings)?
   └─ Use AdaGrad
      • Per-feature learning rates
      • Good for rare features
      • Might stop learning eventually
```

---

### Comparison Table

| Optimizer | Speed | Memory | Pros | Cons | Best For |
|-----------|-------|--------|------|------|----------|
| **SGD** | ⚡⚡⚡ | 💾 | Simple, stable | Slow, requires tuning | Baseline, simple problems |
| **Momentum** | ⚡⚡⚡ | 💾 | Faster than SGD | Extra memory | Computer vision |
| **RMSprop** | ⚡⚡ | 💾💾 | Adaptive rates | No momentum | RNNs, online learning |
| **Adam** | ⚡⚡ | 💾💾💾 | Works well everywhere | Memory cost | Default choice |
| **AdaGrad** | ⚡⚡ | 💾💾 | Good for sparse data | Learning rate decay | NLP, sparse features |

---

### Hyperparameter Recommendations

```cpp
// SGD - Simple and effective
SGD sgd(0.01);  // lr: 0.001 to 0.1, try 0.01 first

// Momentum - Add to SGD
Momentum momentum(0.01, 0.9);  // lr: 0.001 to 0.1, beta: 0.9 to 0.99

// RMSprop - Good defaults
RMSprop rmsprop(0.001, 0.9, 1e-8);  // lr: 0.001 to 0.01

// Adam - Usually just works
Adam adam(0.001, 0.9, 0.999, 1e-8);  // Use defaults!

// AdaGrad - For sparse data
AdaGrad adagrad(0.01, 1e-8);  // lr: 0.01 to 0.1
```

---

## Mathematical Details

### Convergence Analysis

**SGD**:
```
Update: W = W - α∇L

Converges if:
  • Learning rate decreases: α_t = O(1/t)
  • Or small enough: α < 1/L (L = Lipschitz constant)
```

**Momentum**:
```
Convergence rate: O(1/k²) vs O(1/k) for SGD
  → Faster convergence!

Heavy ball method:
  - Overshoots minimum
  - Oscillates back
  - Settles faster
```

**Adam**:
```
Combines:
  • First moment: E[∇L] (where to go)
  • Second moment: E[∇L²] (how fast to go)
  
Convergence: Proven for convex problems
  • Non-convex: Works well in practice
```

---

### Visual Comparison

```
Loss landscape: Think of mountain/valley

SGD:
●
 ╲
  ╲
   ╲
    ● (slow, straight down)

Momentum:
●
 ╲╲
   ╲╲
     ● (faster, builds speed)

Adam:
●
 ╲
  ━━● (adaptive, smart path)

AdaGrad:
●
 ╲
  ─● (slow down over time)
```

---

### Gradient Descent Family Tree

```
Gradient Descent
       │
       ├─ SGD (1951)
       │   └─ Vanilla gradient descent
       │
       ├─ Momentum (1964)
       │   └─ Adds velocity term
       │
       ├─ AdaGrad (2011)
       │   └─ Adaptive per-parameter rates
       │   └─ Problem: learning rate decay
       │
       ├─ RMSprop (2012)
       │   └─ Fixes AdaGrad decay problem
       │   └─ Exponential moving average
       │
       └─ Adam (2014)
           └─ Momentum + RMSprop + Bias correction
           └─ Current state-of-the-art
```

---

## Practical Examples

See [optimizer_example.cpp](../example/optimizer_example.cpp) for:
1. Basic usage of each optimizer
2. Comparison on simple function
3. Visualization of convergence
4. Learning rate scheduling
5. When each optimizer works best

### Quick Example

```cpp
#include "nn/optimizer.h"
#include "nn/matrix.h"

// Create optimizer
Adam optimizer(0.001);  // learning_rate = 0.001

// Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    // Forward pass
    Matrix predictions = model.forward(inputs);
    
    // Compute loss
    double loss = loss_function.calculate(predictions, targets);
    
    // Backward pass - get gradients
    Matrix gradients = loss_function.gradient(predictions, targets);
    
    // Update weights using optimizer
    Matrix new_weights = optimizer.update(
        model.weights,
        gradients,
        "layer1_weights"  // parameter ID
    );
    
    model.weights = new_weights;
}
```

---

## Summary

### Key Takeaways

1. **Optimizers update weights** during training
   ```
   W_new = W_old - optimizer_update
   ```

2. **Different optimizers, different strategies**:
   - SGD: Simple, direct descent
   - Momentum: Smooth out noise
   - RMSprop: Adapt per-parameter
   - Adam: Best of both + bias correction
   - AdaGrad: Good for sparse data

3. **State vs Stateless**:
   - Stateless: SGD (no memory)
   - Stateful: Momentum, RMSprop, Adam (remember past)

4. **When in doubt, use Adam**
   - Works well on most problems
   - Adaptive + momentum
   - Good defaults

5. **Learning rate is critical**
   - Too large: diverges
   - Too small: slow
   - Adam less sensitive than SGD

### Implementation Highlights

- ✅ Polymorphic design (base + derived classes)
- ✅ Per-parameter state tracking (unordered_map)
- ✅ Virtual functions for extensibility
- ✅ Efficient matrix operations
- ✅ Standard defaults that work

---

## Further Reading

1. **Original Papers**:
   - Momentum: Polyak (1964)
   - AdaGrad: Duchi et al. (2011)
   - RMSprop: Hinton's lecture (2012)
   - Adam: Kingma & Ba (2014)

2. **Advanced Topics**:
   - Learning rate scheduling
   - Gradient clipping
   - Weight decay / L2 regularization
   - Nesterov momentum
   - AdamW, RAdam variants

3. **Practical Tips**:
   - Start with Adam
   - If Adam doesn't work, try SGD with momentum
   - Tune learning rate (most important!)
   - Use learning rate warmup for transformers
   - Consider gradient accumulation for large batches
