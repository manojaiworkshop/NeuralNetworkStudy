# Loss Functions - Quick Reference

## Quick Comparison Table

| Loss | Formula | Use Case | Output Range | Activation |
|------|---------|----------|--------------|------------|
| **MSE** | `(1/n)Σ(y-ŷ)²` | Regression | [-∞, +∞] | Linear |
| **MAE** | `(1/n)Σ\|y-ŷ\|` | Robust Regression | [-∞, +∞] | Linear |
| **BCE** | `-[y·log(ŷ)+(1-y)·log(1-ŷ)]` | Binary Classification | [0, 1] | Sigmoid |
| **CCE** | `-ΣΣy·log(ŷ)` | Multi-class Classification | [0, 1]ⁿ | Softmax |

---

## When to Use

### Regression Problems
```
Goal: Predict continuous values
Examples: House prices, temperature, stock prices

Use MSE if:
  ✓ Want to penalize large errors heavily
  ✓ Data is relatively clean (few outliers)
  ✓ Care more about big mistakes than small ones
  
Use MAE if:
  ✓ Want to treat all errors equally
  ✓ Data has outliers
  ✓ Want robust median estimation
```

### Binary Classification
```
Goal: Classify into 2 categories
Examples: Spam/Not Spam, Yes/No, Positive/Negative

Use Binary Cross-Entropy:
  ✓ Output layer: Sigmoid activation
  ✓ Outputs probability [0, 1]
  ✓ Penalizes confident wrong predictions
```

### Multi-class Classification
```
Goal: Classify into >2 categories
Examples: Digit recognition (0-9), Image classification

Use Categorical Cross-Entropy:
  ✓ Output layer: Softmax activation
  ✓ Outputs probability distribution
  ✓ Targets: One-hot encoded
```

---

## Quick Code Examples

### MSE Loss
```cpp
#include "nn/loss.h"

MSELoss mse;

// Predictions and targets
Matrix pred(1, 3);
pred.set(0, 0, 2.5); pred.set(0, 1, 3.8); pred.set(0, 2, 1.2);

Matrix target(1, 3);
target.set(0, 0, 2.0); target.set(0, 1, 4.0); target.set(0, 2, 1.5);

// Calculate loss
double loss = mse.calculate(pred, target);
// loss = [(2-2.5)² + (4-3.8)² + (1.5-1.2)²] / 3 = 0.1267

// Get gradient
Matrix grad = mse.gradient(pred, target);
// grad = (2/3) * [(2.5-2), (3.8-4), (1.2-1.5)]
//      = [0.333, -0.133, -0.2]
```

### Binary Cross-Entropy Loss
```cpp
BinaryCrossEntropyLoss bce;

// Email spam classification
Matrix pred(3, 1);
pred.set(0, 0, 0.9);  // 90% spam
pred.set(1, 0, 0.3);  // 30% spam
pred.set(2, 0, 0.7);  // 70% spam

Matrix target(3, 1);
target.set(0, 0, 1.0);  // Is spam
target.set(1, 0, 0.0);  // Not spam
target.set(2, 0, 1.0);  // Is spam

double loss = bce.calculate(pred, target);
// loss = -[1·log(0.9) + 0·log(0.3) + 1·log(0.7)] / 3
//      = 0.273

Matrix grad = bce.gradient(pred, target);
// Tells model how to adjust each prediction
```

### Categorical Cross-Entropy Loss
```cpp
CategoricalCrossEntropyLoss cce;

// Digit recognition (3 classes shown)
Matrix pred(2, 3);  // 2 samples, 3 classes
// Sample 1: 70% cat, 20% dog, 10% bird
pred.set(0, 0, 0.7); pred.set(0, 1, 0.2); pred.set(0, 2, 0.1);
// Sample 2: 10% cat, 30% dog, 60% bird
pred.set(1, 0, 0.1); pred.set(1, 1, 0.3); pred.set(1, 2, 0.6);

Matrix target(2, 3);
// Sample 1 is cat (one-hot encoded)
target.set(0, 0, 1.0); target.set(0, 1, 0.0); target.set(0, 2, 0.0);
// Sample 2 is bird
target.set(1, 0, 0.0); target.set(1, 1, 0.0); target.set(1, 2, 1.0);

double loss = cce.calculate(pred, target);
// loss = -[log(0.7) + log(0.6)] / 2 = 0.434

Matrix grad = cce.gradient(pred, target);
```

### MAE Loss
```cpp
MAELoss mae;

// House price prediction ($1000s)
Matrix pred(1, 3);
pred.set(0, 0, 100); pred.set(0, 1, 250); pred.set(0, 2, 180);

Matrix target(1, 3);
target.set(0, 0, 95); target.set(0, 1, 240); target.set(0, 2, 200);

double loss = mae.calculate(pred, target);
// loss = (|100-95| + |250-240| + |180-200|) / 3
//      = (5 + 10 + 20) / 3 = 11.67

Matrix grad = mae.gradient(pred, target);
// grad = sign(pred - target) / n
//      = [sign(5)/3, sign(10)/3, sign(-20)/3]
//      = [0.333, 0.333, -0.333]
```

---

## Gradient Formulas

### MSE
```
Loss:      MSE = (1/n)Σ(y - ŷ)²
Gradient:  ∂MSE/∂ŷ = (2/n)(ŷ - y)
```

### MAE
```
Loss:      MAE = (1/n)Σ|y - ŷ|
Gradient:  ∂MAE/∂ŷ = (1/n)sign(ŷ - y)
```

### Binary Cross-Entropy
```
Loss:      BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
Gradient:  ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / n
```

### Categorical Cross-Entropy
```
Loss:      CCE = -ΣΣy_ij·log(ŷ_ij)
Gradient:  ∂CCE/∂ŷ_ij = -y_ij/(ŷ_ij·n)
```

---

## Training Loop Integration

```cpp
// Typical training loop
MSELoss loss;
double learning_rate = 0.01;

for (int epoch = 0; epoch < 100; epoch++) {
    // 1. Forward pass
    Matrix predictions = network.forward(inputs);
    
    // 2. Compute loss
    double loss_value = loss.calculate(predictions, targets);
    
    // 3. Backward pass - get gradient
    Matrix loss_grad = loss.gradient(predictions, targets);
    
    // 4. Backpropagate through network
    Matrix weight_grad = network.backward(loss_grad);
    
    // 5. Update weights
    weights = weights - learning_rate * weight_grad;
    
    if (epoch % 10 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss_value << "\n";
    }
}
```

---

## Common Pitfalls

### 1. Wrong Loss for Problem Type
```cpp
// ❌ WRONG: Using MSE for classification
MSELoss mse;
Matrix binary_pred(1, 1);  // 0.8 (80% probability)
Matrix binary_target(1, 1); // 1 (is spam)
double loss = mse.calculate(binary_pred, binary_target);
// This works but BCE is better!

// ✅ CORRECT: Use BCE for binary classification
BinaryCrossEntropyLoss bce;
double loss = bce.calculate(binary_pred, binary_target);
```

### 2. Forgetting to Use Softmax with CCE
```cpp
// ❌ WRONG: Raw network outputs
Matrix raw_outputs = network.forward(input);
CategoricalCrossEntropyLoss cce;
double loss = cce.calculate(raw_outputs, targets);
// raw_outputs might not be valid probabilities!

// ✅ CORRECT: Apply Softmax first
SoftmaxActivation softmax;
Matrix probabilities = softmax.forward(raw_outputs);
double loss = cce.calculate(probabilities, targets);
```

### 3. Shape Mismatches
```cpp
// ❌ WRONG: Mismatched shapes
Matrix pred(2, 3);
Matrix target(3, 2);  // Transposed!
MSELoss mse;
double loss = mse.calculate(pred, target);
// Throws exception!

// ✅ CORRECT: Matching shapes
Matrix pred(2, 3);
Matrix target(2, 3);  // Same shape!
double loss = mse.calculate(pred, target);
```

### 4. Not Clipping BCE/CCE Inputs
```cpp
// ⚠️ DANGEROUS: Extreme predictions
Matrix pred(1, 1);
pred.set(0, 0, 0.0);  // Exactly 0!
BinaryCrossEntropyLoss bce;
double loss = bce.calculate(pred, target);
// log(0) = -∞ → Problem!

// ✅ SAFE: BCE internally clips to [epsilon, 1-epsilon]
// epsilon = 1e-7 by default
// So 0.0 becomes 1e-7, preventing log(0)
```

---

## Loss Visualization

### MSE vs MAE
```
Error Penalty:

MSE (quadratic):        MAE (linear):
  
  25│    ****                25│        *
    │   *    *                 │       *
  20│  *      *             20│      *
    │ *        *               │     *
  15│*          *           15│    *
    │            *             │   *
  10│             *         10│  *
    │                          │ *
   5│              *        5│*
    │               *          │
   0└────────────────        0└──────────
    -5  0  5                  -5  0  5
        Error                     Error
        
Small errors: MAE penalizes more
Large errors: MSE penalizes more
```

### BCE Behavior
```
When true class = 1:

Loss
  5│           *
   │          *
  4│         *
   │        *
  3│       *
   │      *
  2│     *
   │    *
  1│  **
   │**
  0└────────────
   0  0.5  1.0
   Predicted Probability
   
High confidence wrong → High loss
High confidence right → Low loss
```

---

## Performance Tips

1. **Batch Processing**: Process multiple samples at once
   ```cpp
   // Process 32 samples at once (batch size = 32)
   Matrix predictions(32, 10);  // 32 samples, 10 classes
   Matrix targets(32, 10);
   double loss = cce.calculate(predictions, targets);
   ```

2. **Reuse Loss Objects**: Don't create new ones each time
   ```cpp
   // ✅ GOOD: Create once, use many times
   MSELoss mse;
   for (int i = 0; i < epochs; i++) {
       double loss = mse.calculate(pred, target);
   }
   
   // ❌ BAD: Create in loop
   for (int i = 0; i < epochs; i++) {
       MSELoss mse;  // Unnecessary!
       double loss = mse.calculate(pred, target);
   }
   ```

3. **Shape Validation**: Happens automatically
   ```cpp
   // Loss functions validate shapes automatically
   // No need to check yourself!
   try {
       double loss = mse.calculate(pred, target);
   } catch (const std::invalid_argument& e) {
       std::cerr << "Error: " << e.what() << "\n";
   }
   ```

---

## Files to Explore

1. **Header File**: [include/nn/loss.h](../include/nn/loss.h)
   - Loss class declarations
   - Method signatures
   
2. **Implementation**: [src/loss.cpp](../src/loss.cpp)
   - Complete implementations
   - Gradient calculations
   
3. **Complete Guide**: [LOSS_FUNCTIONS_COMPLETE_GUIDE.md](LOSS_FUNCTIONS_COMPLETE_GUIDE.md)
   - Line-by-line explanations
   - Mathematical derivations
   - Detailed examples
   
4. **Example Code**: [example/loss_detailed_example.cpp](../example/loss_detailed_example.cpp)
   - Interactive demonstrations
   - Step-by-step calculations
   - Real-world examples

---

## Build and Run

```bash
# Build the loss example
cmake -S . -B build
cmake --build build --target loss_detailed_example

# Run the interactive example
./build/loss_detailed_example
```

---

## Summary

✅ **MSE** - Regression, penalizes large errors  
✅ **MAE** - Regression, robust to outliers  
✅ **BCE** - Binary classification with Sigmoid  
✅ **CCE** - Multi-class classification with Softmax  

Remember: **Choose loss based on problem type, not preference!**
