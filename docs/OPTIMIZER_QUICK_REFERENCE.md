# Optimizer Quick Reference

## One-Page Cheat Sheet

### Formula Summary

| Optimizer | Update Formula | Key Parameters |
|-----------|---------------|----------------|
| **SGD** | `θ = θ - α·∇θ` | `α` (learning rate) |
| **Momentum** | `v = β·v + ∇θ`<br>`θ = θ - α·v` | `α`, `β` (momentum=0.9) |
| **RMSprop** | `v = β·v + (1-β)·∇θ²`<br>`θ = θ - α·∇θ/√(v+ε)` | `α`, `β` (decay=0.9), `ε` (1e-8) |
| **Adam** | `m = β₁·m + (1-β₁)·∇θ`<br>`v = β₂·v + (1-β₂)·∇θ²`<br>`m̂ = m/(1-β₁ᵗ)`<br>`v̂ = v/(1-β₂ᵗ)`<br>`θ = θ - α·m̂/√(v̂+ε)` | `α`, `β₁` (0.9), `β₂` (0.999), `ε` |
| **AdaGrad** | `G = G + ∇θ²`<br>`θ = θ - α·∇θ/√(G+ε)` | `α`, `ε` (1e-8) |

---

## Quick Start

```cpp
#include "nn/optimizer.h"

// 1. Create optimizer
Adam optimizer(0.001);  // learning_rate = 0.001

// 2. Training loop
for (int epoch = 0; epoch < 100; epoch++) {
    // Get gradients from backward pass
    Matrix gradients = compute_gradients();
    
    // Update parameters
    weights = optimizer.update(weights, gradients, "layer1_weights");
}
```

---

## Recommended Settings

```cpp
// SGD - Simple baseline
SGD sgd(0.01);

// Momentum - Faster than SGD
Momentum momentum(0.01, 0.9);

// RMSprop - Good for RNNs
RMSprop rmsprop(0.001, 0.9, 1e-8);

// Adam - Best default choice
Adam adam(0.001, 0.9, 0.999, 1e-8);

// AdaGrad - For sparse data
AdaGrad adagrad(0.01, 1e-8);
```

---

## When to Use Which?

```
┌─────────────────────────────────────────────────────────────┐
│ PROBLEM TYPE                      │ RECOMMENDED OPTIMIZER   │
├───────────────────────────────────┼─────────────────────────┤
│ Don't know what to use            │ Adam                    │
│ Deep neural networks (general)    │ Adam                    │
│ Computer vision (CNN)             │ SGD + Momentum          │
│ RNN / LSTM / Sequential           │ RMSprop or Adam         │
│ NLP / Sparse features             │ AdaGrad or Adam         │
│ Simple/Convex problem             │ SGD                     │
│ Need fast convergence             │ Adam                    │
│ Want best generalization          │ SGD + Momentum          │
└───────────────────────────────────┴─────────────────────────┘
```

---

## Comparison Chart

| Feature | SGD | Momentum | RMSprop | Adam | AdaGrad |
|---------|-----|----------|---------|------|---------|
| **Speed** | ★☆☆ | ★★☆ | ★★☆ | ★★★ | ★★☆ |
| **Memory** | ★★★ | ★★☆ | ★★☆ | ★☆☆ | ★★☆ |
| **Ease of Use** | ★★★ | ★★★ | ★★☆ | ★★★ | ★★☆ |
| **Robustness** | ★☆☆ | ★★☆ | ★★☆ | ★★★ | ★★☆ |
| **Generalization** | ★★★ | ★★★ | ★★☆ | ★★☆ | ★★☆ |

Legend: ★★★ = Excellent, ★★☆ = Good, ★☆☆ = Fair

---

## Common Issues & Solutions

### Issue: Slow convergence
```cpp
// Try increasing learning rate
optimizer.setLearningRate(0.01);  // was 0.001

// Or switch to faster optimizer
Adam adam(0.001);  // instead of SGD
```

### Issue: Exploding gradients
```cpp
// Reduce learning rate
optimizer.setLearningRate(0.0001);  // was 0.01

// Or use gradient clipping (implement separately)
```

### Issue: Parameters not updating
```cpp
// Check gradients are non-zero
std::cout << "Gradient: " << gradients.get(0, 0) << std::endl;

// Increase learning rate
optimizer.setLearningRate(0.1);

// Reset optimizer state
optimizer.reset();
```

### Issue: Overfitting
```cpp
// Use SGD + Momentum for better generalization
Momentum optimizer(0.001, 0.9);

// Reduce learning rate
optimizer.setLearningRate(0.0001);
```

---

## Visual Guide

### SGD Path
```
     Start (●)
       ↓
       ↓  Straight path
       ↓  down gradient
       ↓
    Minimum (⊙)
```

### Momentum Path
```
     Start (●)
        ↘  Builds up
         ↘  speed
          ↓
    Minimum (⊙)
```

### Adam Path
```
     Start (●)
        ↘  Smart adaptive
       ↙   path finds
      ↓    minimum fast
    Minimum (⊙)
```

---

## Parameter Tracking

```cpp
// Each parameter has independent state
optimizer.update(w1, grad1, "layer1_weights");  // State for layer 1
optimizer.update(w2, grad2, "layer2_weights");  // State for layer 2
optimizer.update(b1, grad_b1, "layer1_bias");   // State for bias

// Reset all states when switching tasks
optimizer.reset();
```

---

## Learning Rate Schedule

```cpp
// Start large, decrease over time
Adam optimizer(0.01);

for (int epoch = 0; epoch < 100; epoch++) {
    // Reduce every 20 epochs
    if (epoch % 20 == 0 && epoch > 0) {
        double new_lr = optimizer.getLearningRate() * 0.5;
        optimizer.setLearningRate(new_lr);
        std::cout << "New LR: " << new_lr << std::endl;
    }
    
    train_one_epoch();
}
```

---

## Advanced Tips

### 1. Warmup (for Adam)
```cpp
// Start with small LR, gradually increase
for (int step = 0; step < 1000; step++) {
    double lr = 0.001 * (step / 1000.0);
    optimizer.setLearningRate(lr);
    train_step();
}
```

### 2. Cosine Annealing
```cpp
// Decrease LR following cosine curve
for (int epoch = 0; epoch < max_epochs; epoch++) {
    double lr = 0.001 * 0.5 * (1 + cos(M_PI * epoch / max_epochs));
    optimizer.setLearningRate(lr);
    train_epoch();
}
```

### 3. Reduce on Plateau
```cpp
// If validation loss doesn't improve, reduce LR
double best_loss = INFINITY;
int patience = 0;

for (int epoch = 0; epoch < 100; epoch++) {
    double val_loss = validate();
    
    if (val_loss < best_loss) {
        best_loss = val_loss;
        patience = 0;
    } else {
        patience++;
        if (patience > 5) {
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.5);
            patience = 0;
        }
    }
}
```

---

## Debug Checklist

- [ ] Gradients are non-zero?
- [ ] Learning rate appropriate? (0.001 - 0.1)
- [ ] Loss decreasing?
- [ ] Parameters changing?
- [ ] Using correct param_id?
- [ ] Reset optimizer between tasks?
- [ ] Tried Adam with defaults?

---

## Example Output

```
Training with Adam (lr=0.001):
Epoch   1: loss = 2.456
Epoch   2: loss = 1.234
Epoch   3: loss = 0.678
Epoch   4: loss = 0.345
...
Epoch  50: loss = 0.012 ✓ Converged!
```

---

## One-Line Summary

| Optimizer | In One Sentence |
|-----------|-----------------|
| **SGD** | Simple gradient descent - subtract gradient times learning rate |
| **Momentum** | SGD with velocity - remembers previous direction to accelerate |
| **RMSprop** | Adaptive per-parameter learning rates using gradient history |
| **Adam** | Momentum + RMSprop + bias correction = best default choice |
| **AdaGrad** | Adapts rates based on all past gradients - good for sparse data |

---

## Further Reading

- **Code**: `example/optimizer_example.cpp`
- **Detailed Guide**: `docs/OPTIMIZER_COMPLETE_GUIDE.md`
- **Implementation**: `src/optimizer.cpp`
- **Header**: `include/nn/optimizer.h`

**Papers**:
- Momentum: Polyak (1964)
- AdaGrad: Duchi et al. (2011)
- RMSprop: Hinton Lecture 6e (2012)
- Adam: Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
