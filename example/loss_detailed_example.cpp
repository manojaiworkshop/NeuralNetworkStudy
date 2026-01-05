/**
 * =====================================================
 * LOSS FUNCTIONS - COMPLETE GUIDE WITH EXAMPLES
 * =====================================================
 * 
 * This example explains:
 * 1. What are loss functions and WHY we need them
 * 2. How each loss function works mathematically
 * 3. When to use which loss function
 * 4. Step-by-step calculations with real numbers
 * 5. How loss connects to training (backpropagation)
 */

#include "nn/matrix.h"
#include "nn/loss.h"
#include "nn/activation.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

// =====================================================
// PART 1: WHAT ARE LOSS FUNCTIONS?
// =====================================================

void explainLossConcept() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║           WHAT ARE LOSS FUNCTIONS?                    ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "PROBLEM: How do we measure how WRONG our predictions are?\n\n";
    
    cout << "Example: Predicting house prices\n";
    cout << "─────────────────────────────────\n";
    cout << "  True price:      $250,000\n";
    cout << "  Predicted price: $280,000\n";
    cout << "  \n";
    cout << "  How bad is this prediction? → We need a LOSS FUNCTION!\n\n";
    
    cout << "SOLUTION: Loss function measures the ERROR\n";
    cout << "          Loss = f(prediction, true_value)\n";
    cout << "          \n";
    cout << "          • HIGH loss = BAD predictions\n";
    cout << "          • LOW loss = GOOD predictions\n";
    cout << "          • Goal: MINIMIZE loss!\n\n";
    
    cout << "ASCII DIAGRAM - Training Process:\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    cout << "                  FORWARD PASS\n";
    cout << "   Input → [Neural Network] → Prediction\n";
    cout << "     x   →   (W, b, σ)      →     ŷ\n";
    cout << "                                   ↓\n";
    cout << "                            Compare with true value\n";
    cout << "                                   ↓\n";
    cout << "                            LOSS = f(ŷ, y)\n";
    cout << "                                   ↓\n";
    cout << "                            BACKWARD PASS\n";
    cout << "                            (Compute gradients)\n";
    cout << "                                   ↓\n";
    cout << "                            Update weights\n";
    cout << "                            W_new = W_old - α·∇Loss\n\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    
    cout << "KEY CONCEPTS:\n";
    cout << "  1. Loss measures how wrong predictions are\n";
    cout << "  2. During training, we try to MINIMIZE loss\n";
    cout << "  3. Loss gradient tells us HOW to adjust weights\n";
    cout << "  4. Different problems need different loss functions\n\n";
}

// =====================================================
// PART 2: MSE LOSS - DETAILED EXPLANATION
// =====================================================

void demonstrateMSELoss() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║        MEAN SQUARED ERROR (MSE) LOSS                   ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: MSE = (1/n) × Σ(y - ŷ)²\n\n";
    cout << "WHERE:\n";
    cout << "  y = true value (target)\n";
    cout << "  ŷ = predicted value (y-hat)\n";
    cout << "  n = number of samples\n\n";
    
    cout << "WHEN TO USE:\n";
    cout << "  • Regression problems (predicting continuous values)\n";
    cout << "  • Examples: house prices, temperature, stock prices\n";
    cout << "  • Output: Any real number\n\n";
    
    cout << "WHY SQUARED?\n";
    cout << "  • Penalizes large errors more (exponentially)\n";
    cout << "  • Always positive (no negative loss)\n";
    cout << "  • Smooth gradient (easy to optimize)\n\n";
    
    cout << "STEP-BY-STEP CALCULATION:\n";
    cout << "─────────────────────────────────\n\n";
    
    // Create example data
    Matrix predictions(1, 3);
    predictions.set(0, 0, 2.5);  // Predicted values
    predictions.set(0, 1, 3.8);
    predictions.set(0, 2, 1.2);
    
    Matrix targets(1, 3);
    targets.set(0, 0, 2.0);  // True values
    targets.set(0, 1, 4.0);
    targets.set(0, 2, 1.5);
    
    cout << "Predictions (ŷ): [2.5, 3.8, 1.2]\n";
    cout << "Targets (y):     [2.0, 4.0, 1.5]\n\n";
    
    cout << "Step 1: Calculate differences (y - ŷ)\n";
    cout << "  Sample 1: 2.0 - 2.5 = -0.5\n";
    cout << "  Sample 2: 4.0 - 3.8 =  0.2\n";
    cout << "  Sample 3: 1.5 - 1.2 =  0.3\n\n";
    
    cout << "Step 2: Square each difference\n";
    cout << "  Sample 1: (-0.5)² = 0.25\n";
    cout << "  Sample 2: ( 0.2)² = 0.04\n";
    cout << "  Sample 3: ( 0.3)² = 0.09\n\n";
    
    cout << "Step 3: Sum all squared differences\n";
    cout << "  Sum = 0.25 + 0.04 + 0.09 = 0.38\n\n";
    
    cout << "Step 4: Divide by n (number of samples)\n";
    cout << "  MSE = 0.38 / 3 = 0.1267\n\n";
    
    // Calculate using our implementation
    MSELoss mse;
    double loss = mse.calculate(predictions, targets);
    
    cout << "✓ Computed MSE Loss: " << fixed << setprecision(4) << loss << "\n\n";
    
    // Calculate gradient
    cout << "GRADIENT CALCULATION:\n";
    cout << "─────────────────────\n\n";
    cout << "Formula: ∂MSE/∂ŷ = (2/n) × (ŷ - y)\n\n";
    
    Matrix grad = mse.gradient(predictions, targets);
    
    cout << "Gradients:\n";
    cout << "  ∂MSE/∂ŷ₁ = (2/3) × (2.5 - 2.0) = " << grad.get(0, 0) << "\n";
    cout << "  ∂MSE/∂ŷ₂ = (2/3) × (3.8 - 4.0) = " << grad.get(0, 1) << "\n";
    cout << "  ∂MSE/∂ŷ₃ = (2/3) × (1.2 - 1.5) = " << grad.get(0, 2) << "\n\n";
    
    cout << "INTERPRETATION:\n";
    cout << "  • Positive gradient → Prediction too high, decrease it\n";
    cout << "  • Negative gradient → Prediction too low, increase it\n";
    cout << "  • Magnitude → How much to adjust\n\n";
}

// =====================================================
// PART 3: BINARY CROSS-ENTROPY LOSS
// =====================================================

void demonstrateBCELoss() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║      BINARY CROSS-ENTROPY (BCE) LOSS                   ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: BCE = -(1/n) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]\n\n";
    cout << "WHERE:\n";
    cout << "  y ∈ {0, 1}  (true class: 0 or 1)\n";
    cout << "  ŷ ∈ (0, 1)  (predicted probability)\n\n";
    
    cout << "WHEN TO USE:\n";
    cout << "  • Binary classification (2 classes)\n";
    cout << "  • Examples: spam/not spam, cat/dog, yes/no\n";
    cout << "  • Output layer: Sigmoid activation\n\n";
    
    cout << "WHY CROSS-ENTROPY?\n";
    cout << "  • Measures difference between probability distributions\n";
    cout << "  • Penalizes confident wrong predictions heavily\n";
    cout << "  • Works better than MSE for classification\n\n";
    
    cout << "INTUITION:\n";
    cout << "─────────\n";
    cout << "If true class is 1:\n";
    cout << "  ŷ = 0.9 → loss = -log(0.9) = 0.105  (low loss, good!)\n";
    cout << "  ŷ = 0.5 → loss = -log(0.5) = 0.693  (medium loss)\n";
    cout << "  ŷ = 0.1 → loss = -log(0.1) = 2.303  (high loss, bad!)\n\n";
    
    cout << "If true class is 0:\n";
    cout << "  ŷ = 0.1 → loss = -log(0.9) = 0.105  (low loss, good!)\n";
    cout << "  ŷ = 0.5 → loss = -log(0.5) = 0.693  (medium loss)\n";
    cout << "  ŷ = 0.9 → loss = -log(0.1) = 2.303  (high loss, bad!)\n\n";
    
    cout << "EXAMPLE CALCULATION:\n";
    cout << "────────────────────\n\n";
    
    // Example: Spam classifier
    Matrix predictions(3, 1);
    predictions.set(0, 0, 0.9);  // Model says 90% spam
    predictions.set(1, 0, 0.3);  // Model says 30% spam
    predictions.set(2, 0, 0.7);  // Model says 70% spam
    
    Matrix targets(3, 1);
    targets.set(0, 0, 1.0);  // Actually spam
    targets.set(1, 0, 0.0);  // Actually not spam
    targets.set(2, 0, 1.0);  // Actually spam
    
    cout << "Email Classification (Spam = 1, Not Spam = 0):\n\n";
    cout << "Email 1: Predicted = 0.9, True = 1 (spam)\n";
    cout << "  Loss = -[1·log(0.9) + 0·log(0.1)]\n";
    cout << "       = -log(0.9) = 0.105\n\n";
    
    cout << "Email 2: Predicted = 0.3, True = 0 (not spam)\n";
    cout << "  Loss = -[0·log(0.3) + 1·log(0.7)]\n";
    cout << "       = -log(0.7) = 0.357\n\n";
    
    cout << "Email 3: Predicted = 0.7, True = 1 (spam)\n";
    cout << "  Loss = -[1·log(0.7) + 0·log(0.3)]\n";
    cout << "       = -log(0.7) = 0.357\n\n";
    
    cout << "Average Loss = (0.105 + 0.357 + 0.357) / 3 = 0.273\n\n";
    
    BinaryCrossEntropyLoss bce;
    double loss = bce.calculate(predictions, targets);
    
    cout << "✓ Computed BCE Loss: " << fixed << setprecision(4) << loss << "\n\n";
    
    cout << "GRADIENT:\n";
    Matrix grad = bce.gradient(predictions, targets);
    
    cout << "  ∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ)) / n\n\n";
    cout << "Email 1: gradient = " << grad.get(0, 0) << "\n";
    cout << "Email 2: gradient = " << grad.get(1, 0) << "\n";
    cout << "Email 3: gradient = " << grad.get(2, 0) << "\n\n";
}

// =====================================================
// PART 4: CATEGORICAL CROSS-ENTROPY LOSS
// =====================================================

void demonstrateCCELoss() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║    CATEGORICAL CROSS-ENTROPY (CCE) LOSS                ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: CCE = -(1/n) × ΣΣ y_ij · log(ŷ_ij)\n\n";
    cout << "WHERE:\n";
    cout << "  y_ij = 1 if sample i belongs to class j, else 0 (one-hot)\n";
    cout << "  ŷ_ij = predicted probability for sample i, class j\n\n";
    
    cout << "WHEN TO USE:\n";
    cout << "  • Multi-class classification (>2 classes)\n";
    cout << "  • Examples: digit recognition (0-9), image classification\n";
    cout << "  • Output layer: Softmax activation\n\n";
    
    cout << "ONE-HOT ENCODING:\n";
    cout << "─────────────────\n";
    cout << "Class 0 (cat):   [1, 0, 0]\n";
    cout << "Class 1 (dog):   [0, 1, 0]\n";
    cout << "Class 2 (bird):  [0, 0, 1]\n\n";
    
    cout << "EXAMPLE: Image Classification (3 classes)\n";
    cout << "──────────────────────────────────────────\n\n";
    
    // Example: 2 samples, 3 classes
    Matrix predictions(2, 3);
    // Sample 1 predictions (probabilities for cat, dog, bird)
    predictions.set(0, 0, 0.7);  // 70% cat
    predictions.set(0, 1, 0.2);  // 20% dog
    predictions.set(0, 2, 0.1);  // 10% bird
    
    // Sample 2 predictions
    predictions.set(1, 0, 0.1);  // 10% cat
    predictions.set(1, 1, 0.3);  // 30% dog
    predictions.set(1, 2, 0.6);  // 60% bird
    
    Matrix targets(2, 3);
    // Sample 1 is actually a cat
    targets.set(0, 0, 1.0);
    targets.set(0, 1, 0.0);
    targets.set(0, 2, 0.0);
    
    // Sample 2 is actually a bird
    targets.set(1, 0, 0.0);
    targets.set(1, 1, 0.0);
    targets.set(1, 2, 1.0);
    
    cout << "Sample 1:\n";
    cout << "  Predictions: [Cat:0.7, Dog:0.2, Bird:0.1]\n";
    cout << "  True class:  [Cat:1,   Dog:0,   Bird:0]  ← It's a cat!\n";
    cout << "  Loss = -[1·log(0.7) + 0·log(0.2) + 0·log(0.1)]\n";
    cout << "       = -log(0.7) = 0.357\n\n";
    
    cout << "Sample 2:\n";
    cout << "  Predictions: [Cat:0.1, Dog:0.3, Bird:0.6]\n";
    cout << "  True class:  [Cat:0,   Dog:0,   Bird:1]  ← It's a bird!\n";
    cout << "  Loss = -[0·log(0.1) + 0·log(0.3) + 1·log(0.6)]\n";
    cout << "       = -log(0.6) = 0.511\n\n";
    
    cout << "Average Loss = (0.357 + 0.511) / 2 = 0.434\n\n";
    
    CategoricalCrossEntropyLoss cce;
    double loss = cce.calculate(predictions, targets);
    
    cout << "✓ Computed CCE Loss: " << fixed << setprecision(4) << loss << "\n\n";
    
    cout << "KEY INSIGHT:\n";
    cout << "  • Only the predicted probability of the TRUE class matters\n";
    cout << "  • Model should assign high probability to correct class\n";
    cout << "  • Works great with Softmax (outputs probabilities)\n\n";
}

// =====================================================
// PART 5: MAE LOSS
// =====================================================

void demonstrateMAELoss() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║        MEAN ABSOLUTE ERROR (MAE) LOSS                  ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: MAE = (1/n) × Σ|y - ŷ|\n\n";
    
    cout << "WHEN TO USE:\n";
    cout << "  • Regression problems\n";
    cout << "  • When you want to treat all errors equally\n";
    cout << "  • Robust to outliers (unlike MSE)\n\n";
    
    cout << "MSE vs MAE:\n";
    cout << "───────────\n";
    cout << "Small error (0.5):\n";
    cout << "  MSE: 0.5² = 0.25\n";
    cout << "  MAE: |0.5| = 0.50\n\n";
    
    cout << "Large error (5.0):\n";
    cout << "  MSE: 5.0² = 25.0   ← Heavily penalizes large errors!\n";
    cout << "  MAE: |5.0| = 5.0   ← Linear penalty\n\n";
    
    cout << "EXAMPLE CALCULATION:\n";
    cout << "────────────────────\n\n";
    
    Matrix predictions(1, 4);
    predictions.set(0, 0, 100.0);  // Predicted house prices ($1000s)
    predictions.set(0, 1, 250.0);
    predictions.set(0, 2, 180.0);
    predictions.set(0, 3, 320.0);
    
    Matrix targets(1, 4);
    targets.set(0, 0, 95.0);   // True house prices
    targets.set(0, 1, 240.0);
    targets.set(0, 2, 200.0);
    targets.set(0, 3, 310.0);
    
    cout << "House Price Predictions ($1000s):\n\n";
    cout << "House 1: Predicted = $100k, True = $95k\n";
    cout << "  Error = |100 - 95| = 5\n\n";
    
    cout << "House 2: Predicted = $250k, True = $240k\n";
    cout << "  Error = |250 - 240| = 10\n\n";
    
    cout << "House 3: Predicted = $180k, True = $200k\n";
    cout << "  Error = |180 - 200| = 20\n\n";
    
    cout << "House 4: Predicted = $320k, True = $310k\n";
    cout << "  Error = |320 - 310| = 10\n\n";
    
    cout << "MAE = (5 + 10 + 20 + 10) / 4 = 11.25\n\n";
    
    MAELoss mae;
    double loss = mae.calculate(predictions, targets);
    
    cout << "✓ Computed MAE Loss: " << fixed << setprecision(2) << loss << "\n\n";
    
    cout << "Interpretation: On average, predictions are off by $" 
         << (loss * 1000) << "\n\n";
}

// =====================================================
// PART 6: LOSS IN TRAINING LOOP
// =====================================================

void demonstrateTrainingLoop() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║        HOW LOSS IS USED IN TRAINING                    ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "TRAINING PROCESS:\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    cout << "1. FORWARD PASS\n";
    cout << "   ────────────\n";
    cout << "   Input → Network → Prediction\n";
    cout << "   \n";
    cout << "2. COMPUTE LOSS\n";
    cout << "   ────────────\n";
    cout << "   Loss = f(Prediction, Target)\n";
    cout << "   \n";
    cout << "3. BACKWARD PASS\n";
    cout << "   ─────────────\n";
    cout << "   Compute gradients: ∂Loss/∂weights\n";
    cout << "   \n";
    cout << "4. UPDATE WEIGHTS\n";
    cout << "   ──────────────\n";
    cout << "   W_new = W_old - learning_rate × gradient\n";
    cout << "   \n";
    cout << "5. REPEAT\n";
    cout << "   ──────\n";
    cout << "   Go back to step 1 until loss is small enough\n\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    
    // Simulate simple training
    cout << "SIMULATION: Training a Simple Model\n";
    cout << "────────────────────────────────────\n\n";
    
    cout << "Problem: Learn to predict y = 2x + 1\n\n";
    
    // Training data
    Matrix X(3, 1);
    X.set(0, 0, 1.0);
    X.set(1, 0, 2.0);
    X.set(2, 0, 3.0);
    
    Matrix Y(3, 1);
    Y.set(0, 0, 3.0);  // 2*1 + 1 = 3
    Y.set(1, 0, 5.0);  // 2*2 + 1 = 5
    Y.set(2, 0, 7.0);  // 2*3 + 1 = 7
    
    // Initialize random weight and bias
    double w = 0.5;  // Start with wrong values
    double b = 0.0;
    double learning_rate = 0.1;
    
    cout << "Initial: w = " << w << ", b = " << b << "\n\n";
    
    MSELoss mse;
    
    // Training loop
    for (int epoch = 0; epoch < 5; epoch++) {
        // Forward pass
        Matrix predictions(3, 1);
        for (int i = 0; i < 3; i++) {
            predictions.set(i, 0, w * X.get(i, 0) + b);
        }
        
        // Compute loss
        double loss = mse.calculate(predictions, Y);
        
        // Compute gradients (simplified)
        Matrix grad = mse.gradient(predictions, Y);
        
        double grad_w = 0.0;
        double grad_b = 0.0;
        for (int i = 0; i < 3; i++) {
            grad_w += grad.get(i, 0) * X.get(i, 0);
            grad_b += grad.get(i, 0);
        }
        grad_w /= 3;
        grad_b /= 3;
        
        cout << "Epoch " << (epoch + 1) << ":\n";
        cout << "  w = " << fixed << setprecision(3) << w 
             << ", b = " << b << "\n";
        cout << "  Predictions: [" << predictions.get(0, 0) << ", "
             << predictions.get(1, 0) << ", " << predictions.get(2, 0) << "]\n";
        cout << "  Loss = " << loss << "\n";
        
        // Update weights
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;
        
        cout << "  Updated: w = " << w << ", b = " << b << "\n\n";
    }
    
    cout << "Final model: y = " << w << "x + " << b << "\n";
    cout << "Target model: y = 2.0x + 1.0\n\n";
    cout << "✓ Loss decreased → Model learned the pattern!\n\n";
}

// =====================================================
// PART 7: CHOOSING THE RIGHT LOSS
// =====================================================

void lossSelectionGuide() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         HOW TO CHOOSE THE RIGHT LOSS FUNCTION          ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "DECISION TREE:\n";
    cout << "═════════════════════════════════════════════════════════\n\n";
    cout << "What type of problem?\n";
    cout << "  |\n";
    cout << "  ├─ REGRESSION (continuous output)\n";
    cout << "  │   |\n";
    cout << "  │   ├─ Care more about large errors? → MSE\n";
    cout << "  │   │   Examples: predicting exact values\n";
    cout << "  │   │\n";
    cout << "  │   └─ Treat all errors equally? → MAE\n";
    cout << "  │       Examples: robust to outliers\n";
    cout << "  │\n";
    cout << "  └─ CLASSIFICATION (discrete output)\n";
    cout << "      |\n";
    cout << "      ├─ Binary (2 classes) → Binary Cross-Entropy\n";
    cout << "      │   Examples: spam/not spam, yes/no\n";
    cout << "      │   Use with: Sigmoid activation\n";
    cout << "      │\n";
    cout << "      └─ Multi-class (>2 classes) → Categorical Cross-Entropy\n";
    cout << "          Examples: digit recognition, image classification\n";
    cout << "          Use with: Softmax activation\n\n";
    
    cout << "SUMMARY TABLE:\n";
    cout << "═════════════════════════════════════════════════════════\n";
    cout << setw(20) << left << "Problem Type"
         << setw(25) << "Loss Function"
         << "Output Activation\n";
    cout << string(70, '─') << "\n";
    cout << setw(20) << "Regression"
         << setw(25) << "MSE or MAE"
         << "Linear (none)\n";
    cout << setw(20) << "Binary Class."
         << setw(25) << "Binary Cross-Entropy"
         << "Sigmoid\n";
    cout << setw(20) << "Multi-class"
         << setw(25) << "Categorical Cross-Ent."
         << "Softmax\n";
    cout << "═════════════════════════════════════════════════════════\n\n";
}

// =====================================================
// PART 8: PRACTICAL EXAMPLES
// =====================================================

void practicalExamples() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║           PRACTICAL REAL-WORLD EXAMPLES                ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    // Example 1: House Price Prediction
    cout << "EXAMPLE 1: House Price Prediction (Regression)\n";
    cout << "────────────────────────────────────────────────\n\n";
    cout << "Task: Predict house price from features\n";
    cout << "Loss: MSE (penalize large errors heavily)\n\n";
    
    Matrix house_pred(1, 1);
    house_pred.set(0, 0, 285000);
    Matrix house_true(1, 1);
    house_true.set(0, 0, 300000);
    
    MSELoss house_mse;
    double house_loss = house_mse.calculate(house_pred, house_true);
    
    cout << "Predicted: $285,000\n";
    cout << "Actual:    $300,000\n";
    cout << "MSE Loss:  " << house_loss << "\n";
    cout << "RMSE:      $" << sqrt(house_loss) << " (typical error)\n\n";
    
    // Example 2: Email Spam Filter
    cout << "EXAMPLE 2: Email Spam Filter (Binary Classification)\n";
    cout << "─────────────────────────────────────────────────────\n\n";
    cout << "Task: Classify email as spam (1) or not spam (0)\n";
    cout << "Loss: Binary Cross-Entropy\n\n";
    
    Matrix spam_pred(1, 1);
    spam_pred.set(0, 0, 0.95);  // 95% confident it's spam
    Matrix spam_true(1, 1);
    spam_true.set(0, 0, 1.0);   // Actually is spam
    
    BinaryCrossEntropyLoss bce;
    double spam_loss = bce.calculate(spam_pred, spam_true);
    
    cout << "Predicted probability: 0.95 (95% spam)\n";
    cout << "Actual: 1 (spam)\n";
    cout << "BCE Loss: " << spam_loss << " (low = good prediction!)\n\n";
    
    // Example 3: Digit Recognition
    cout << "EXAMPLE 3: Digit Recognition (Multi-class Classification)\n";
    cout << "──────────────────────────────────────────────────────────\n\n";
    cout << "Task: Recognize handwritten digits (0-9)\n";
    cout << "Loss: Categorical Cross-Entropy\n\n";
    
    // Predict probabilities for digit being 0,1,2
    Matrix digit_pred(1, 3);
    digit_pred.set(0, 0, 0.1);  // 10% it's a 0
    digit_pred.set(0, 1, 0.8);  // 80% it's a 1
    digit_pred.set(0, 2, 0.1);  // 10% it's a 2
    
    // True digit is 1
    Matrix digit_true(1, 3);
    digit_true.set(0, 0, 0.0);
    digit_true.set(0, 1, 1.0);  // It's a 1
    digit_true.set(0, 2, 0.0);
    
    CategoricalCrossEntropyLoss cce;
    double digit_loss = cce.calculate(digit_pred, digit_true);
    
    cout << "Predicted: [0:10%, 1:80%, 2:10%]\n";
    cout << "Actual: 1\n";
    cout << "CCE Loss: " << digit_loss << "\n";
    cout << "✓ Model is confident and correct!\n\n";
}

// =====================================================
// MAIN
// =====================================================

int main() {
    cout << "\n";
    cout << "████████████████████████████████████████████████████████████\n";
    cout << "█                                                          █\n";
    cout << "█        LOSS FUNCTIONS - COMPLETE TUTORIAL                █\n";
    cout << "█        Understanding Error Measurement in ML             █\n";
    cout << "█                                                          █\n";
    cout << "████████████████████████████████████████████████████████████\n";
    
    explainLossConcept();
    
    cout << "\nPress Enter to learn about MSE Loss...";
    cin.get();
    demonstrateMSELoss();
    
    cout << "\nPress Enter to learn about Binary Cross-Entropy...";
    cin.get();
    demonstrateBCELoss();
    
    cout << "\nPress Enter to learn about Categorical Cross-Entropy...";
    cin.get();
    demonstrateCCELoss();
    
    cout << "\nPress Enter to learn about MAE Loss...";
    cin.get();
    demonstrateMAELoss();
    
    cout << "\nPress Enter to see how loss is used in training...";
    cin.get();
    demonstrateTrainingLoop();
    
    cout << "\nPress Enter for loss selection guide...";
    cin.get();
    lossSelectionGuide();
    
    cout << "\nPress Enter for practical examples...";
    cin.get();
    practicalExamples();
    
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║                   TUTORIAL COMPLETE!                   ║\n";
    cout << "║                                                        ║\n";
    cout << "║  Key Takeaways:                                        ║\n";
    cout << "║  • Loss measures how wrong predictions are             ║\n";
    cout << "║  • Different problems need different loss functions    ║\n";
    cout << "║  • Loss gradient guides weight updates                 ║\n";
    cout << "║  • Minimizing loss = Better predictions               ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}
