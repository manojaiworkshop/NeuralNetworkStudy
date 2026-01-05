/**
 * =====================================================
 * ACTIVATION FUNCTIONS - COMPLETE GUIDE WITH EXAMPLES
 * =====================================================
 * 
 * This example explains:
 * 1. What are activation functions and WHY we need them
 * 2. How each activation function works mathematically
 * 3. How matrices flow through activation functions
 * 4. Forward and backward pass (derivatives)
 * 5. Real neural network example with activations
 */

#include "nn/matrix.h"
#include "nn/activation.h"
#include <iostream>
#include <iomanip>
#include <memory>

using namespace std;

// =====================================================
// PART 1: WHAT ARE ACTIVATION FUNCTIONS?
// =====================================================

void explainActivationConcept() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║         WHAT ARE ACTIVATION FUNCTIONS?                ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "PROBLEM: Without activation functions, neural networks are just\n";
    cout << "         LINEAR combinations (matrix multiplications + addition).\n";
    cout << "         No matter how many layers, it's still just: y = W₃(W₂(W₁x))\n";
    cout << "         This can only learn LINEAR relationships!\n\n";
    
    cout << "SOLUTION: Activation functions add NON-LINEARITY!\n";
    cout << "          After each linear operation (matrix multiply), we apply\n";
    cout << "          a non-linear function to enable learning complex patterns.\n\n";
    
    cout << "ASCII DIAGRAM - Neural Network with Activation:\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    cout << "   Input                Hidden Layer              Output\n";
    cout << "   Layer               (with activation)          Layer\n";
    cout << "                                                          \n";
    cout << "    x₁ ──┐                                               \n";
    cout << "         ├──► [W₁·x] ──► σ(z) ──┐                       \n";
    cout << "    x₂ ──┤     LINEAR    NONLIN  ├──► [W₂·h] ──► y    \n";
    cout << "         │     (matrix   (activ.  │     LINEAR          \n";
    cout << "    x₃ ──┘     multiply) function)┘     multiply)       \n";
    cout << "                                                          \n";
    cout << "           z = W₁·x + b₁         h = σ(z)     y = W₂·h + b₂\n\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    
    cout << "KEY CONCEPT:\n";
    cout << "  • Before activation: z = W·x + b  (LINEAR)\n";
    cout << "  • After activation:  a = σ(z)     (NON-LINEAR)\n";
    cout << "  • The 'a' becomes input to next layer\n\n";
    
    cout << "MATRIX VIEW:\n";
    cout << "  If input matrix is [batch_size × features], activation\n";
    cout << "  applies the function to EACH ELEMENT independently!\n\n";
}

// =====================================================
// PART 2: SIGMOID ACTIVATION - DETAILED EXPLANATION
// =====================================================

void demonstrateSigmoid() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║              SIGMOID ACTIVATION FUNCTION               ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: σ(x) = 1 / (1 + e^(-x))\n\n";
    
    cout << "PROPERTIES:\n";
    cout << "  • Output range: (0, 1)\n";
    cout << "  • Smooth S-shaped curve\n";
    cout << "  • Used for: Binary classification (output layer)\n";
    cout << "  • Derivative: σ'(x) = σ(x) × (1 - σ(x))\n\n";
    
    cout << "ASCII GRAPH:\n";
    cout << "  1.0 ┤         ╭─────────\n";
    cout << "      │       ╭─╯         \n";
    cout << "  0.5 ┤     ╭─╯           \n";
    cout << "      │   ╭─╯             \n";
    cout << "  0.0 ┤───╯               \n";
    cout << "      └─────────────────► x\n";
    cout << "       -5  -2  0  2  5\n\n";
    
    // Create example matrix
    cout << "EXAMPLE WITH MATRIX:\n";
    cout << "─────────────────────\n\n";
    
    Matrix input(2, 3);
    input.set(0, 0, -2.0); input.set(0, 1,  0.0); input.set(0, 2,  2.0);
    input.set(1, 0, -5.0); input.set(1, 1,  1.0); input.set(1, 2,  5.0);
    
    cout << "Input Matrix (2×3):\n";
    input.print();
    
    Sigmoid sigmoid;
    Matrix output = sigmoid.forward(input);
    
    cout << "\nAfter Sigmoid σ(x):\n";
    output.print();
    
    cout << "\nELEMENT-WISE CALCULATION:\n";
    cout << "  σ(-2.0) = 1/(1+e^2.0)  = " << output.get(0, 0) << "\n";
    cout << "  σ(0.0)  = 1/(1+e^0)    = " << output.get(0, 1) << "\n";
    cout << "  σ(2.0)  = 1/(1+e^-2.0) = " << output.get(0, 2) << "\n";
    cout << "  σ(-5.0) = 1/(1+e^5.0)  = " << output.get(1, 0) << "\n";
    cout << "  σ(1.0)  = 1/(1+e^-1.0) = " << output.get(1, 1) << "\n";
    cout << "  σ(5.0)  = 1/(1+e^-5.0) = " << output.get(1, 2) << "\n\n";
    
    // Demonstrate backward pass
    cout << "BACKWARD PASS (for learning):\n";
    cout << "─────────────────────────────\n\n";
    
    Matrix grad_output(2, 3);
    grad_output.fill(1.0);  // Assume gradient from next layer is all 1s
    
    cout << "Gradient from next layer (2×3):\n";
    grad_output.print();
    
    Matrix grad_input = sigmoid.backward(input, grad_output);
    
    cout << "\nGradient with respect to input:\n";
    grad_input.print();
    
    cout << "\nDERIVATIVE FORMULA: σ'(x) = σ(x) × (1 - σ(x))\n";
    cout << "Example: For x=0.0, σ(0)=0.5, so σ'(0) = 0.5 × (1-0.5) = 0.25\n\n";
}

// =====================================================
// PART 3: ReLU ACTIVATION - DETAILED EXPLANATION
// =====================================================

void demonstrateReLU() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║                ReLU ACTIVATION FUNCTION                ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: ReLU(x) = max(0, x)\n\n";
    
    cout << "PROPERTIES:\n";
    cout << "  • Output range: [0, ∞)\n";
    cout << "  • Dead simple: negative values become 0, positive stay same\n";
    cout << "  • Used for: Hidden layers (MOST POPULAR!)\n";
    cout << "  • Derivative: 1 if x > 0, else 0\n\n";
    
    cout << "ASCII GRAPH:\n";
    cout << "  5 ┤            ╱\n";
    cout << "  4 ┤          ╱ \n";
    cout << "  3 ┤        ╱   \n";
    cout << "  2 ┤      ╱     \n";
    cout << "  1 ┤    ╱       \n";
    cout << "  0 ┤────╯       \n";
    cout << "    └─────────────► x\n";
    cout << "    -3 -1 0 1 3\n\n";
    
    // Create example matrix
    cout << "EXAMPLE WITH MATRIX:\n";
    cout << "─────────────────────\n\n";
    
    Matrix input(2, 3);
    input.set(0, 0, -2.0); input.set(0, 1,  0.0); input.set(0, 2,  3.0);
    input.set(1, 0, -5.0); input.set(1, 1,  1.5); input.set(1, 2,  4.0);
    
    cout << "Input Matrix (2×3):\n";
    input.print();
    
    ReLU relu;
    Matrix output = relu.forward(input);
    
    cout << "\nAfter ReLU max(0, x):\n";
    output.print();
    
    cout << "\nELEMENT-WISE CALCULATION:\n";
    cout << "  ReLU(-2.0) = max(0, -2.0) = " << output.get(0, 0) << "\n";
    cout << "  ReLU(0.0)  = max(0, 0.0)  = " << output.get(0, 1) << "\n";
    cout << "  ReLU(3.0)  = max(0, 3.0)  = " << output.get(0, 2) << "\n";
    cout << "  ReLU(-5.0) = max(0, -5.0) = " << output.get(1, 0) << "  ← Negative becomes 0\n";
    cout << "  ReLU(1.5)  = max(0, 1.5)  = " << output.get(1, 1) << "  ← Positive stays same\n";
    cout << "  ReLU(4.0)  = max(0, 4.0)  = " << output.get(1, 2) << "\n\n";
    
    // Demonstrate backward pass
    cout << "BACKWARD PASS (for learning):\n";
    cout << "─────────────────────────────\n\n";
    
    Matrix grad_output(2, 3);
    grad_output.fill(1.0);
    
    cout << "Gradient from next layer (2×3):\n";
    grad_output.print();
    
    Matrix grad_input = relu.backward(input, grad_output);
    
    cout << "\nGradient with respect to input:\n";
    grad_input.print();
    
    cout << "\nDERIVATIVE: 1 if x > 0, else 0\n";
    cout << "  • Negative inputs get gradient 0 (no learning)\n";
    cout << "  • Positive inputs get gradient 1 (full gradient passes through)\n\n";
}

// =====================================================
// PART 4: TANH ACTIVATION - DETAILED EXPLANATION
// =====================================================

void demonstrateTanh() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║               TANH ACTIVATION FUNCTION                 ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))\n\n";
    
    cout << "PROPERTIES:\n";
    cout << "  • Output range: (-1, 1)\n";
    cout << "  • Zero-centered (unlike sigmoid)\n";
    cout << "  • Used for: Hidden layers, RNNs\n";
    cout << "  • Derivative: tanh'(x) = 1 - tanh²(x)\n\n";
    
    cout << "ASCII GRAPH:\n";
    cout << "  1.0 ┤        ╭────────\n";
    cout << "      │      ╭─╯        \n";
    cout << "  0.0 ┤────╭─╯──────────\n";
    cout << "      │  ╭─╯            \n";
    cout << " -1.0 ┤──╯              \n";
    cout << "      └─────────────────► x\n";
    cout << "       -3  -1  0  1  3\n\n";
    
    Matrix input(2, 3);
    input.set(0, 0, -2.0); input.set(0, 1,  0.0); input.set(0, 2,  2.0);
    input.set(1, 0, -1.0); input.set(1, 1,  0.5); input.set(1, 2,  1.0);
    
    cout << "Input Matrix (2×3):\n";
    input.print();
    
    Tanh tanh_fn;
    Matrix output = tanh_fn.forward(input);
    
    cout << "\nAfter Tanh:\n";
    output.print();
    
    cout << "\nNOTE: Values are between -1 and 1 (unlike sigmoid's 0 to 1)\n\n";
}

// =====================================================
// PART 5: SOFTMAX - FOR CLASSIFICATION
// =====================================================

void demonstrateSoftmax() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║            SOFTMAX ACTIVATION FUNCTION                 ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORMULA: softmax(x_i) = e^(x_i) / Σ(e^(x_j))\n\n";
    
    cout << "PROPERTIES:\n";
    cout << "  • Output range: (0, 1) and SUM = 1 (probability distribution!)\n";
    cout << "  • Used for: Multi-class classification OUTPUT LAYER\n";
    cout << "  • Converts raw scores to probabilities\n\n";
    
    cout << "EXAMPLE: Classifying an image into 3 classes\n";
    cout << "─────────────────────────────────────────────\n\n";
    
    // Simulating output scores from neural network
    Matrix scores(1, 3);  // 1 sample, 3 classes
    scores.set(0, 0, 2.0);  // Score for class 0 (cat)
    scores.set(0, 1, 1.0);  // Score for class 1 (dog)
    scores.set(0, 2, 0.5);  // Score for class 2 (bird)
    
    cout << "Raw scores from neural network (1×3):\n";
    cout << "  [Cat: 2.0, Dog: 1.0, Bird: 0.5]\n";
    scores.print();
    
    Softmax softmax;
    Matrix probabilities = softmax.forward(scores);
    
    cout << "\nAfter Softmax (probabilities):\n";
    probabilities.print();
    
    cout << "\nCALCULATION:\n";
    cout << "  e^2.0 = " << exp(2.0) << "\n";
    cout << "  e^1.0 = " << exp(1.0) << "\n";
    cout << "  e^0.5 = " << exp(0.5) << "\n";
    cout << "  Sum = " << (exp(2.0) + exp(1.0) + exp(0.5)) << "\n\n";
    cout << "  P(Cat)  = e^2.0 / Sum = " << probabilities.get(0, 0) << "\n";
    cout << "  P(Dog)  = e^1.0 / Sum = " << probabilities.get(0, 1) << "\n";
    cout << "  P(Bird) = e^0.5 / Sum = " << probabilities.get(0, 2) << "\n\n";
    
    double sum = probabilities.get(0, 0) + probabilities.get(0, 1) + probabilities.get(0, 2);
    cout << "  Verification: Sum = " << sum << " (should be 1.0)\n\n";
    
    cout << "INTERPRETATION:\n";
    cout << "  The network is " << (probabilities.get(0, 0) * 100) << "% confident it's a CAT\n";
    cout << "  Prediction: Cat (highest probability)\n\n";
}

// =====================================================
// PART 6: COMPLETE NEURAL NETWORK EXAMPLE
// =====================================================

void completeNeuralNetworkExample() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║        COMPLETE NEURAL NETWORK WITH ACTIVATIONS        ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "NETWORK ARCHITECTURE:\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    cout << "   Input (4 features)\n";
    cout << "        ↓\n";
    cout << "   [Linear: 4 → 6]  ← Matrix multiply with weights W1 (4×6)\n";
    cout << "        ↓\n";
    cout << "   [ReLU activation] ← Apply ReLU element-wise\n";
    cout << "        ↓\n";
    cout << "   Hidden (6 neurons)\n";
    cout << "        ↓\n";
    cout << "   [Linear: 6 → 3]  ← Matrix multiply with weights W2 (6×3)\n";
    cout << "        ↓\n";
    cout << "   [Softmax]         ← Convert to probabilities\n";
    cout << "        ↓\n";
    cout << "   Output (3 classes)\n\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    
    // Simulate a forward pass
    cout << "FORWARD PASS STEP BY STEP:\n";
    cout << "──────────────────────────\n\n";
    
    // Input: 2 samples, 4 features each
    Matrix input(2, 4);
    input.set(0, 0, 0.5); input.set(0, 1, 0.8); input.set(0, 2, 0.3); input.set(0, 3, 0.9);
    input.set(1, 0, 0.2); input.set(1, 1, 0.4); input.set(1, 2, 0.6); input.set(1, 3, 0.1);
    
    cout << "1. INPUT (2 samples × 4 features):\n";
    input.print();
    
    // Layer 1 weights (4 inputs → 6 hidden neurons)
    Matrix W1(4, 6);
    W1.randomize(-0.5, 0.5);
    
    cout << "\n2. WEIGHTS W1 (4×6) - Random initialized:\n";
    W1.print();
    
    // First linear transformation
    Matrix z1 = input * W1;  // Matrix multiplication
    
    cout << "\n3. LINEAR TRANSFORMATION z1 = input × W1 (2×6):\n";
    z1.print();
    cout << "   ↑ These are raw activations (can be any value)\n";
    
    // Apply ReLU activation
    ReLU relu;
    Matrix h1 = relu.forward(z1);
    
    cout << "\n4. AFTER ReLU ACTIVATION h1 = ReLU(z1) (2×6):\n";
    h1.print();
    cout << "   ↑ Negative values became 0, positive values unchanged\n";
    
    // Layer 2 weights (6 hidden → 3 output classes)
    Matrix W2(6, 3);
    W2.randomize(-0.5, 0.5);
    
    cout << "\n5. WEIGHTS W2 (6×3) - Random initialized:\n";
    W2.print();
    
    // Second linear transformation
    Matrix z2 = h1 * W2;
    
    cout << "\n6. LINEAR TRANSFORMATION z2 = h1 × W2 (2×3):\n";
    z2.print();
    cout << "   ↑ Raw scores for 3 classes\n";
    
    // Apply Softmax
    Softmax softmax;
    Matrix output = softmax.forward(z2);
    
    cout << "\n7. AFTER SOFTMAX output = softmax(z2) (2×3):\n";
    output.print();
    cout << "   ↑ Probabilities for each class (each row sums to 1.0)\n\n";
    
    cout << "INTERPRETATION FOR SAMPLE 1:\n";
    cout << "  Class 0 probability: " << (output.get(0, 0) * 100) << "%\n";
    cout << "  Class 1 probability: " << (output.get(0, 1) * 100) << "%\n";
    cout << "  Class 2 probability: " << (output.get(0, 2) * 100) << "%\n\n";
    
    // Find predicted class
    int predicted_class = 0;
    double max_prob = output.get(0, 0);
    for (int i = 1; i < 3; i++) {
        if (output.get(0, i) > max_prob) {
            max_prob = output.get(0, i);
            predicted_class = i;
        }
    }
    
    cout << "  Predicted class: " << predicted_class << " (highest probability)\n\n";
    
    cout << "═══════════════════════════════════════════════════════════\n";
    cout << "KEY TAKEAWAYS:\n";
    cout << "  1. Linear layers do matrix multiplication (W × x)\n";
    cout << "  2. Activation functions add non-linearity ELEMENT-WISE\n";
    cout << "  3. ReLU in hidden layers, Softmax in output layer\n";
    cout << "  4. Each activation operates on EVERY element of the matrix\n";
    cout << "  5. Matrices allow processing multiple samples in PARALLEL\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
}

// =====================================================
// PART 7: HOW BACKWARD PASS WORKS
// =====================================================

void demonstrateBackwardPass() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║           HOW BACKWARD PASS WORKS (BACKPROP)           ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    cout << "FORWARD PASS: Input → Network → Output → Loss\n";
    cout << "BACKWARD PASS: Loss → Gradients → Update Weights\n\n";
    
    cout << "CHAIN RULE VISUALIZATION:\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    cout << "Forward:  x → [W₁·x] → σ(z) → [W₂·h] → Loss\n";
    cout << "           ↑          ↑         ↑        ↑\n";
    cout << "Backward: ∂L/∂x ← ∂L/∂z ← ∂L/∂h ← ∂L/∂y ← ∂L/∂L\n\n";
    cout << "═══════════════════════════════════════════════════════════\n\n";
    
    cout << "EXAMPLE: Single ReLU activation\n";
    cout << "────────────────────────────────\n\n";
    
    Matrix input(1, 3);
    input.set(0, 0, -1.0); input.set(0, 1, 2.0); input.set(0, 2, 0.5);
    
    cout << "Input x:\n";
    input.print();
    
    ReLU relu;
    Matrix output = relu.forward(input);
    
    cout << "\nForward: y = ReLU(x)\n";
    output.print();
    
    // Simulate gradient coming from loss
    Matrix grad_from_loss(1, 3);
    grad_from_loss.set(0, 0, 0.5); grad_from_loss.set(0, 1, 1.0); grad_from_loss.set(0, 2, 0.3);
    
    cout << "\nGradient from loss ∂L/∂y:\n";
    grad_from_loss.print();
    
    Matrix grad_input = relu.backward(input, grad_from_loss);
    
    cout << "\nBackward: ∂L/∂x = ∂L/∂y ⊙ ∂y/∂x\n";
    cout << "         (⊙ means element-wise multiplication)\n";
    grad_input.print();
    
    cout << "\nEXPLANATION:\n";
    cout << "  x[0] = -1.0 → ReLU = 0 → derivative = 0 → gradient blocked!\n";
    cout << "  x[1] =  2.0 → ReLU = 2 → derivative = 1 → gradient = 0.5 × 1 = 0.5\n";
    cout << "  x[2] =  0.5 → ReLU = 0.5 → derivative = 1 → gradient = 1.0 × 1 = 1.0\n\n";
    
    cout << "This gradient ∂L/∂x is then used to update weights in previous layer!\n\n";
}

// =====================================================
// MAIN FUNCTION
// =====================================================

int main() {
    cout << "\n";
    cout << "████████████████████████████████████████████████████████████\n";
    cout << "█                                                          █\n";
    cout << "█     ACTIVATION FUNCTIONS - COMPLETE TUTORIAL             █\n";
    cout << "█     Understanding Neural Network Non-linearity           █\n";
    cout << "█                                                          █\n";
    cout << "████████████████████████████████████████████████████████████\n";
    
    explainActivationConcept();
    
    cout << "\nPress Enter to continue to Sigmoid demonstration...";
    cin.get();
    demonstrateSigmoid();
    
    cout << "\nPress Enter to continue to ReLU demonstration...";
    cin.get();
    demonstrateReLU();
    
    cout << "\nPress Enter to continue to Tanh demonstration...";
    cin.get();
    demonstrateTanh();
    
    cout << "\nPress Enter to continue to Softmax demonstration...";
    cin.get();
    demonstrateSoftmax();
    
    cout << "\nPress Enter to see complete neural network example...";
    cin.get();
    completeNeuralNetworkExample();
    
    cout << "\nPress Enter to understand backward pass...";
    cin.get();
    demonstrateBackwardPass();
    
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════════╗\n";
    cout << "║                   TUTORIAL COMPLETE!                   ║\n";
    cout << "╚════════════════════════════════════════════════════════╝\n\n";
    
    return 0;
}
