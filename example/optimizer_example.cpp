/**
 * @file optimizer_example.cpp
 * @brief Comprehensive demonstration of all optimizer classes
 * 
 * This example shows:
 * 1. How each optimizer works
 * 2. Comparison of convergence speeds
 * 3. Visualization of optimization paths
 * 4. When to use which optimizer
 * 5. Learning rate effects
 */

#include "../include/nn/optimizer.h"
#include "../include/nn/matrix.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <memory>

// ANSI color codes for pretty printing
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

// Helper function to print section headers
void printHeader(const std::string& title) {
    std::cout << "\n" << BOLD << CYAN << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  " << title << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════" << RESET << "\n\n";
}

void printSubHeader(const std::string& title) {
    std::cout << "\n" << BOLD << YELLOW << "─── " << title << " ───" << RESET << "\n\n";
}

// Simple quadratic function to optimize: f(x) = x^2
// Minimum at x = 0
class QuadraticFunction {
public:
    // Compute function value: f(x) = x^2
    double evaluate(const Matrix& x) const {
        double sum = 0.0;
        for (int i = 0; i < x.getRows(); i++) {
            for (int j = 0; j < x.getCols(); j++) {
                sum += x.get(i, j) * x.get(i, j);
            }
        }
        return sum;
    }
    
    // Compute gradient: ∇f(x) = 2x
    Matrix gradient(const Matrix& x) const {
        return x * 2.0;
    }
};

// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
// Minimum at (1, 1), challenging non-convex function
class RosenbrockFunction {
public:
    double evaluate(const Matrix& x) const {
        if (x.getRows() * x.getCols() < 2) return 0;
        double x_val = x.get(0, 0);
        double y_val = x.get(0, 1);
        return (1 - x_val) * (1 - x_val) + 100 * (y_val - x_val * x_val) * (y_val - x_val * x_val);
    }
    
    Matrix gradient(const Matrix& x) const {
        Matrix grad(x.getRows(), x.getCols());
        if (x.getRows() * x.getCols() < 2) return grad;
        
        double x_val = x.get(0, 0);
        double y_val = x.get(0, 1);
        
        // ∂f/∂x = -2(1-x) - 400x(y-x²)
        double grad_x = -2 * (1 - x_val) - 400 * x_val * (y_val - x_val * x_val);
        // ∂f/∂y = 200(y-x²)
        double grad_y = 200 * (y_val - x_val * x_val);
        
        grad.set(0, 0, grad_x);
        grad.set(0, 1, grad_y);
        return grad;
    }
};

// Visualize optimization path
void visualizePath(const std::vector<double>& values, const std::string& optimizer_name) {
    std::cout << BOLD << optimizer_name << RESET << " convergence:\n";
    
    int width = 50;
    double max_val = values[0];
    for (double v : values) {
        if (v > max_val) max_val = v;
    }
    
    for (size_t i = 0; i < std::min(values.size(), size_t(20)); i++) {
        double normalized = values[i] / max_val;
        int bar_length = static_cast<int>(normalized * width);
        
        std::cout << "Step " << std::setw(2) << i << ": ";
        std::cout << "[";
        for (int j = 0; j < bar_length; j++) std::cout << "█";
        for (int j = bar_length; j < width; j++) std::cout << " ";
        std::cout << "] ";
        
        // Color code based on value
        if (values[i] < 0.01) std::cout << GREEN;
        else if (values[i] < 0.1) std::cout << YELLOW;
        else std::cout << RED;
        
        std::cout << std::fixed << std::setprecision(6) << values[i] << RESET << "\n";
    }
    std::cout << "\n";
}

// Example 1: Basic optimizer usage
void example1_BasicUsage() {
    printHeader("EXAMPLE 1: Basic Optimizer Usage");
    
    std::cout << "Goal: Minimize f(x) = x² starting from x = 10\n";
    std::cout << "Expected minimum: x = 0, f(x) = 0\n\n";
    
    // Create optimizers
    std::vector<std::unique_ptr<Optimizer>> optimizers;
    optimizers.push_back(std::make_unique<SGD>(0.1));
    optimizers.push_back(std::make_unique<Momentum>(0.1, 0.9));
    optimizers.push_back(std::make_unique<RMSprop>(0.1, 0.9, 1e-8));
    optimizers.push_back(std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8));
    optimizers.push_back(std::make_unique<AdaGrad>(0.5, 1e-8));
    
    QuadraticFunction func;
    
    for (auto& optimizer : optimizers) {
        printSubHeader(optimizer->getName());
        
        // Initial parameter
        Matrix x(1, 1);
        x.set(0, 0, 10.0);  // Start at x = 10
        
        std::cout << "Learning rate: " << optimizer->getLearningRate() << "\n\n";
        
        // Optimization steps
        std::cout << std::setw(6) << "Step" << " | " 
                  << std::setw(12) << "x value" << " | " 
                  << std::setw(12) << "f(x)" << " | "
                  << std::setw(12) << "gradient" << "\n";
        std::cout << std::string(55, '-') << "\n";
        
        for (int step = 0; step < 20; step++) {
            double f_val = func.evaluate(x);
            Matrix grad = func.gradient(x);
            
            if (step % 2 == 0) {  // Print every other step
                std::cout << std::setw(6) << step << " | " 
                          << std::setw(12) << std::fixed << std::setprecision(6) << x.get(0, 0) << " | "
                          << std::setw(12) << f_val << " | "
                          << std::setw(12) << grad.get(0, 0) << "\n";
            }
            
            // Update using optimizer
            x = optimizer->update(x, grad, "x");
            
            // Check convergence
            if (f_val < 1e-6) {
                std::cout << GREEN << "\n✓ Converged at step " << step << RESET << "\n";
                break;
            }
        }
        
        std::cout << "\nFinal result: x = " << x.get(0, 0) << ", f(x) = " << func.evaluate(x) << "\n";
        
        // Reset optimizer state for next test
        optimizer->reset();
    }
}

// Example 2: Convergence comparison
void example2_ConvergenceComparison() {
    printHeader("EXAMPLE 2: Convergence Speed Comparison");
    
    std::cout << "Comparing how fast each optimizer converges\n";
    std::cout << "Task: Minimize f(x) = x² starting from x = 10\n\n";
    
    std::vector<std::pair<std::unique_ptr<Optimizer>, std::string>> optimizers;
    optimizers.push_back({std::make_unique<SGD>(0.1), "SGD"});
    optimizers.push_back({std::make_unique<Momentum>(0.1, 0.9), "Momentum"});
    optimizers.push_back({std::make_unique<RMSprop>(0.1, 0.9, 1e-8), "RMSprop"});
    optimizers.push_back({std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), "Adam"});
    optimizers.push_back({std::make_unique<AdaGrad>(0.5, 1e-8), "AdaGrad"});
    
    QuadraticFunction func;
    
    std::cout << std::setw(15) << "Optimizer" << " | " 
              << std::setw(15) << "Steps to 0.01" << " | "
              << std::setw(15) << "Steps to 0.0001" << " | "
              << std::setw(15) << "Final value" << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    for (auto& [optimizer, name] : optimizers) {
        Matrix x(1, 1);
        x.set(0, 0, 10.0);
        
        int steps_to_001 = -1;
        int steps_to_0001 = -1;
        double final_value = 0;
        
        for (int step = 0; step < 100; step++) {
            double f_val = func.evaluate(x);
            
            if (f_val < 0.01 && steps_to_001 == -1) steps_to_001 = step;
            if (f_val < 0.0001 && steps_to_0001 == -1) steps_to_0001 = step;
            
            Matrix grad = func.gradient(x);
            x = optimizer->update(x, grad, "x");
            
            if (step == 99) final_value = func.evaluate(x);
        }
        
        std::cout << std::setw(15) << name << " | " 
                  << std::setw(15) << (steps_to_001 == -1 ? "N/A" : std::to_string(steps_to_001)) << " | "
                  << std::setw(15) << (steps_to_0001 == -1 ? "N/A" : std::to_string(steps_to_0001)) << " | "
                  << std::setw(15) << std::scientific << std::setprecision(3) << final_value << "\n";
        
        optimizer->reset();
    }
    
    std::cout << "\n" << BOLD << "Interpretation:" << RESET << "\n";
    std::cout << "• Lower 'steps to X' = faster convergence\n";
    std::cout << "• Adam typically converges fastest\n";
    std::cout << "• Momentum accelerates SGD significantly\n";
}

// Example 3: Visualization
void example3_VisualizationPaths() {
    printHeader("EXAMPLE 3: Optimization Path Visualization");
    
    std::cout << "Visualizing how each optimizer approaches the minimum\n";
    std::cout << "Function: f(x) = x², starting from x = 10\n\n";
    
    std::vector<std::pair<std::unique_ptr<Optimizer>, std::string>> optimizers;
    optimizers.push_back({std::make_unique<SGD>(0.1), "SGD"});
    optimizers.push_back({std::make_unique<Momentum>(0.1, 0.9), "Momentum"});
    optimizers.push_back({std::make_unique<Adam>(0.1, 0.9, 0.999, 1e-8), "Adam"});
    
    QuadraticFunction func;
    
    for (auto& [optimizer, name] : optimizers) {
        Matrix x(1, 1);
        x.set(0, 0, 10.0);
        
        std::vector<double> values;
        for (int step = 0; step < 20; step++) {
            values.push_back(func.evaluate(x));
            Matrix grad = func.gradient(x);
            x = optimizer->update(x, grad, "x");
        }
        
        visualizePath(values, name);
        optimizer->reset();
    }
    
    std::cout << BOLD << "Observations:" << RESET << "\n";
    std::cout << "• SGD: Steady, predictable decrease\n";
    std::cout << "• Momentum: Faster initial progress, can overshoot\n";
    std::cout << "• Adam: Adaptive, combines best of both\n";
}

// Example 4: Multi-dimensional optimization
void example4_MultiDimensional() {
    printHeader("EXAMPLE 4: 2D Optimization (Rosenbrock Function)");
    
    std::cout << "Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²\n";
    std::cout << "Known minimum: (x,y) = (1, 1), f = 0\n";
    std::cout << "This is a challenging non-convex function!\n\n";
    
    std::vector<std::pair<std::unique_ptr<Optimizer>, std::string>> optimizers;
    optimizers.push_back({std::make_unique<SGD>(0.001), "SGD"});
    optimizers.push_back({std::make_unique<Momentum>(0.001, 0.9), "Momentum"});
    optimizers.push_back({std::make_unique<Adam>(0.01, 0.9, 0.999, 1e-8), "Adam"});
    
    RosenbrockFunction func;
    
    for (auto& [optimizer, name] : optimizers) {
        printSubHeader(name);
        
        Matrix params(1, 2);
        params.set(0, 0, -1.0);  // Start at (-1, -1)
        params.set(0, 1, -1.0);
        
        std::cout << std::setw(6) << "Step" << " | " 
                  << std::setw(10) << "x" << " | " 
                  << std::setw(10) << "y" << " | " 
                  << std::setw(15) << "f(x,y)" << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        for (int step = 0; step < 100; step++) {
            double f_val = func.evaluate(params);
            
            if (step % 10 == 0) {
                std::cout << std::setw(6) << step << " | " 
                          << std::setw(10) << std::fixed << std::setprecision(4) << params.get(0, 0) << " | "
                          << std::setw(10) << params.get(0, 1) << " | "
                          << std::setw(15) << std::scientific << std::setprecision(4) << f_val << "\n";
            }
            
            Matrix grad = func.gradient(params);
            params = optimizer->update(params, grad, "params");
        }
        
        double final_f = func.evaluate(params);
        double dist_to_min = std::sqrt((params.get(0,0)-1)*(params.get(0,0)-1) + 
                                       (params.get(0,1)-1)*(params.get(0,1)-1));
        
        std::cout << "\nFinal: (" << params.get(0, 0) << ", " << params.get(0, 1) << ")\n";
        std::cout << "f(x,y) = " << final_f << "\n";
        std::cout << "Distance to minimum: " << dist_to_min << "\n";
        
        if (final_f < 0.1) {
            std::cout << GREEN << "✓ Successfully minimized!" << RESET << "\n";
        } else {
            std::cout << YELLOW << "⚠ Partial convergence" << RESET << "\n";
        }
        
        optimizer->reset();
    }
}

// Example 5: Learning rate effects
void example5_LearningRateEffects() {
    printHeader("EXAMPLE 5: Learning Rate Impact");
    
    std::cout << "Same optimizer (Adam), different learning rates\n";
    std::cout << "Function: f(x) = x², starting from x = 10\n\n";
    
    std::vector<double> learning_rates = {0.001, 0.01, 0.1, 0.5, 1.0};
    QuadraticFunction func;
    
    std::cout << std::setw(8) << "LR" << " | " 
              << std::setw(15) << "Steps to 0.01" << " | "
              << std::setw(12) << "Final value" << " | "
              << "Status\n";
    std::cout << std::string(60, '=') << "\n";
    
    for (double lr : learning_rates) {
        Adam optimizer(lr);
        
        Matrix x(1, 1);
        x.set(0, 0, 10.0);
        
        int steps_to_convergence = -1;
        double final_value = 0;
        bool diverged = false;
        
        for (int step = 0; step < 50; step++) {
            double f_val = func.evaluate(x);
            
            if (std::isnan(f_val) || std::isinf(f_val) || f_val > 1e10) {
                diverged = true;
                break;
            }
            
            if (f_val < 0.01 && steps_to_convergence == -1) {
                steps_to_convergence = step;
            }
            
            Matrix grad = func.gradient(x);
            x = optimizer.update(x, grad, "x");
            
            if (step == 49) final_value = func.evaluate(x);
        }
        
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << lr << " | ";
        
        if (diverged) {
            std::cout << std::setw(15) << "N/A" << " | "
                      << std::setw(12) << "Diverged" << " | "
                      << RED << "✗ Too large!" << RESET << "\n";
        } else {
            std::cout << std::setw(15) << (steps_to_convergence == -1 ? "N/A" : std::to_string(steps_to_convergence)) << " | "
                      << std::setw(12) << std::scientific << std::setprecision(2) << final_value << " | ";
            
            if (final_value < 0.0001) {
                std::cout << GREEN << "✓ Good" << RESET << "\n";
            } else if (final_value < 0.01) {
                std::cout << YELLOW << "○ Okay" << RESET << "\n";
            } else {
                std::cout << RED << "✗ Too slow" << RESET << "\n";
            }
        }
    }
    
    std::cout << "\n" << BOLD << "Key Insights:" << RESET << "\n";
    std::cout << "• Too small: Slow convergence\n";
    std::cout << "• Too large: Divergence/instability\n";
    std::cout << "• Sweet spot: Fast convergence, stable\n";
    std::cout << "• Adam is more robust to lr than SGD\n";
}

// Example 6: When to use which optimizer
void example6_Recommendations() {
    printHeader("EXAMPLE 6: Optimizer Selection Guide");
    
    std::cout << BOLD << "1. SGD (Stochastic Gradient Descent)" << RESET << "\n";
    std::cout << "   Formula: θ = θ - α·∇θ\n";
    std::cout << "   Use when:\n";
    std::cout << "   • You want simple, interpretable optimization\n";
    std::cout << "   • You have a convex problem\n";
    std::cout << "   • You want baseline comparison\n";
    std::cout << "   Pros: Simple, stable, well-understood\n";
    std::cout << "   Cons: Can be slow, sensitive to learning rate\n\n";
    
    std::cout << BOLD << "2. Momentum" << RESET << "\n";
    std::cout << "   Formula: v = β·v + ∇θ, θ = θ - α·v\n";
    std::cout << "   Use when:\n";
    std::cout << "   • You want faster convergence than SGD\n";
    std::cout << "   • Training computer vision models\n";
    std::cout << "   • Dealing with noisy gradients\n";
    std::cout << "   Pros: Faster than SGD, smooth convergence\n";
    std::cout << "   Cons: Extra hyperparameter (β), can overshoot\n\n";
    
    std::cout << BOLD << "3. RMSprop" << RESET << "\n";
    std::cout << "   Formula: v = β·v + (1-β)·∇θ², θ = θ - α·∇θ/√(v+ε)\n";
    std::cout << "   Use when:\n";
    std::cout << "   • Training RNNs/LSTMs\n";
    std::cout << "   • Online learning scenarios\n";
    std::cout << "   • Different parameters need different rates\n";
    std::cout << "   Pros: Adaptive rates, good for RNNs\n";
    std::cout << "   Cons: More complex than SGD\n\n";
    
    std::cout << BOLD << "4. Adam (Recommended Default)" << RESET << "\n";
    std::cout << "   Formula: Combines momentum + RMSprop + bias correction\n";
    std::cout << "   Use when:\n";
    std::cout << "   • You don't know which optimizer to use\n";
    std::cout << "   • Training deep neural networks\n";
    std::cout << "   • You want good performance out-of-the-box\n";
    std::cout << "   Pros: Works well everywhere, adaptive, momentum\n";
    std::cout << "   Cons: More memory, sometimes generalizes worse\n\n";
    
    std::cout << BOLD << "5. AdaGrad" << RESET << "\n";
    std::cout << "   Formula: G = G + ∇θ², θ = θ - α·∇θ/√(G+ε)\n";
    std::cout << "   Use when:\n";
    std::cout << "   • Working with sparse data (NLP, embeddings)\n";
    std::cout << "   • Different features have vastly different frequencies\n";
    std::cout << "   • Early stage of training\n";
    std::cout << "   Pros: Great for sparse features\n";
    std::cout << "   Cons: Learning rate decay can stop learning\n\n";
    
    std::cout << BOLD << "Quick Decision Tree:" << RESET << "\n";
    std::cout << "├─ Default choice? → Adam\n";
    std::cout << "├─ Computer Vision? → SGD with Momentum\n";
    std::cout << "├─ RNN/LSTM? → RMSprop or Adam\n";
    std::cout << "├─ NLP/Sparse data? → AdaGrad or Adam\n";
    std::cout << "└─ Simple problem? → SGD\n";
}

// Example 7: Parameter-specific optimization
void example7_MultipleParameters() {
    printHeader("EXAMPLE 7: Multiple Parameters");
    
    std::cout << "Optimizing multiple parameters independently\n";
    std::cout << "Each parameter maintains its own optimization state\n\n";
    
    Adam optimizer(0.1);
    QuadraticFunction func;
    
    // Create three different parameters
    Matrix w1(1, 1), w2(1, 1), w3(1, 1);
    w1.set(0, 0, 5.0);
    w2.set(0, 0, 10.0);
    w3.set(0, 0, -8.0);
    
    std::cout << "Initial values: w1=5, w2=10, w3=-8\n";
    std::cout << "Target: All converge to 0\n\n";
    
    std::cout << std::setw(6) << "Step" << " | " 
              << std::setw(10) << "w1" << " | " 
              << std::setw(10) << "w2" << " | " 
              << std::setw(10) << "w3" << "\n";
    std::cout << std::string(45, '-') << "\n";
    
    for (int step = 0; step < 20; step++) {
        // Update each parameter independently
        Matrix grad1 = func.gradient(w1);
        Matrix grad2 = func.gradient(w2);
        Matrix grad3 = func.gradient(w3);
        
        w1 = optimizer.update(w1, grad1, "w1");  // Separate state for w1
        w2 = optimizer.update(w2, grad2, "w2");  // Separate state for w2
        w3 = optimizer.update(w3, grad3, "w3");  // Separate state for w3
        
        if (step % 2 == 0) {
            std::cout << std::setw(6) << step << " | " 
                      << std::setw(10) << std::fixed << std::setprecision(5) << w1.get(0, 0) << " | "
                      << std::setw(10) << w2.get(0, 0) << " | "
                      << std::setw(10) << w3.get(0, 0) << "\n";
        }
    }
    
    std::cout << "\n" << BOLD << "Key Point:" << RESET << " The optimizer tracks separate momentum/variance\n";
    std::cout << "for each parameter using the param_id string!\n";
}

int main() {
    std::cout << BOLD << GREEN << R"(
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           OPTIMIZER DEMONSTRATION - COMPLETE GUIDE                ║
║                                                                   ║
║  Learn how optimizers work and when to use each one!             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
)" << RESET << "\n";
    
    try {
        example1_BasicUsage();
        example2_ConvergenceComparison();
        example3_VisualizationPaths();
        example4_MultiDimensional();
        example5_LearningRateEffects();
        example6_Recommendations();
        example7_MultipleParameters();
        
        printHeader("SUMMARY");
        std::cout << GREEN << "✓" << RESET << " All examples completed successfully!\n\n";
        std::cout << BOLD << "What you learned:" << RESET << "\n";
        std::cout << "1. How each optimizer works mathematically\n";
        std::cout << "2. Convergence speed comparison\n";
        std::cout << "3. Visual optimization paths\n";
        std::cout << "4. Multi-dimensional optimization\n";
        std::cout << "5. Learning rate importance\n";
        std::cout << "6. When to use which optimizer\n";
        std::cout << "7. Multi-parameter optimization\n\n";
        
        std::cout << BOLD << "Recommended Next Steps:" << RESET << "\n";
        std::cout << "• Read OPTIMIZER_COMPLETE_GUIDE.md for detailed explanations\n";
        std::cout << "• Try different learning rates in your own problems\n";
        std::cout << "• Start with Adam, fall back to SGD+Momentum if needed\n";
        std::cout << "• Implement learning rate scheduling\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
    
    return 0;
}
