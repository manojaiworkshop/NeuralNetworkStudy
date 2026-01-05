/**
 * COMPREHENSIVE MATRIX EXAMPLE FOR NEURAL NETWORKS
 * 
 * This example demonstrates:
 * 1. How matrices are stored in memory (Stack vs Heap)
 * 2. Matrix addition and other operations
 * 3. How matrices are used in neural networks
 * 4. Memory management in C++
 */

#include "nn/matrix.h"
#include <iostream>
#include <iomanip>

using namespace std;

// Function to demonstrate stack vs heap allocation
void demonstrateMemoryAllocation() {
    cout << "\n========================================\n";
    cout << "PART 1: MEMORY ALLOCATION (Stack vs Heap)\n";
    cout << "========================================\n\n";
    
    // STACK ALLOCATION
    // - Fast allocation
    // - Automatically deallocated when function ends
    // - Limited size (usually a few MB)
    // - Memory address is in stack region
    cout << "1. STACK ALLOCATION:\n";
    cout << "   Matrix stackMatrix(2, 2);  // Created on stack\n\n";
    
    Matrix stackMatrix(2, 2);  // This object itself is on stack
    stackMatrix.set(0, 0, 1.0);
    stackMatrix.set(0, 1, 2.0);
    stackMatrix.set(1, 0, 3.0);
    stackMatrix.set(1, 1, 4.0);
    
    cout << "   Stack Matrix address: " << &stackMatrix << "\n";
    cout << "   Size of Matrix object: " << sizeof(stackMatrix) << " bytes\n";
    cout << "   (This is just the object structure, not the actual data!)\n\n";
    
    // HEAP ALLOCATION (using new)
    // - Slower allocation
    // - Must manually deallocate (using delete)
    // - Large size available (limited by RAM)
    // - Memory address is in heap region
    cout << "2. HEAP ALLOCATION:\n";
    cout << "   Matrix* heapMatrix = new Matrix(2, 2);  // Created on heap\n\n";
    
    Matrix* heapMatrix = new Matrix(2, 2);  // Pointer on stack, object on heap
    heapMatrix->set(0, 0, 5.0);
    heapMatrix->set(0, 1, 6.0);
    heapMatrix->set(1, 0, 7.0);
    heapMatrix->set(1, 1, 8.0);
    
    cout << "   Heap Matrix address: " << heapMatrix << "\n";
    cout << "   Size of pointer: " << sizeof(heapMatrix) << " bytes\n\n";
    
    // IMPORTANT: The actual matrix data (numbers) are stored in HEAP
    // even for stackMatrix, because Matrix uses std::vector internally!
    cout << "3. WHERE IS THE ACTUAL DATA?\n";
    cout << "   Both matrices use std::vector<std::vector<double>> internally.\n";
    cout << "   std::vector ALWAYS allocates its data on the HEAP!\n";
    cout << "   So the actual numbers are on heap in both cases.\n";
    cout << "   Only the Matrix object wrapper differs (stack vs heap).\n\n";
    
    // Clean up heap memory
    delete heapMatrix;  // MUST do this for heap-allocated objects!
    cout << "   delete heapMatrix;  // Freeing heap memory\n\n";
    
    // stackMatrix is automatically cleaned up when function ends
}

// Function to demonstrate matrix addition internals
void demonstrateMatrixAddition() {
    cout << "\n========================================\n";
    cout << "PART 2: MATRIX ADDITION - HOW IT WORKS\n";
    cout << "========================================\n\n";
    
    // Create two 2x3 matrices
    cout << "Creating Matrix A (2x3):\n";
    Matrix A(2, 3);
    A.set(0, 0, 1.0); A.set(0, 1, 2.0); A.set(0, 2, 3.0);
    A.set(1, 0, 4.0); A.set(1, 1, 5.0); A.set(1, 2, 6.0);
    A.print();
    
    cout << "\nCreating Matrix B (2x3):\n";
    Matrix B(2, 3);
    B.set(0, 0, 10.0); B.set(0, 1, 20.0); B.set(0, 2, 30.0);
    B.set(1, 0, 40.0); B.set(1, 1, 50.0); B.set(1, 2, 60.0);
    B.print();
    
    cout << "\n--- HOW ADDITION WORKS INTERNALLY ---\n";
    cout << "When we do: Matrix C = A + B;\n";
    cout << "Steps:\n";
    cout << "1. Check if dimensions match (2x3 == 2x3) ✓\n";
    cout << "2. Create new result matrix C(2, 3) on HEAP\n";
    cout << "3. Loop through each element:\n";
    cout << "   For i = 0 to 1 (rows):\n";
    cout << "     For j = 0 to 2 (cols):\n";
    cout << "       C[i][j] = A[i][j] + B[i][j]\n\n";
    
    cout << "Example calculations:\n";
    cout << "  C[0][0] = A[0][0] + B[0][0] = " << A.get(0,0) << " + " << B.get(0,0) << " = " << (A.get(0,0) + B.get(0,0)) << "\n";
    cout << "  C[0][1] = A[0][1] + B[0][1] = " << A.get(0,1) << " + " << B.get(0,1) << " = " << (A.get(0,1) + B.get(0,1)) << "\n";
    cout << "  C[1][2] = A[1][2] + B[1][2] = " << A.get(1,2) << " + " << B.get(1,2) << " = " << (A.get(1,2) + B.get(1,2)) << "\n\n";
    
    // Perform the addition
    Matrix C = A + B;
    
    cout << "Result Matrix C = A + B:\n";
    C.print();
    
    cout << "\n--- MEMORY LAYOUT ---\n";
    cout << "In memory, matrix data is stored as:\n";
    cout << "Matrix A: [1.0][2.0][3.0][4.0][5.0][6.0]  <- Contiguous in vector\n";
    cout << "          Row 0 ---------  Row 1 --------\n";
    cout << "This is a vector of vectors (2D array).\n";
    cout << "Each row is a separate vector on the heap.\n\n";
}

// Function to demonstrate neural network operations
void demonstrateNeuralNetworkOperations() {
    cout << "\n========================================\n";
    cout << "PART 3: NEURAL NETWORK OPERATIONS\n";
    cout << "========================================\n\n";
    
    cout << "In a neural network, matrices represent:\n";
    cout << "- Weights: connections between layers\n";
    cout << "- Inputs: data fed into the network\n";
    cout << "- Activations: outputs from each layer\n\n";
    
    // Simple neural network example: 2 inputs -> 3 hidden -> 2 outputs
    
    // Input layer: 1 sample with 2 features (1x2 matrix)
    cout << "1. INPUT LAYER (1x2):\n";
    cout << "   One data sample with 2 features\n";
    Matrix input(1, 2);
    input.set(0, 0, 0.5);  // Feature 1
    input.set(0, 1, 0.8);  // Feature 2
    input.print();
    
    // Weights from input to hidden layer (2x3 matrix)
    cout << "\n2. WEIGHTS: Input -> Hidden (2x3):\n";
    cout << "   2 inputs connected to 3 hidden neurons\n";
    Matrix weights1(2, 3);
    weights1.set(0, 0, 0.1); weights1.set(0, 1, 0.2); weights1.set(0, 2, 0.3);
    weights1.set(1, 0, 0.4); weights1.set(1, 1, 0.5); weights1.set(1, 2, 0.6);
    weights1.print();
    
    // Matrix multiplication: input * weights1
    cout << "\n3. MATRIX MULTIPLICATION: Input × Weights1\n";
    cout << "   (1x2) × (2x3) = (1x3)\n";
    cout << "   Formula: result[i][j] = sum(input[i][k] × weights1[k][j])\n\n";
    
    Matrix hidden = input * weights1;
    cout << "   Hidden Layer (before activation):\n";
    hidden.print();
    
    cout << "\n   Calculation details:\n";
    cout << "   hidden[0][0] = (0.5×0.1) + (0.8×0.4) = " << hidden.get(0, 0) << "\n";
    cout << "   hidden[0][1] = (0.5×0.2) + (0.8×0.5) = " << hidden.get(0, 1) << "\n";
    cout << "   hidden[0][2] = (0.5×0.3) + (0.8×0.6) = " << hidden.get(0, 2) << "\n";
    
    // Bias addition
    cout << "\n4. BIAS ADDITION:\n";
    cout << "   Biases help shift activation function\n";
    Matrix bias1(1, 3);
    bias1.set(0, 0, 0.1); bias1.set(0, 1, 0.2); bias1.set(0, 2, 0.3);
    cout << "   Bias:\n";
    bias1.print();
    
    hidden = hidden + bias1;
    cout << "\n   Hidden Layer (after adding bias):\n";
    hidden.print();
    
    // Element-wise operations (Hadamard product)
    cout << "\n5. ELEMENT-WISE OPERATIONS (Hadamard Product):\n";
    cout << "   Used for gradient calculations in backpropagation\n";
    Matrix grad1(1, 3);
    grad1.set(0, 0, 0.5); grad1.set(0, 1, 0.6); grad1.set(0, 2, 0.7);
    Matrix grad2(1, 3);
    grad2.set(0, 0, 2.0); grad2.set(0, 1, 3.0); grad2.set(0, 2, 4.0);
    
    cout << "   Gradient 1:\n";
    grad1.print();
    cout << "\n   Gradient 2:\n";
    grad2.print();
    
    Matrix hadamard_result = grad1.hadamard(grad2);
    cout << "\n   Hadamard Product (element-wise multiply):\n";
    hadamard_result.print();
    
    cout << "   Calculation: [0.5×2.0, 0.6×3.0, 0.7×4.0] = [" 
         << hadamard_result.get(0,0) << ", " 
         << hadamard_result.get(0,1) << ", " 
         << hadamard_result.get(0,2) << "]\n";
}

// Function to demonstrate variable storage and manipulation
void demonstrateVariableStorage() {
    cout << "\n========================================\n";
    cout << "PART 4: VARIABLE STORAGE & MANIPULATION\n";
    cout << "========================================\n\n";
    
    // Value types vs reference types
    cout << "1. VALUE SEMANTICS (Copy by value):\n";
    Matrix original(2, 2);
    original.set(0, 0, 1.0); original.set(0, 1, 2.0);
    original.set(1, 0, 3.0); original.set(1, 1, 4.0);
    
    cout << "   Original matrix:\n";
    original.print();
    
    Matrix copy = original;  // Copy constructor called
    copy.set(0, 0, 999.0);   // Modify the copy
    
    cout << "\n   After modifying copy to 999:\n";
    cout << "   Copy:\n";
    copy.print();
    cout << "   Original (unchanged):\n";
    original.print();
    cout << "   ✓ Original is NOT affected (deep copy)\n\n";
    
    // Reference semantics
    cout << "2. REFERENCE SEMANTICS (Alias):\n";
    Matrix& reference = original;  // reference is an alias to original
    reference.set(0, 0, 777.0);
    
    cout << "   After modifying reference to 777:\n";
    cout << "   Original (changed!):\n";
    original.print();
    cout << "   ✓ Original IS affected (same memory)\n\n";
    
    // Pointer semantics
    cout << "3. POINTER SEMANTICS:\n";
    Matrix* pointer = &original;  // pointer stores the address
    pointer->set(0, 0, 555.0);
    
    cout << "   After modifying via pointer to 555:\n";
    cout << "   Original (changed!):\n";
    original.print();
    cout << "   ✓ Original IS affected (pointer points to same memory)\n\n";
    
    // Temporary objects
    cout << "4. TEMPORARY OBJECTS:\n";
    cout << "   When you do: Matrix result = A + B;\n";
    cout << "   - operator+ creates a temporary Matrix\n";
    cout << "   - This temporary is moved into 'result'\n";
    cout << "   - C++ optimizes this (Return Value Optimization)\n";
    cout << "   - No unnecessary copies in modern C++\n\n";
}

// Main function
int main() {
    cout << "\n";
    cout << "╔════════════════════════════════════════════════════╗\n";
    cout << "║  NEURAL NETWORK MATRIX OPERATIONS IN C++          ║\n";
    cout << "║  A Comprehensive Guide for Beginners              ║\n";
    cout << "╚════════════════════════════════════════════════════╝\n";
    
    try {
        // Part 1: Memory allocation
        demonstrateMemoryAllocation();
        
        // Part 2: Matrix addition
        demonstrateMatrixAddition();
        
        // Part 3: Neural network operations
        demonstrateNeuralNetworkOperations();
        
        // Part 4: Variable storage
        demonstrateVariableStorage();
        
        cout << "\n========================================\n";
        cout << "SUMMARY\n";
        cout << "========================================\n";
        cout << "✓ Stack: Fast, auto-cleanup, limited size\n";
        cout << "✓ Heap: Slower, manual cleanup, large size\n";
        cout << "✓ Matrix data is stored in heap (std::vector)\n";
        cout << "✓ Matrix addition: element-wise operation\n";
        cout << "✓ Matrix multiplication: for neural network layers\n";
        cout << "✓ Hadamard product: for gradient calculations\n";
        cout << "✓ Copy by value: safe but uses more memory\n";
        cout << "✓ References/pointers: efficient but must be careful\n";
        cout << "\nProgram completed successfully!\n\n";
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
