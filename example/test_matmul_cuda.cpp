#include "nn/matrix_cuda.h"
#include <iostream>

int main() {
    std::cout << "Testing CUDA Matrix Multiplication" << std::endl;
    
    // Test 1: 6x64 * 64x16 (works)
    std::cout << "\nTest 1: 6x64 * 64x16" << std::endl;
    try {
        MatrixCUDA A1(6, 64, 1.0);
        MatrixCUDA B1(64, 16, 1.0);
        MatrixCUDA C1 = A1.multiplyGPU(B1);
        std::cout << "  SUCCESS: Result is " << C1.getRows() << "x" << C1.getCols() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    
    // Test 2: 7x64 * 64x16 (fails?)
    std::cout << "\nTest 2: 7x64 * 64x16" << std::endl;
    try {
        MatrixCUDA A2(7, 64, 1.0);
        MatrixCUDA B2(64, 16, 1.0);
        MatrixCUDA C2 = A2.multiplyGPU(B2);
        std::cout << "  SUCCESS: Result is " << C2.getRows() << "x" << C2.getCols() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    
    // Test 3: 8x64 * 64x16
    std::cout << "\nTest 3: 8x64 * 64x16" << std::endl;
    try {
        MatrixCUDA A3(8, 64, 1.0);
        MatrixCUDA B3(64, 16, 1.0);
        MatrixCUDA C3 = A3.multiplyGPU(B3);
        std::cout << "  SUCCESS: Result is " << C3.getRows() << "x" << C3.getCols() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  FAILED: " << e.what() << std::endl;
    }
    
    // Test 4: Multiple sequential 7x64 * 64x16
    std::cout << "\nTest 4: Multiple 7x64 * 64x16" << std::endl;
    for (int i = 0; i < 5; i++) {
        try {
            MatrixCUDA A(7, 64, 1.0);
            MatrixCUDA B(64, 16, 1.0);
            MatrixCUDA C = A.multiplyGPU(B);
            std::cout << "  Iteration " << i << ": SUCCESS" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  Iteration " << i << ": FAILED - " << e.what() << std::endl;
        }
    }
    
    return 0;
}
