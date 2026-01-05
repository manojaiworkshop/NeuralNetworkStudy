#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>

/**
 * @brief Matrix class for neural network operations
 * 
 * This class provides matrix operations needed for neural network computations
 * including matrix multiplication, element-wise operations, and transformations.
 */
class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double value);
    Matrix(const std::vector<std::vector<double>>& data);
    
    // Copy constructor and assignment
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    
    // Arithmetic operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;  // Matrix multiplication
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    
    // Compound assignment operators
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);
    
    // Element-wise operations
    Matrix hadamard(const Matrix& other) const;  // Element-wise multiplication
    Matrix divide(const Matrix& other) const;     // Element-wise division
    
    // Matrix operations
    Matrix transpose() const;
    Matrix reshape(size_t new_rows, size_t new_cols) const;
    
    // Statistical operations
    double sum() const;
    double mean() const;
    double max() const;
    double min() const;
    Matrix sumRows() const;  // Sum along rows (returns column vector)
    Matrix sumCols() const;  // Sum along columns (returns row vector)
    
    // Apply function to all elements
    Matrix apply(std::function<double(double)> func) const;
    void applyInPlace(std::function<double(double)> func);
    
    // Initialization methods
    void fill(double value);
    void zeros();
    void ones();
    void randomize(double min, double max);
    void randomNormal(double mean = 0.0, double stddev = 1.0);
    void xavierInit(size_t fan_in, size_t fan_out);
    void heInit(size_t fan_in);
    
    // Getters and setters
    double get(size_t i, size_t j) const;
    void set(size_t i, size_t j, double value);
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    
    // Access operators
    std::vector<double>& operator[](size_t i);
    const std::vector<double>& operator[](size_t i) const;
    
    // Utility functions
    void print() const;
    bool sameShape(const Matrix& other) const;
    
    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);
    static Matrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);
};

// Friend function for scalar * matrix
Matrix operator*(double scalar, const Matrix& matrix);

#endif // MATRIX_H
