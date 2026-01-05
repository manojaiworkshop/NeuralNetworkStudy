#include "nn/matrix.h"
#include <iomanip>
#include <cmath>
#include <random>

// Default constructor
Matrix::Matrix() : rows(0), cols(0) {}

// Constructor with dimensions
Matrix::Matrix(size_t rows, size_t cols) 
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

// Constructor with dimensions and initial value
Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, value)) {}

// Constructor from 2D vector
Matrix::Matrix(const std::vector<std::vector<double>>& data)
    : data(data), rows(data.size()), cols(data.empty() ? 0 : data[0].size()) {}

// Copy constructor
Matrix::Matrix(const Matrix& other)
    : data(other.data), rows(other.rows), cols(other.cols) {}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }
    return *this;
}

// Matrix addition
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

// Matrix subtraction
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

// Matrix multiplication
Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

// Scalar multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

// Scalar division
Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return (*this) * (1.0 / scalar);
}

// Compound assignment operators
Matrix& Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    *this = *this * scalar;
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    *this = *this / scalar;
    return *this;
}

// Hadamard product (element-wise multiplication)
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for Hadamard product");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

// Element-wise division
Matrix Matrix::divide(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise division");
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (other.data[i][j] == 0.0) {
                throw std::invalid_argument("Division by zero in element-wise division");
            }
            result.data[i][j] = data[i][j] / other.data[i][j];
        }
    }
    return result;
}

// Transpose
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

// Reshape
Matrix Matrix::reshape(size_t new_rows, size_t new_cols) const {
    if (rows * cols != new_rows * new_cols) {
        throw std::invalid_argument("New dimensions must have same total elements");
    }
    
    Matrix result(new_rows, new_cols);
    size_t idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t new_i = idx / new_cols;
            size_t new_j = idx % new_cols;
            result.data[new_i][new_j] = data[i][j];
            ++idx;
        }
    }
    return result;
}

// Sum of all elements
double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            total += data[i][j];
        }
    }
    return total;
}

// Mean of all elements
double Matrix::mean() const {
    return sum() / (rows * cols);
}

// Maximum element
double Matrix::max() const {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot find max of empty matrix");
    }
    
    double max_val = data[0][0];
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (data[i][j] > max_val) {
                max_val = data[i][j];
            }
        }
    }
    return max_val;
}

// Minimum element
double Matrix::min() const {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot find min of empty matrix");
    }
    
    double min_val = data[0][0];
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (data[i][j] < min_val) {
                min_val = data[i][j];
            }
        }
    }
    return min_val;
}

// Sum along rows (returns column vector)
Matrix Matrix::sumRows() const {
    Matrix result(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            sum += data[i][j];
        }
        result.data[i][0] = sum;
    }
    return result;
}

// Sum along columns (returns row vector)
Matrix Matrix::sumCols() const {
    Matrix result(1, cols);
    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            sum += data[i][j];
        }
        result.data[0][j] = sum;
    }
    return result;
}

// Apply function to all elements
Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = func(data[i][j]);
        }
    }
    return result;
}

// Apply function in place
void Matrix::applyInPlace(std::function<double(double)> func) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = func(data[i][j]);
        }
    }
}

// Fill with value
void Matrix::fill(double value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = value;
        }
    }
}

// Fill with zeros
void Matrix::zeros() {
    fill(0.0);
}

// Fill with ones
void Matrix::ones() {
    fill(1.0);
}

// Randomize with uniform distribution
void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// Randomize with normal distribution
void Matrix::randomNormal(double mean, double stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(mean, stddev);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

// Xavier/Glorot initialization
void Matrix::xavierInit(size_t fan_in, size_t fan_out) {
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    randomize(-limit, limit);
}

// He initialization
void Matrix::heInit(size_t fan_in) {
    double stddev = std::sqrt(2.0 / fan_in);
    randomNormal(0.0, stddev);
}

// Get element
double Matrix::get(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data[i][j];
}

// Set element
void Matrix::set(size_t i, size_t j, double value) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    data[i][j] = value;
}

// Access operators
std::vector<double>& Matrix::operator[](size_t i) {
    if (i >= rows) {
        throw std::out_of_range("Matrix row index out of bounds");
    }
    return data[i];
}

const std::vector<double>& Matrix::operator[](size_t i) const {
    if (i >= rows) {
        throw std::out_of_range("Matrix row index out of bounds");
    }
    return data[i];
}

// Print matrix
void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Matrix (" << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << data[i][j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

// Check if matrices have same shape
bool Matrix::sameShape(const Matrix& other) const {
    return (rows == other.rows && cols == other.cols);
}

// Static factory methods
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result.data[i][i] = 1.0;
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    Matrix result(rows, cols);
    result.randomize(min, max);
    return result;
}

// Friend function for scalar * matrix
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}
