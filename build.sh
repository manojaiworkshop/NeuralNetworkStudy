#!/bin/bash

# BUILD SCRIPT FOR C++ NEURAL NETWORK PROJECT
# This script automates the CMake build process

echo "======================================"
echo "Building Neural Network Project"
echo "======================================"
echo ""

# Step 1: Create build directory
echo "Step 1: Creating build directory..."
if [ -d "build" ]; then
    echo "  - Build directory exists, cleaning..."
    rm -rf build
fi
mkdir build
echo "  ✓ Build directory created"
echo ""

# Step 2: Navigate to build directory and run CMake
echo "Step 2: Running CMake configuration..."
cd build
cmake ..
if [ $? -ne 0 ]; then
    echo "  ✗ CMake configuration failed!"
    exit 1
fi
echo "  ✓ CMake configuration successful"
echo ""

# Step 3: Compile the project
echo "Step 3: Compiling the project..."
make
if [ $? -ne 0 ]; then
    echo "  ✗ Compilation failed!"
    exit 1
fi
echo "  ✓ Compilation successful"
echo ""

# Step 4: Show the generated files
echo "Step 4: Build artifacts created:"
ls -lh
echo ""

echo "======================================"
echo "Build Complete!"
echo "======================================"
echo ""
echo "To run the example:"
echo "  ./matrix_example"
echo ""
echo "Or from project root:"
echo "  ./build/matrix_example"
echo ""
