# CMake minimum version required
cmake_minimum_required(VERSION 3.10)

# Set CUDA toolkit path to use CUDA 12.9 (compatible with GCC 11)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

# Project name and version - now with CUDA support!
project(NeuralNetworkFromScratch VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (75 for Quadro RTX 5000 - Turing)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# CUDA compiler flags - fix compatibility with GCC 11
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -std=c++17")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")  # Enable CUDA debugging
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")  # Optimization

# Create a library from matrix source files (CPU)
add_library(matrix_lib STATIC
    src/matrix.cpp
)

# Create a CUDA library from matrix CUDA source files (GPU)
add_library(matrix_cuda_lib STATIC
    src/matrix_cuda.cu
)

# Set CUDA properties
set_target_properties(matrix_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create the matrix example executable (CPU only)
add_executable(matrix_example
    example/matrix_example.cpp
)

# Link the library to the executable
target_link_libraries(matrix_example matrix_lib)

# Create the CUDA matrix example executable (CPU + GPU)
add_executable(matrix_cuda_example
    example/matrix_cuda_example.cpp
)

# Link both libraries to CUDA example
target_link_libraries(matrix_cuda_example 
    matrix_lib 
    matrix_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(matrix_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create activation library
add_library(activation_lib STATIC
    src/activation.cpp
)

# Link activation library with matrix library
target_link_libraries(activation_lib matrix_lib)

# Create activation functions example executable
add_executable(activation_example
    example/activation_detailed_example.cpp
)

# Link the libraries to the activation example
target_link_libraries(activation_example 
    matrix_lib
    activation_lib
)

# Create CUDA activation library
add_library(activation_cuda_lib STATIC
    src/activation_cuda.cu
)

# Set CUDA properties for activation library
set_target_properties(activation_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link activation CUDA library with other libraries
target_link_libraries(activation_cuda_lib 
    matrix_lib
    matrix_cuda_lib
    activation_lib
)

# Create CUDA activation functions example executable
add_executable(activation_cuda_example
    example/activation_cuda_example.cpp
)

# Link all libraries to the CUDA activation example
target_link_libraries(activation_cuda_example 
    matrix_lib
    matrix_cuda_lib
    activation_lib
    activation_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(activation_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create loss library
add_library(loss_lib STATIC
    src/loss.cpp
)

# Link loss library with matrix library
target_link_libraries(loss_lib matrix_lib)

# Create loss functions detailed example executable
add_executable(loss_detailed_example
    example/loss_detailed_example.cpp
)

# Link the libraries to the loss example
target_link_libraries(loss_detailed_example 
    matrix_lib
    activation_lib
    loss_lib
)

# Create CUDA loss library
add_library(loss_cuda_lib STATIC
    src/loss_cuda.cu
)

# Set CUDA properties for loss library
set_target_properties(loss_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link loss CUDA library with other libraries
target_link_libraries(loss_cuda_lib 
    matrix_lib
    matrix_cuda_lib
    loss_lib
)

# Create CUDA loss functions example executable
add_executable(loss_cuda_example
    example/loss_cuda_example.cpp
)

# Link all libraries to the CUDA loss example
target_link_libraries(loss_cuda_example 
    matrix_lib
    matrix_cuda_lib
    loss_lib
    loss_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(loss_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create optimizer library
add_library(optimizer_lib STATIC
    src/optimizer.cpp
)

# Link optimizer library with matrix library
target_link_libraries(optimizer_lib matrix_lib)

# Create optimizer example executable
add_executable(optimizer_example
    example/optimizer_example.cpp
)

# Link the libraries to the optimizer example
target_link_libraries(optimizer_example 
    matrix_lib
    optimizer_lib
)

# Create CUDA optimizer library
add_library(optimizer_cuda_lib STATIC
    src/optimizer_cuda.cu
)

# Set CUDA properties for optimizer library
set_target_properties(optimizer_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link optimizer CUDA library with other libraries
target_link_libraries(optimizer_cuda_lib 
    matrix_lib
    matrix_cuda_lib
    optimizer_lib
)

# Create CUDA optimizer example executable
add_executable(optimizer_cuda_example
    example/optimizer_cuda_example.cpp
)

# Link all libraries to the CUDA optimizer example
target_link_libraries(optimizer_cuda_example 
    matrix_lib
    matrix_cuda_lib
    optimizer_lib
    optimizer_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(optimizer_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create layer library
add_library(layer_lib STATIC
    src/layer.cpp
)

# Link layer library with required libraries
target_link_libraries(layer_lib 
    matrix_lib
    activation_lib
)

# Create layer example executable
add_executable(layer_example
    example/layer_example.cpp
)

# Link the libraries to the layer example
target_link_libraries(layer_example 
    matrix_lib
    activation_lib
    loss_lib
    layer_lib
)

# Create CUDA layer library
add_library(layer_cuda_lib STATIC
    src/layer_cuda.cu
)

# Set CUDA properties for layer library
set_target_properties(layer_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link layer CUDA library with other libraries
target_link_libraries(layer_cuda_lib 
    matrix_lib
    matrix_cuda_lib
    activation_lib
    activation_cuda_lib
    layer_lib
)

# Create CUDA layer example executable
add_executable(layer_cuda_example
    example/layer_cuda_example.cpp
)

# Link all libraries to the CUDA layer example
target_link_libraries(layer_cuda_example 
    matrix_lib
    matrix_cuda_lib
    activation_lib
    activation_cuda_lib
    loss_lib
    loss_cuda_lib
    layer_lib
    layer_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(layer_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
