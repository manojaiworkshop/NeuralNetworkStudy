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
