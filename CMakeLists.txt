cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkFromScratch CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (RTX 5000 = Turing = compute_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Fix CUDA 11.5 compatibility with GCC 11
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler -std=c++14")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# =============================================================================
# CORE LIBRARIES (CPU)
# =============================================================================

# Matrix Library (CPU)
add_library(matrix_lib STATIC src/matrix.cpp)
target_include_directories(matrix_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Activation Library (CPU)
add_library(activation_lib STATIC src/activation.cpp)
target_link_libraries(activation_lib matrix_lib)
target_include_directories(activation_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Layer Library (CPU)
add_library(layer_lib STATIC src/layer.cpp)
target_link_libraries(layer_lib matrix_lib activation_lib)
target_include_directories(layer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Loss Library (CPU)
add_library(loss_lib STATIC src/loss.cpp)
target_link_libraries(loss_lib matrix_lib)
target_include_directories(loss_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optimizer Library (CPU)
add_library(optimizer_lib STATIC src/optimizer.cpp)
target_link_libraries(optimizer_lib matrix_lib layer_lib)
target_include_directories(optimizer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Network Library (CPU)
add_library(network_lib STATIC src/network.cpp)
target_link_libraries(network_lib matrix_lib layer_lib activation_lib loss_lib optimizer_lib)
target_include_directories(network_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# RNN Library (CPU)
add_library(rnn_lib STATIC src/rnn.cpp)
target_link_libraries(rnn_lib matrix_lib activation_lib)
target_include_directories(rnn_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# LSTM Library (CPU)
add_library(lstm_lib STATIC src/lstm.cpp)
target_link_libraries(lstm_lib matrix_lib activation_lib)
target_include_directories(lstm_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# GRU Library (CPU)
add_library(gru_lib STATIC src/gru.cpp)
target_link_libraries(gru_lib matrix_lib activation_lib)
target_include_directories(gru_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Attention Library (CPU)
add_library(attention_lib STATIC src/attention.cpp)
target_link_libraries(attention_lib matrix_lib activation_lib)
target_include_directories(attention_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# =============================================================================
# CUDA LIBRARIES (GPU)
# =============================================================================

# Matrix Library (CUDA)
add_library(matrix_cuda_lib STATIC src/matrix_cuda.cu)
target_include_directories(matrix_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(matrix_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Activation Library (CUDA)
add_library(activation_cuda_lib STATIC src/activation_cuda.cu)
target_link_libraries(activation_cuda_lib matrix_cuda_lib)
target_include_directories(activation_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(activation_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Layer Library (CUDA)
add_library(layer_cuda_lib STATIC src/layer_cuda.cu)
target_link_libraries(layer_cuda_lib matrix_cuda_lib activation_cuda_lib)
target_include_directories(layer_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(layer_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Loss Library (CUDA)
add_library(loss_cuda_lib STATIC src/loss_cuda.cu)
target_link_libraries(loss_cuda_lib matrix_cuda_lib)
target_include_directories(loss_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(loss_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optimizer Library (CUDA)
add_library(optimizer_cuda_lib STATIC src/optimizer_cuda.cu)
target_link_libraries(optimizer_cuda_lib matrix_cuda_lib layer_cuda_lib)
target_include_directories(optimizer_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(optimizer_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Network Library (CUDA)
add_library(network_cuda_lib STATIC src/network_cuda.cu)
target_link_libraries(network_cuda_lib matrix_cuda_lib layer_cuda_lib activation_cuda_lib loss_cuda_lib optimizer_cuda_lib)
target_include_directories(network_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(network_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# RNN Library (CUDA)
add_library(rnn_cuda_lib STATIC src/rnn_cuda.cu)
target_link_libraries(rnn_cuda_lib matrix_cuda_lib activation_cuda_lib)
target_include_directories(rnn_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(rnn_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# LSTM Library (CUDA)
add_library(lstm_cuda_lib STATIC src/lstm_cuda.cu)
target_link_libraries(lstm_cuda_lib matrix_cuda_lib activation_cuda_lib)
target_include_directories(lstm_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(lstm_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Attention Library (CUDA)
add_library(attention_cuda_lib STATIC src/attention_cuda.cu)
target_link_libraries(attention_cuda_lib matrix_cuda_lib activation_cuda_lib)
target_include_directories(attention_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(attention_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =============================================================================
# CPU EXAMPLES
# =============================================================================

# Matrix Example (CPU)
add_executable(matrix_example example/matrix_example.cpp)
target_link_libraries(matrix_example matrix_lib)

# Activation Example (CPU)
add_executable(activation_example example/activation_detailed_example.cpp)
target_link_libraries(activation_example activation_lib matrix_lib)

# Layer Example (CPU)
add_executable(layer_example example/layer_example.cpp)
target_link_libraries(layer_example layer_lib matrix_lib activation_lib)

# Loss Example (CPU)
add_executable(loss_example example/loss_detailed_example.cpp)
target_link_libraries(loss_example loss_lib matrix_lib)

# Optimizer Example (CPU)
add_executable(optimizer_example example/optimizer_example.cpp)
target_link_libraries(optimizer_example optimizer_lib layer_lib matrix_lib activation_lib)

# Network Example (CPU)
add_executable(network_example example/network_complete_example.cpp)
target_link_libraries(network_example network_lib matrix_lib layer_lib activation_lib loss_lib optimizer_lib)

# RNN Example (CPU)
add_executable(rnn_example example/rnn_example.cpp)
target_link_libraries(rnn_example rnn_lib matrix_lib activation_lib)

# LSTM Example (CPU)
add_executable(lstm_example example/lstm_example.cpp)
target_link_libraries(lstm_example lstm_lib matrix_lib activation_lib)

# Attention Example (CPU)
add_executable(attention_example example/attention_example.cpp)
target_link_libraries(attention_example attention_lib lstm_lib gru_lib rnn_lib matrix_lib activation_lib)

# =============================================================================
# CUDA EXAMPLES
# =============================================================================

# Matrix Example (CUDA)
add_executable(matrix_cuda_example example/matrix_cuda_example.cpp)
target_link_libraries(matrix_cuda_example matrix_cuda_lib)
set_target_properties(matrix_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Activation Example (CUDA)
add_executable(activation_cuda_example example/activation_cuda_example.cpp)
target_link_libraries(activation_cuda_example activation_cuda_lib matrix_cuda_lib)
set_target_properties(activation_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Layer Example (CUDA)
add_executable(layer_cuda_example example/layer_cuda_example.cpp)
target_link_libraries(layer_cuda_example layer_cuda_lib matrix_cuda_lib activation_cuda_lib)
set_target_properties(layer_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Loss Example (CUDA)
add_executable(loss_cuda_example example/loss_cuda_example.cpp)
target_link_libraries(loss_cuda_example loss_cuda_lib matrix_cuda_lib)
set_target_properties(loss_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optimizer Example (CUDA)
add_executable(optimizer_cuda_example example/optimizer_cuda_example.cpp)
target_link_libraries(optimizer_cuda_example optimizer_cuda_lib layer_cuda_lib matrix_cuda_lib activation_cuda_lib)
set_target_properties(optimizer_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Network Example (CUDA)
add_executable(network_cuda_example example/network_cuda_example.cpp)
target_link_libraries(network_cuda_example network_cuda_lib matrix_cuda_lib layer_cuda_lib activation_cuda_lib loss_cuda_lib optimizer_cuda_lib)
set_target_properties(network_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# RNN Example (CUDA)
add_executable(rnn_cuda_example example/rnn_cuda_example.cpp)
target_link_libraries(rnn_cuda_example rnn_cuda_lib matrix_cuda_lib activation_cuda_lib)
set_target_properties(rnn_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# LSTM Example (CUDA)
add_executable(lstm_cuda_example example/lstm_cuda_example.cpp)
target_link_libraries(lstm_cuda_example lstm_cuda_lib matrix_cuda_lib activation_cuda_lib)
set_target_properties(lstm_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Attention Example (CUDA)
add_executable(attention_cuda_example example/attention_cuda_example.cpp)
target_link_libraries(attention_cuda_example attention_cuda_lib matrix_cuda_lib activation_cuda_lib)
set_target_properties(attention_cuda_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =============================================================================
# BUILD INFORMATION
# =============================================================================

message(STATUS "====================================")
message(STATUS "Neural Network From Scratch - Build Configuration")
message(STATUS "====================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "====================================")
