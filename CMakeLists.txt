cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkFromScratch LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (Turing for RTX 5000)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Enable CUDA language support (modern CMake approach)
enable_language(CUDA)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

# ============================================================================
# MATRIX LIBRARY
# ============================================================================

# CPU Matrix Library
add_library(matrix_lib STATIC src/matrix.cpp)
target_include_directories(matrix_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Matrix Library
add_library(matrix_cuda_lib STATIC src/matrix_cuda.cu)
target_include_directories(matrix_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(matrix_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Matrix Example (CPU)
add_executable(matrix_example example/matrix_example.cpp)
target_link_libraries(matrix_example matrix_lib)

# Matrix CUDA Example
add_executable(matrix_cuda_example example/matrix_cuda_example.cpp)
target_link_libraries(matrix_cuda_example matrix_cuda_lib)

# ============================================================================
# ACTIVATION LIBRARY
# ============================================================================

# CPU Activation Library
add_library(activation_lib STATIC src/activation.cpp)
target_link_libraries(activation_lib matrix_lib)
target_include_directories(activation_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Activation Library
add_library(activation_cuda_lib STATIC src/activation_cuda.cu)
target_link_libraries(activation_cuda_lib matrix_cuda_lib)
target_include_directories(activation_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(activation_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Activation Example (CPU)
add_executable(activation_example example/activation_detailed_example.cpp)
target_link_libraries(activation_example activation_lib matrix_lib)

# Activation CUDA Example
add_executable(activation_cuda_example example/activation_cuda_example.cpp)
target_link_libraries(activation_cuda_example activation_cuda_lib matrix_cuda_lib)

# ============================================================================
# LAYER LIBRARY
# ============================================================================

# CPU Layer Library
add_library(layer_lib STATIC src/layer.cpp)
target_link_libraries(layer_lib matrix_lib activation_lib)
target_include_directories(layer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Layer Library
add_library(layer_cuda_lib STATIC src/layer_cuda.cu)
target_link_libraries(layer_cuda_lib matrix_cuda_lib activation_cuda_lib)
target_include_directories(layer_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(layer_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Layer Example (CPU)
add_executable(layer_example example/layer_example.cpp)
target_link_libraries(layer_example layer_lib activation_lib matrix_lib)

# Layer CUDA Example
add_executable(layer_cuda_example example/layer_cuda_example.cpp)
target_link_libraries(layer_cuda_example layer_cuda_lib activation_cuda_lib matrix_cuda_lib)

# ============================================================================
# LOSS LIBRARY
# ============================================================================

# CPU Loss Library
add_library(loss_lib STATIC src/loss.cpp)
target_link_libraries(loss_lib matrix_lib)
target_include_directories(loss_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Loss Library
add_library(loss_cuda_lib STATIC src/loss_cuda.cu)
target_link_libraries(loss_cuda_lib matrix_cuda_lib)
target_include_directories(loss_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(loss_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Loss Example (CPU)
add_executable(loss_detailed_example example/loss_detailed_example.cpp)
target_link_libraries(loss_detailed_example loss_lib matrix_lib)

# Loss CUDA Example
add_executable(loss_cuda_example example/loss_cuda_example.cpp)
target_link_libraries(loss_cuda_example loss_cuda_lib matrix_cuda_lib)

# ============================================================================
# OPTIMIZER LIBRARY
# ============================================================================

# CPU Optimizer Library
add_library(optimizer_lib STATIC src/optimizer.cpp)
target_link_libraries(optimizer_lib matrix_lib)
target_include_directories(optimizer_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Optimizer Library
add_library(optimizer_cuda_lib STATIC src/optimizer_cuda.cu)
target_link_libraries(optimizer_cuda_lib matrix_cuda_lib)
target_include_directories(optimizer_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(optimizer_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optimizer Example (CPU)
add_executable(optimizer_example example/optimizer_example.cpp)
target_link_libraries(optimizer_example optimizer_lib matrix_lib)

# Optimizer CUDA Example
add_executable(optimizer_cuda_example example/optimizer_cuda_example.cpp)
target_link_libraries(optimizer_cuda_example optimizer_cuda_lib matrix_cuda_lib)

# ============================================================================
# NETWORK LIBRARY
# ============================================================================

# CPU Network Library
add_library(network_lib STATIC src/network.cpp)
target_link_libraries(network_lib layer_lib activation_lib loss_lib optimizer_lib matrix_lib)
target_include_directories(network_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA Network Library
add_library(network_cuda_lib STATIC src/network_cuda.cu)
target_link_libraries(network_cuda_lib layer_cuda_lib activation_cuda_lib loss_cuda_lib optimizer_cuda_lib matrix_cuda_lib)
target_include_directories(network_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(network_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Network Complete Example (CPU)
add_executable(network_complete_example example/network_complete_example.cpp)
target_link_libraries(network_complete_example network_lib layer_lib activation_lib loss_lib optimizer_lib matrix_lib)

# Network CUDA Example
add_executable(network_cuda_example example/network_cuda_example.cpp)
target_link_libraries(network_cuda_example network_cuda_lib layer_cuda_lib activation_cuda_lib loss_cuda_lib optimizer_cuda_lib matrix_cuda_lib)

# Network CUDA Sample
add_executable(network_cuda_sample example/network_cuda_sample.cpp)
target_link_libraries(network_cuda_sample network_cuda_lib layer_cuda_lib activation_cuda_lib loss_cuda_lib optimizer_cuda_lib matrix_cuda_lib)

# ============================================================================
# RNN LIBRARY
# ============================================================================

# CPU RNN Library
add_library(rnn_lib STATIC src/rnn.cpp)
target_link_libraries(rnn_lib matrix_lib activation_lib optimizer_lib loss_lib)
target_include_directories(rnn_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# CUDA RNN Library
add_library(rnn_cuda_lib STATIC src/rnn_cuda.cu)
target_link_libraries(rnn_cuda_lib matrix_cuda_lib activation_cuda_lib optimizer_cuda_lib loss_cuda_lib)
target_include_directories(rnn_cuda_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(rnn_cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# RNN Example (CPU)
add_executable(rnn_example example/rnn_example.cpp)
target_link_libraries(rnn_example rnn_lib matrix_lib activation_lib optimizer_lib loss_lib)

# RNN CUDA Example
add_executable(rnn_cuda_example example/rnn_cuda_example.cpp)
target_link_libraries(rnn_cuda_example rnn_cuda_lib matrix_cuda_lib activation_cuda_lib optimizer_cuda_lib loss_cuda_lib)

# ============================================================================
# LSTM LIBRARY
# ============================================================================

# CPU LSTM Library
add_library(lstm_lib STATIC src/lstm.cpp)
target_link_libraries(lstm_lib matrix_lib activation_lib)
target_include_directories(lstm_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# LSTM Example (CPU)
add_executable(lstm_example example/lstm_example.cpp)
target_link_libraries(lstm_example lstm_lib gru_lib rnn_lib matrix_lib activation_lib)

# Comparison Example (RNN vs LSTM vs GRU)
add_executable(comparison_example example/comparison_example.cpp)
target_link_libraries(comparison_example lstm_lib gru_lib rnn_lib matrix_lib activation_lib)

# ============================================================================
# GRU LIBRARY
# ============================================================================

# CPU GRU Library
add_library(gru_lib STATIC src/gru.cpp)
target_link_libraries(gru_lib matrix_lib activation_lib)
target_include_directories(gru_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ============================================================================
# BUILD INFORMATION
# ============================================================================

message(STATUS "===========================================")
message(STATUS "Neural Network From Scratch - Build Info")
message(STATUS "===========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Build Directory: ${CMAKE_BINARY_DIR}")
message(STATUS "===========================================")
message(STATUS "Components:")
message(STATUS "  - Matrix (CPU + CUDA)")
message(STATUS "  - Activation Functions (CPU + CUDA)")
message(STATUS "  - Layers (CPU + CUDA)")
message(STATUS "  - Loss Functions (CPU + CUDA)")
message(STATUS "  - Optimizers (CPU + CUDA)")
message(STATUS "  - Networks (CPU + CUDA)")
message(STATUS "  - RNN (CPU + CUDA)")
message(STATUS "===========================================")
message(STATUS "Available Examples:")
message(STATUS "  CPU Examples:")
message(STATUS "    - matrix_example")
message(STATUS "    - activation_example")
message(STATUS "    - layer_example")
message(STATUS "    - loss_detailed_example")
message(STATUS "    - optimizer_example")
message(STATUS "    - lstm_example")
message(STATUS "    - network_complete_example")
message(STATUS "    - rnn_example")
message(STATUS "  GPU Examples:")
message(STATUS "    - matrix_cuda_example")
message(STATUS "    - activation_cuda_example")
message(STATUS "    - layer_cuda_example")
message(STATUS "    - loss_cuda_example")
message(STATUS "    - optimizer_cuda_example")
message(STATUS "    - network_cuda_example")
message(STATUS "    - network_cuda_sample")
message(STATUS "    - rnn_cuda_example")
message(STATUS "===========================================")
