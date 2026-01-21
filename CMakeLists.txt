# CMake minimum version required
cmake_minimum_required(VERSION 3.10)























































































































































































































































































































































































































































































###########################################################
# RNN (RECURRENT NEURAL NETWORK) LIBRARIES AND EXAMPLES
###########################################################

# Create RNN library (CPU)
add_library(rnn_lib STATIC
    src/rnn.cpp
)

# Link RNN library with required libraries
target_link_libraries(rnn_lib 
    matrix_lib
    activation_lib
)

# Create RNN example executable (CPU)
add_executable(rnn_example
    example/rnn_example.cpp
)

# Link libraries to RNN example
target_link_libraries(rnn_example 
    matrix_lib
    activation_lib
    rnn_lib
)

# Create CUDA RNN library (GPU)
add_library(rnn_cuda_lib STATIC
    src/rnn_cuda.cu
)

# Set CUDA properties for RNN CUDA library
set_target_properties(rnn_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link RNN CUDA library with other libraries
target_link_libraries(rnn_cuda_lib 
    matrix_lib
    matrix_cuda_lib
    activation_lib
    activation_cuda_lib
    rnn_lib
)

# Create CUDA RNN example executable (GPU)
add_executable(rnn_cuda_example
    example/rnn_cuda_example.cpp
)

# Link all libraries to CUDA RNN example
target_link_libraries(rnn_cuda_example 
    matrix_lib
    matrix_cuda_lib
    activation_lib
    activation_cuda_lib
    rnn_lib
    rnn_cuda_lib
)

# Set CUDA properties for the executable
set_target_properties(rnn_cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

###########################################################
