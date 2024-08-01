# Copyright (c) 2020-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(ast_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  ast_example 
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# Configure your project here
add_executable(ast_example src/ast_test.cpp)
target_link_libraries(ast_example PRIVATE cudf::cudftestutil)
target_compile_features(ast_example PRIVATE cxx_std_17)

install(TARGETS ast_example DESTINATION bin/examples/libcudf)
