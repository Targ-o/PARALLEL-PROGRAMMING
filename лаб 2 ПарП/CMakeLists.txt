
cmake_minimum_required(VERSION 3.31)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
    "0e5b6991-d74f-4b3d-a41c-cf096e0b2508"
)

set(CMAKE_CXX_MODULE_STD ON)

project(MatrixMultiplication LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(OpenMP REQUIRED)

add_executable(matrix_multiply
    src/Source1.cpp
)

target_link_libraries(matrix_multiply PRIVATE OpenMP::OpenMP_CXX)
