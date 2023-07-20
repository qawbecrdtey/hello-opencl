cmake_minimum_required(VERSION 3.25)
project(hello-opencl)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCL REQUIRED)

add_executable(opencl-info src/opencl-info.cpp)
target_compile_options(opencl-info PRIVATE -Wall -Wextra)
target_include_directories(opencl-info PRIVATE include)
target_link_libraries(opencl-info PRIVATE OpenCL::OpenCL)

add_executable(opencl-vector-addition src/opencl-vector-addition.cpp)
target_compile_options(opencl-vector-addition PRIVATE -Wall -Wextra)
target_include_directories(opencl-vector-addition PRIVATE include)
target_link_libraries(opencl-vector-addition PRIVATE OpenCL::OpenCL)