cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(NexusCache-Engine VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug Release RelWithDebInfo" FORCE)
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERION}")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE TORCH_PREFIX_ERROR
)
if(NOT TORCH_CMAKE_PREFIX)
    message(FATAL_ERROR "Could not find PyTorch cmake prefix path. Verify 'torch_env' is activated: ${TORCH_PREFIX_ERROR}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
message(STATUS "PyTorch/LibTorch Path: ${Torch_DIR}")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_PREFIX}")
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "PyBind11 CMake Path: ${PYBIND11_CMAKE_PREFIX}")
else()
    message(WARNING "PyBind11 CMake files not found directly; falling back to standard include paths.")
endif()
set(NEXUS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include")
set(NEXUS_SOURCES
    csrc/src/block_manager.cpp
    csrc/src/page_table.cpp
    csrc/src/memory_utils.cpp
    csrc/kernels/paged_attention_kernel.cu
    csrc/kernels/memory_kernels.cu
)
add_library(nexuscache_core STATIC ${NEXUS_SOURCES})
target_include_directories(nexuscache_core
    PUBLIC
        ${NEXUS_INCLUDE_DIR}
        ${NEXUS_INCLUDE_DIR}/nexuscache
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(nexuscache_core
    PUBLIC
        ${TORCH_LIBRARIES}
)

set_target_properties(nexuscache_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
set(BINDINGS_SOURCE csrc/bindings/bindings.cpp)

if(TARGET pybind11::module)
    pybind11_add_module(_C ${BINDINGS_SOURCE})
else()
    add_library(_C MODULE ${BINDINGS_SOURCE})
    target_link_libraries(_C PRIVATE Python::Python)
endif()

target_link_libraries(_C PRIVATE nexuscache_core)

set_target_properties(_C PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nexuscache"
)
target_compile_definitions(_C PRIVATE
    TORCH_API_INCLUDE_EXTENSION_H
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
)
option(BUILD_CPP_TESTS "Build C++ unit tests for block manager and memory ops" ON)

if(BUILD_CPP_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(nexuscache_cpp_tests
        tests/cpp/test_block_allocator.cpp
        tests/cpp/test_pinned_memory.cpp
    )

    target_link_libraries(nexuscache_cpp_tests
        PRIVATE
            nexuscache_core
            gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(nexuscache_cpp_tests)
endif()
message(STATUS "NexusCache-Engine CMake Configured Successfully!")
