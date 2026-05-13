cmake_minimum_required(VERSION 3.24)
project(capp_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

add_library(capp_native_core
  cpp/src/version.cpp
  cpp/src/solver.cpp
)

target_include_directories(capp_native_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

target_link_libraries(capp_native_core
  PUBLIC
    pybind11::headers
    Python3::Module
    Python3::NumPy
)

pybind11_add_module(capp_native
  cpp/bindings/module.cpp
)

target_link_libraries(capp_native
  PRIVATE
    capp_native_core
)

set_target_properties(capp_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

