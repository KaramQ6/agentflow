cmake_minimum_required(VERSION 3.15)
project(agentflow_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_agentflow_cpp
    src/agentflow/cpp_core/bindings.cpp
    src/agentflow/cpp_core/dag_engine.cpp
)

target_include_directories(_agentflow_cpp PRIVATE src/agentflow/cpp_core)
