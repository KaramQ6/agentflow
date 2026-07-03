cmake_minimum_required(VERSION 3.15)
project(agentflow_cpp LANGUAGES NONE)

set(SKIP_CPP_EXTENSION OFF CACHE BOOL "Skip building the C++ pybind11 extension")

if(NOT SKIP_CPP_EXTENSION)
    include(CheckLanguage)
    check_language(CXX)
    if(CMAKE_CXX_COMPILER)
        enable_language(CXX)
        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()
endif()

if(CMAKE_CXX_COMPILER AND NOT SKIP_CPP_EXTENSION)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found — building _agentflow_cpp extension")
        pybind11_add_module(_agentflow_cpp
            src/agentflow/cpp_core/bindings.cpp
            src/agentflow/cpp_core/dag_engine.cpp
        )
        target_include_directories(_agentflow_cpp PRIVATE src/agentflow/cpp_core)
    else()
        message(STATUS "pybind11 not found — skipping _agentflow_cpp extension (Python fallback will be used)")
    endif()
else()
    message(STATUS "C++ compiler not available — skipping _agentflow_cpp extension (Python fallback will be used)")
endif()
