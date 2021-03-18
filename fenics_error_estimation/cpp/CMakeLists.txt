cmake_minimum_required(VERSION 3.0)

project(fenics_error_estimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

pybind11_add_module(cpp SHARED
    projected_local_solver.cpp
)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -Werror -pedantic" HAVE_PEDANTIC)
if (HAVE_PEDANTIC)
  target_compile_options(cpp PRIVATE -Wall;-Werror;-pedantic)
endif()

# In Debug mode override pybind11 symbols visibility
# Symbols must be visible to backtrace_symbols() to produce nice logs
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(cpp PRIVATE "-fvisibility=default")
endif()

target_include_directories(cpp PRIVATE dolfinx petsc mpi)
target_link_libraries(cpp PRIVATE pybind11::module dolfinx Eigen3::Eigen)
