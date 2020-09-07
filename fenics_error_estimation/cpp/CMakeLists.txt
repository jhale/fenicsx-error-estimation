cmake_minimum_required(VERSION 2.8)

project(bank_weiser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DOLFINX REQUIRED)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

pybind11_add_module(cpp SHARED
    projected_local_solver.cpp
)

target_include_directories(cpp PRIVATE dolfinx petsc mpi)
target_link_libraries(cpp PRIVATE pybind11::module dolfinx)
