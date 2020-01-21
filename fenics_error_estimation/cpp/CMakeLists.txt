cmake_minimum_required(VERSION 2.8)

project(bank_weiser)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 HAVE_STD_CPP11)
if (HAVE_STD_CPP11)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  CHECK_CXX_COMPILER_FLAG(-std=c++0x HAVE_STD_CPP0x)
  if (HAVE_STD_CPP0x)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  endif()
endif()

find_package(DOLFIN REQUIRED)
find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})
include(${DOLFIN_USE_FILE})

pybind11_add_module(cpp SHARED
    projected_local_solver.cpp
)

set(CMAKE_CXX_FLAGS "-march=native -O3")

target_link_libraries(cpp PRIVATE pybind11::module ${DOLFIN_LIBRARIES}) 
