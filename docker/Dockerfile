ARG IMAGE=fenicsproject/test-env:latest-mpich
FROM ${IMAGE}

ARG CMAKE_BUILD_TYPE=Debug
ARG CXXFLAGS="-g -Wall"
ARG PIP_EXTRA_FLAGS="--global-option --debug"

WORKDIR /src

RUN git clone https://github.com/xtensor-stack/xtensor-blas.git && \
    cd xtensor-blas && \
    git checkout 0.19.2 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtensor-blas
RUN git clone https://github.com/xtensor-stack/xtensor-python.git && \
    cd xtensor-python && \
    git checkout 0.25.3 && \
    mkdir build && \
    cd build && \
    export pybind11_DIR=`python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'` && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtensor-python

RUN python3 -m pip install pip setuptools --upgrade && \
    python3 -m pip install git+git://github.com/FEniCS/basix.git@main && \
    python3 -m pip install git+git://github.com/FEniCS/ufl.git@main && \
    python3 -m pip install git+git://github.com/FEniCS/ffcx.git@main

ENV PETSC_DIR=/usr/local/petsc
ENV PETSC_ARCH=linux-gnu-real-32
RUN git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && git checkout jhale/remove-function-space-check && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CXXFLAGS} -G Ninja -B build-dir -S cpp && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    cd python && \
    CXXFLAGS=${CXXFLAGS} python3 -m pip -v install --global-option build ${PIP_EXTRA_FLAGS} .

RUN python3 -m pip install pandas matplotlib mpltools
