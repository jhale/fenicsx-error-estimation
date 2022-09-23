ARG IMAGE=fenicsproject/test-env:v0.5.1-r1-mpich
FROM ${IMAGE}

ARG CMAKE_BUILD_TYPE=Debug
ARG CXXFLAGS="-g -Wall"
ARG PIP_EXTRA_FLAGS="--global-option --debug"

WORKDIR /src

RUN git clone https://github.com/xtensor-stack/xtl.git && \
    cd xtl && \
    git checkout 0.7.4 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtl
RUN git clone https://github.com/xtensor-stack/xtensor.git && \
    cd xtensor && \
    git checkout 0.24.3 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtensor
RUN git clone https://github.com/xtensor-stack/xtensor-blas.git && \
    cd xtensor-blas && \
    git checkout 0.20.0 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtensor-blas
RUN git clone https://github.com/xtensor-stack/xtensor-python.git && \
    cd xtensor-python && \
    git checkout 0.26.1 && \
    mkdir build && \
    cd build && \
    export pybind11_DIR=`python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'` && \
    cmake ../ && \
    make install && \
    rm -rf /src/xtensor-python

RUN python3 -m pip install git+https://github.com/FEniCS/basix.git@v0.5.1 && \
    python3 -m pip install git+https://github.com/FEniCS/ufl.git@2022.2.0 && \
    python3 -m pip install git+https://github.com/FEniCS/ffcx.git@v0.5.0.post0

ENV PETSC_DIR=/usr/local/petsc
ENV PETSC_ARCH=linux-gnu-real-32
RUN git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && git checkout jhale/remove-function-space-check && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CXXFLAGS} -G Ninja -B build-dir -S cpp && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    cd python && \
    CXXFLAGS=${CXXFLAGS} python3 -m pip -v install --global-option build ${PIP_EXTRA_FLAGS} .
