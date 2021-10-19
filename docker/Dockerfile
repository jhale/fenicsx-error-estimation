ARG IMAGE=fenicsproject/test-env:latest-mpich
FROM ${IMAGE}

ARG CMAKE_BUILD_TYPE=Debug
ARG CXXFLAGS="-g -Wall"
ARG PIP_EXTRA_FLAGS="--global-option --debug"

WORKDIR /src

RUN git clone https://github.com/xtensor-stack/xtensor-blas.git && \
    cd xtensor-blas && \
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

RUN git clone https://github.com/FEniCS/dolfinx.git && \
    cd dolfinx && git checkout jhale/remove-function-space-check && cd ../ && \
    git clone --depth 1 https://github.com/FEniCS/ffcx.git && \
    git clone --depth 1 https://github.com/FEniCS/basix.git && \
    git clone --depth 1 https://github.com/FEniCS/ufl.git

RUN cd basix && python3 -m pip install . && \
    cd ../ufl && python3 -m pip install . && \
    cd ../ffcx && python3 -m pip install .

ENV PETSC_DIR=/usr/local/petsc
ENV PETSC_ARCH=linux-gnu-real-32
RUN cd dolfinx && \
    mkdir build && \
    cd build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CXXFLAGS} ../cpp && \
    ninja install && \
    cd ../python && \
    CXXFLAGS=${CXXFLAGS} python3 -m pip -v install --global-option build ${PIP_EXTRA_FLAGS} .
