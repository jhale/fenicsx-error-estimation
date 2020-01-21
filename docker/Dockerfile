FROM quay.io/fenicsproject/dev:latest

USER root
WORKDIR /tmp
ARG GMSH_VERSION=4.5.1

RUN apt-get update && \
    pip3 install --no-cache-dir --upgrade pytest && \
    pip3 install --no-cache-dir --upgrade pip && \
    apt-get install -y libxcursor-dev \
                       libxft2 \
                       libglu1 \
                       libxft2 \
                       libxinerama1 \
                       python3-lxml && \
    export HDF5_MPI="ON" && \
    pip3 install --no-cache-dir --no-binary=h5py flake8 h5py meshio mpltools pandas\
                                                 pygmsh sphinx sphinx_rtd_theme && \
    cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64/bin:$PATH
