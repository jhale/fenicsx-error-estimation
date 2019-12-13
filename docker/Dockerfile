FROM quay.io/fenicsproject/dev-env:latest

USER root
WORKDIR /tmp
ARG GMSH_VERSION=4.4.1

RUN git clone https://bitbucket.org/jackhale/dolfin.git && \
    cd dolfin && git checkout jackhale/wrap-meshdata-exists && cd ../ && \
    git clone https://bitbucket.org/fenics-project/ufl.git && \
    cd ufl && git checkout f77247ff383e123ece55c9e11434b6c122efc295 && cd ../ && \
    git clone https://bitbucket.org/fenics-project/ffc.git && \
    git clone https://bitbucket.org/fenics-project/dijitso.git && \
    git clone https://bitbucket.org/fenics-project/fiat.git && \
    git clone https://bitbucket.org/fenics-project/mshr.git && \
    FENICS_PREFIX=/usr/local FENICS_SRC_DIR=/tmp /home/fenics/bin/fenics-build

RUN apt-get update && \
    pip3 install --upgrade pytest && \
    pip3 install --upgrade pip && \
    pip3 install pandas && \
    apt-get install -y libxcursor-dev \
                       libxft2 \
                       libglu1 \
                       libxft2 \
                       libxinerama1 \
                       python3-lxml && \
    export HDF5_MPI="ON" && \
    pip3 install --no-cache-dir --no-binary=h5py cffi decorator flake8 \
                                                 h5py pygmsh pytest-xdist \
                                                 sphinx sphinx_rtd_theme && \
    pip3 install pygmsh && \
    cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64/bin:$PATH
