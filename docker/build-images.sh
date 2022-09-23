#!/bin/bash
set -e
#CONTAINER_ENGINE="podman"
CONTAINER_ENGINE="docker"

${CONTAINER_ENGINE} build -t jhale/fenicsx-error-estimation:debug .
${CONTAINER_ENGINE} build \
   --build-arg CMAKE_BUILD_TYPE=RelWithDebug --build-arg PIP_EXTRA_FLAGS='' \
   --build-arg CXXFLAGS='-O3' --build-arg IMAGE=dolfinx/dev-env:v0.5.1-r1 -t jhale/fenicsx-error-estimation:release .
