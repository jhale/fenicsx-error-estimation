#!/bin/bash
set -e
#CONTAINER_ENGINE="podman"
CONTAINER_ENGINE="docker"

${CONTAINER_ENGINE} pull fenicsproject/test-env:latest-mpich
${CONTAINER_ENGINE} pull dolfinx/dev-env
${CONTAINER_ENGINE} build --no-cache -t jhale/fenicsx-error-estimation:debug .
${CONTAINER_ENGINE} build --no-cache \
   --build-arg CMAKE_BUILD_TYPE=RelWithDebug --build-arg PIP_EXTRA_FLAGS='' \
   --build-arg CXXFLAGS='-O2 -g' --build-arg IMAGE=dolfinx/dev-env -t jhale/fenicsx-error-estimation:release .
