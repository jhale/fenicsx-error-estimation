#!/bin/bash
podman run --env PETSC_ARCH=linux-gnu-real-32 -ti --rm -v "$(pwd)":/root/shared -w /root/shared fenics-error-estimation/dolfinx:release
