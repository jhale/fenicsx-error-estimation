#!/bin/bash
docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --env PETSC_ARCH=linux-gnu-real-32 -ti --rm -v "$(pwd)":/root/shared -w /root/shared dolfinx/dolfinx
