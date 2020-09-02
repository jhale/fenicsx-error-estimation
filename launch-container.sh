#!/bin/bash
docker run --env PETSC_ARCH=linux-gnu-real-32 -ti --rm -v "$(pwd)":/root/shared -w /root/shared fenicsproject/test-env:mpich
