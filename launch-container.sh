#!/bin/bash
docker run --cap-add SYS_PTRACE -ti --rm -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared rbulle/bank-weiser:latest
