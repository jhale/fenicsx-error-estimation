#!/bin/bash
docker run -ti --rm -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared rbulle/fenics-error-estimation:latest
