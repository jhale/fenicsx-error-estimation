#!/bin/bash
podman run -ti --rm -v "$(pwd)":/root/shared -w /root/shared fenics-error-estimation/dolfinx:debug
