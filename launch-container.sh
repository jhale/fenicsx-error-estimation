#!/bin/bash
docker run -ti --rm -v "$(pwd)":/root/shared -w /root/shared dolfinx/dolfinx
