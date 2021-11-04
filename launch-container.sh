#!/bin/bash
podman run -ti --rm -v "$(pwd)":/root/shared -w /root/shared jhale/fenicsx-error-estimation:release
