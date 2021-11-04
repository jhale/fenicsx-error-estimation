#!/bin/bash
set -e
CONTAINER_ENGINE="podman"
#CONTAINER_ENGINE="docker"

${CONTAINER_ENGINE} push jhale/fenicsx-error-estimation:debug
${CONTAINER_ENGINE} push jhale/fenicsx-error-estimation:release
