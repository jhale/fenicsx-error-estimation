#!/bin/bash -l
set -ex

LAUNCHER="fenicsx-launcher.sh"
COMMAND="python3 demo_three_dimensions.py"

#sbatch --out 1.out -N 1 --ntasks-per-node 128 ${LAUNCHER} ${COMMAND}
#sbatch --out 2.out -N 2 --ntasks-per-node 128 ${LAUNCHER} ${COMMAND}
#sbatch --out 4.out -N 4 --ntasks-per-node 128 ${LAUNCHER} ${COMMAND}
sbatch --out 16.out -N 16 --ntasks-per-node 128 ${LAUNCHER} ${COMMAND}
