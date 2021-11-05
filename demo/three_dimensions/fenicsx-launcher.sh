#!/bin/bash -l
#SBATCH --time=0-00:03:00
#SBATCH -p batch
#SBATCH -J fenicsx-ee-scaling
#SBATCH --contiguous
#SBATCH --exclusive
#SBATCH -c 1
set -e

source $HOME/fenicsx-aion-master-r23/bin/env-fenics.sh

echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Number of nodes: ${SLURM_NNODES}"
echo "== Number of tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"

cd $SLURM_SUBMIT_DIR
echo $@
srun -v --cpu-bind=cores "$@"

echo "== Finished at $(date)"
