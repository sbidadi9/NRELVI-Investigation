#!/bin/bash

#SBATCH --nodes=32
#SBATCH --time=03:00:00
#SBATCH --account=hfm
#SBATCH --job-name=nrelvi_u_7_sst
#SBATCH --output=out.%x_%j

module purge
export SPACK_MANAGER=/scratch/sbidadi/spack-manager-3/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
source ${SPACK_MANAGER}/configs/eagle/env.sh
module load mpt
spack env activate -d ${SPACK_MANAGER}/environments/sst_test
spack load exawind

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. MPI Ranks = $mpi_ranks"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"

mpirun -np 1152 exawind --awind 972 --nwind 180 nrelvi-sst.yaml 
