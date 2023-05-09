#!/bin/bash

#SBATCH --nodes=59
#SBATCH --time=1-24:00:00
#SBATCH --account=hfm
#SBATCH --job-name=nrelvi_u_7_syn_turb_cm_imp
#SBATCH --output=out.%x_%j
#SBATCH --qos=high

module purge
export SPACK_MANAGER=/scratch/sbidadi/spack-manager-5/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
source ${SPACK_MANAGER}/configs/eagle/env.sh
module load mpt 
spack env activate -d ${SPACK_MANAGER}/environments/iddes
spack load exawind

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. MPI Ranks = $mpi_ranks"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"

#cd sst 
#mpirun -np 1152 exawind --awind 972 --nwind 180 nrelvi-sst.yaml 

#cd ../iddes_10r
#mpirun -np 1152 exawind --awind 972 --nwind 180 nrelvi-iddes.yaml

cd iddes_20r2
mpirun -np 2124 exawind --awind 1944 --nwind 180 nrelvi-iddes.yaml
