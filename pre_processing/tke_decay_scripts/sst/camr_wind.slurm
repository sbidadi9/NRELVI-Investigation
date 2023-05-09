#!/bin/bash

#SBATCH --nodes=3
#SBATCH --time=8:00:00
#SBATCH --account=hfm
#SBATCH --job-name=u_12_tke_0p85_sdr_150
#SBATCH --output=out.%x_%j

module purge
export SPACK_MANAGER=/scratch/sbidadi/spack-manager-2/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
source ${SPACK_MANAGER}/configs/eagle/env.sh
module load mpt 
spack env activate -d ${SPACK_MANAGER}/environments/iddes_fix
spack load amr-wind

###source /nopt/nrel/ecom/exawind/exawind/scripts/exawind-env-intel.sh

ranks_per_node=36
mpi_ranks=$(expr $SLURM_JOB_NUM_NODES \* $ranks_per_node)
export OMP_NUM_THREADS=1  # Max hardware threads = 4
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

####amr_wind_exec=/projects/hfm/gvijayak/exawind/install.master/intel/amr-wind/bin/amr_wind

echo "Job name       = $SLURM_JOB_NAME"
echo "Num. nodes     = $SLURM_JOB_NUM_NODES"
echo "Num. MPI Ranks = $mpi_ranks"
echo "Num. threads   = $OMP_NUM_THREADS"
echo "Working dir    = $PWD"

mpirun -np ${mpi_ranks} amr_wind tke_decay_ref_sst_u_12_tke_0p85_sdr_150.inp  &> u_12_tke_0p85_sdr_150.log 
