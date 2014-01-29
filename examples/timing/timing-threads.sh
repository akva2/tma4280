#!/bin/bash

#PBS -l select=1:ncpus=16:mpiprocs=1
#PBS -lwalltime=1:00:00
#PBS -lpmem=2000MB
#PBS -m ae
#PBS -j oe
#PBS -A nn9282k

workdir=/work/$USER/$PBS_JOBID
mkdir -p $workdir
cd $workdir

module load intelcomp
module load cmkl

mxm_sizes="100 500 1000"
daxpy_sizes="100 1000 10000 100000 1000000 10000000"

# C MxM threaded tests
for size in ${mxm_sizes}
do
  echo "---- C MxM threaded size=${size}----" > mxm_c_par_${size}
  OMP_NUM_THREADS=16 ${PBS_O_WORKDIR}/mxm-O3 ${size} 0 >> mxm_c_par_${size}
done

# Fortran MxM threaded tests
#for size in ${mxm_sizes}
#do
#  echo "---- Fortran MxM -O${opt} size=${size} ----" > mxm_f_par_${size}
#  OMP_NUM_THREADS=16 mpiexec_mpt ${PBS_O_WORKDIR}/mxm-f-O3-${size} >> mxm_f_par_${size}
#done

# BLAS MxM threaded tests
for size in ${mxm_sizes}
do
  echo "---- C MxM threaded size=${size}----" > mxm_blas_par_${size}
  OMP_NUM_THREADS=16 ${PBS_O_WORKDIR}/mxm-O3 ${size} 2 >> mxm_blas_par_${size}
done
