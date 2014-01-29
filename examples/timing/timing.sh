#!/bin/bash

#PBS -l select=1:ncpus=16:mpiprocs=16
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
module load mpt

mxm_sizes="1500"
daxpy_sizes="100 1000 10000 100000 1000000 10000000"
opts="0 1 2 3"

# Standard C MxM tests
for opt in ${opts}
do
  for size in ${mxm_sizes}
  do
    echo "---- C MxM -O${opt} size=${size}----" > mxm_c_${opt}_${size}
    mpiexec_mpt ${PBS_O_WORKDIR}/mxm-O${opt} ${size} 0 >> mxm_c_${opt}_${size}
  done
done

# Standard Fortran MxM tests
for opt in ${opts}
do
  for size in ${mxm_sizes}
  do
    echo "---- Fortran MxM -O${opt} size=${size} ----" > mxm_f_${opt}_${size}
    mpiexec_mpt ${PBS_O_WORKDIR}/mxm-f-O${opt}-${size} >> mxm_f_${opt}_${size}
  done
done

# Fortran unrolled tests
for opt in ${opts} 
do
  echo "---- Fortran unrolled MxM -O${opt}  ----" > mxm_f_unr_${opt}
  mpiexec_mpt ${PBS_O_WORKDIR}/mxm-unr-f-O${opt} >> mxm_f_unr_${opt}
done

# C unrolled tests
for opt in ${opts} 
do
  echo "---- C unrolled MxM -O${opt}  ----" > mxm_c_unr_${opt}
  mpiexec_mpt ${PBS_O_WORKDIR}/mxm-O${opt} 10 1 >> mxm_c_unr_${opt}
done

# BLAS tests
for size in ${mxm_sizes}
do
  echo "---- BLAS MxM size=${size} ----" > mxm_blas_${size}
  mpiexec_mpt ${PBS_O_WORKDIR}/mxm-O3 ${size} 2 >> mxm_blas_${size}
done

# C daxpy tests
for opt in ${opts}
do
  for size in ${daxpy_sizes}
  do
    echo "---- C daxpy -O${opt} size=${size}----" > daxpy_c_${opt}_${size}
    mpiexec_mpt ${PBS_O_WORKDIR}/daxpy-O${opt} ${size} 0 >> daxpy_c_${opt}_${size}
  done
done

# Fortran daxpy tests
for opt in ${opts}
do
  for size in ${daxpy_sizes}
  do
    echo "---- Fortran daxpy -O${opt} size=${size} ----" > daxpy_f_${opt}_${size}
    mpiexec_mpt ${PBS_O_WORKDIR}/daxpy-f-O${opt}-${size} >> daxpy_f_${opt}_${size}
  done
done

# BLAS daxpy tests
for size in ${daxpy_sizes}
do
  echo "---- BLAS daxpy size=${size}----" > daxpy_blas_${size}
  mpiexec_mpt ${PBS_O_WORKDIR}/daxpy-O3 ${size} 1 >> daxpy_blas_${size}
done
