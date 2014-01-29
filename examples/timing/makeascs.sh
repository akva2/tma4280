#!/bin/bash

mxm_sizes="100 500 1000 1500"
daxpy_sizes="100 1000 10000 100000 1000000 10000000"
opts="0 1 2 3"

echo $mxm_sizes > mxm_sizes.asc
echo $daxpy_sizes > daxpy_sizes.asc

# C and Fortran MxM
for opt in ${opts}
do
  rm -f mxm_c_${opt}.asc
  rm -f mxm_f_${opt}.asc
  for size in ${mxm_sizes}
  do
    entries=""
    while read line
    do
      entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
      entries+=' '
    done < mxm_c_${opt}_${size}
    echo $entries >> mxm_c_${opt}.asc
    entries=""
    while read line
    do
      entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
      entries+=' '
    done < mxm_f_${opt}_${size}
    echo $entries >> mxm_f_${opt}.asc
  done
done

# C and Fortran unrolled MxM
for opt in ${opts}
do
  rm -f mxm_c_unr_${opt}.asc
  rm -f mxm_f_unr_${opt}.asc
  entries=""
  while read line
  do
    entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
    entries+=' '
  done < mxm_c_unr_${opt}
  echo $entries >> mxm_c_unr_${opt}.asc
  entries=""
  while read line
  do
    entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
    entries+=' '
  done < mxm_f_unr_${opt}
  echo $entries >> mxm_f_unr_${opt}.asc
done

# BLAS MxM
rm -f mxm_blas.asc
for size in ${mxm_sizes}
do
  entries=""
  while read line
  do
    entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
    entries+=' '
  done < mxm_blas_${size}
  echo $entries >> mxm_blas.asc
done

# C and Fortran daxpy
for opt in ${opts}
do
  rm -f daxpy_c_${opt}.asc
  rm -f daxpy_f_${opt}.asc
  for size in ${daxpy_sizes}
  do
    entries=""
    while read line
    do
      entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
      entries+=' '
    done < daxpy_c_${opt}_${size}
    echo $entries >> daxpy_c_${opt}.asc
    entries=""
    while read line
    do
      entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
      entries+=' '
    done < daxpy_f_${opt}_${size}
    echo $entries >> daxpy_f_${opt}.asc
  done
done

# BLAS daxpy
rm -f daxpy_blas.asc
for size in ${daxpy_sizes}
do
  entries=""
  while read line
  do
    entries+=`echo $line |grep 'r= ' | awk -F 'r= ' '{print $2}'`
    entries+=' '
  done < daxpy_blas_${size}
  echo $entries >> daxpy_blas.asc
done
