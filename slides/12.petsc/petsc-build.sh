PETSC_DIR=/home/akva/kode/petsc-3.4.2
PETSC_ARCH=linux-gnu-cxx-opt
./configure --with-precision=double 
--with-scalar-type=real 
--with-debugging=0 
--COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 
--FCOPTFLAGS=-O3 
--with-blas-lib=/usr/lib/libblas.a
--with-lapack-lib=/usr/lib/liblapack.a 
--with-64-bit-indices=0 --with-clanguage=c++
--with-mpi=1 --LIBS=-ldl
--with-hypre=1 --with-umfpack=1
--with-ml=1
--with-superlu=1
