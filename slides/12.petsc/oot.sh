mkdir serial
cd serial
PETSC_ARCH=linux-gnu-cxx-opt cmake ..
make
cd ..
mkdir MPI
cd MPI
PETSC_ARCH=linux-gnu-cxx-opt-mpi cmake ..
           -DENABLE_MPI=1
make
