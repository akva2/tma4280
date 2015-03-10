#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

int main(int argc, char** argv)
{
  int i, j, N, flag;
  Vector grid;
  Matrix b, e;
  double time, sum, h;
  int rank, size;
  int mpi_top_coords[2];
  int mpi_top_sizes[2];

  init_app(argc, argv, &rank, &size);

  N=atoi(argv[1]);

  // setup topology
  mpi_top_sizes[0] = mpi_top_sizes[1] = 0;
  MPI_Dims_create(size, 2, mpi_top_sizes);
  int periodic[2] = {0, 0};
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_top_sizes, periodic, 0, &comm);
  MPI_Cart_coords(comm, rank, 2, mpi_top_coords);

  int* size1;
  int* displ1;
  int* size2;
  int* displ2;
  splitVector(N, mpi_top_sizes[0], &size1, &displ1);
  splitVector(N, mpi_top_sizes[1], &size2, &displ2);

  b = createMatrix(size1[mpi_top_coords[0]], size2[mpi_top_coords[1]]);
  for (j=0;j<b->cols;++j)
    for(i=0;i<b->rows;++i)
      b->data[j][i] = (j+displ2[mpi_top_coords[1]])*N+1+(i+displ1[mpi_top_coords[0]]);
  b->glob_rows = N;
  b->glob_cols = N;
  b->as_vec->comm = &comm;

  saveMatrixMPI(b, "meh.asc");

  freeMatrix(b);
  MPI_Comm_free(&comm);

  close_app();
  return 0;
}
