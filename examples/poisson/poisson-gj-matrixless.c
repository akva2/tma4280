#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void GJ(Matrix u, double tolerance, int maxit, double mu)
{
  int it=0;
  Matrix b = cloneMatrix(u);
  Matrix e = cloneMatrix(u);
  Matrix v = cloneMatrix(u);
  copyVector(b->as_vec, u->as_vec);
  fillVector(u->as_vec, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it <= maxit) {
    copyVector(e->as_vec, u->as_vec);
    copyVector(v->as_vec, u->as_vec);
    copyVector(u->as_vec, b->as_vec);
    collectMatrix(v);
#pragma omp parallel for schedule(static)
    for (int i=1;i<u->cols-1;++i) {
      for (int j=1;j<u->rows-1;++j) {
        u->data[i][j] += v->data[i][j-1];
        u->data[i][j] += v->data[i][j+1];
        u->data[i][j] += v->data[i+1][j];
        u->data[i][j] += v->data[i-1][j];
        u->data[i][j] /= (4.0+mu);
      }
    }
    axpy(e->as_vec, u->as_vec, -1.0);
    max = sqrt(innerproduct(e->as_vec, e->as_vec));
  }
  if (u->as_vec->comm_rank == 0)
    printf("number of iterations %i %f\n", it, max);
  freeMatrix(b);
  freeMatrix(v);
  freeMatrix(e);
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 2) {
    printf("usage: %s <N> [L] [mu]\n",argv[0]);
    close_app();
    return 1;
  }

  /* the total number of grid points in each spatial direction is (N+1) */
  /* the total number of degrees-of-freedom in each spatial direction is (N-1) */
  int N  = atoi(argv[1]);
  int M  = N-1;
  double L=1;
  if (argc > 2)
    L = atof(argv[2]);
  double mu=0.01;
  if (argc > 3)
    mu = atof(argv[3]);

  double h = L/N;

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  int coords[2] = {0};
  int sizes[2] = {1};
#ifdef HAVE_MPI
  sizes[0] = sizes[1] = 0;
  MPI_Dims_create(size,2,sizes);
  int periodic[2];
  periodic[0] = periodic[1] = 0;
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD,2,sizes,periodic,0,&comm);
  MPI_Cart_coords(comm,rank,2,coords);
#endif

  int* len[2];
  int* displ[2];
  splitVector(M, sizes[0], &len[0], &displ[0]);
  splitVector(M, sizes[1], &len[1], &displ[1]);

#ifdef HAVE_MPI
  Matrix u = createMatrixMPI(len[0][coords[0]]+2, len[1][coords[1]]+2, M, M, &comm);
#else
  Matrix u = createMatrix(M+2, M+2);
#endif
  evalMeshDispl(u, grid, grid, poisson_source,
                displ[0][coords[0]], displ[1][coords[1]]);
  evalMesh2Displ(u, grid, grid, exact_solution, mu,
                displ[0][coords[0]], displ[1][coords[1]]);
  scaleVector(u->as_vec, h*h);

  double time = WallTime();
  GJ(u, 1e-6, 10000, h*h*mu);

  evalMesh2Displ(u, grid, grid, exact_solution, -1.0,
                 displ[0][coords[0]], displ[1][coords[1]]);

  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeVector(grid);
  for (int i=0;i<2;++i) {
    free(len[i]);
    free(displ[i]);
  }

  close_app();
  return 0;
}
