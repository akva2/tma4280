#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <memory.h>

#include "common.h"

#ifndef HAVE_MKL
#define fst fst_
#define fstinv fstinv_
#endif

/* function prototypes */
void fst(double *v, int *n, double *w, int *nn);
void fstinv(double *v, int *n, double *w, int *nn);

Vector createEigenValues(int m)
{ 
  Vector diag = createVector(m);
  for (int i=0; i < m; i++) 
    diag->data[i] = 2.0*(1.0-cos((i+1)*M_PI/(m+1)));

  return diag;
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 2) {
    printf("usage: %s <N> [L]\n",argv[0]);
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

  double h = L/N;
  Vector lambda = createEigenValues(M);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Matrix u = createMatrix(M, M);
  Matrix ut = createMatrix(M, M);
  evalMesh(u->as_vec, grid, grid, poisson_source);
  scaleVector(u->as_vec, h*h);

  int NN = 4*N;
  Vector z    = createVector(NN);

  double time = WallTime();

  for (int j=0; j < M; j++)
    fst(u->data[j], &N, z->data, &NN);

  transposeMatrix(u, ut);

  for (int i=0; i < M; i++)
    fstinv(ut->data[i], &N, z->data, &NN);

  for (int j=0; j < M; j++)
    for (int i=0; i < M; i++)
      ut->data[j][i] /= lambda->data[i]+lambda->data[j];

  for (int i=0; i < M; i++)
    fst(ut->data[i], &N, z->data, &NN);

  transposeMatrix(ut, u);

  for (int j=0; j < M; j++)
    fstinv(u->data[j], &N, z->data, &NN);

  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeMatrix(ut);
  freeVector(grid);
  freeVector(z);
  freeVector(lambda);

  close_app();
  return 0;
}
