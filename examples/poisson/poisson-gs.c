#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void GS(Matrix A, Vector u, double tolerance, int maxit)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(u, b);
  fillVector(u, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(u, e);
    for (int i=0;i<A->rows;++i) {
      u->data[i] = b->data[i];
      for (int j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*v->data[j];
      }
      u->data[i] /= A->data[i][i];
      v->data[i] = u->data[i];
    }
    axpy(u, e, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  printf("number of iterations %i %f\n", it, max);
  freeVector(b);
  freeVector(e);
  freeVector(v);
}

void GShog(Matrix A, Vector u, double tolerance, int maxit)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector v = createVector(u->len);
  Matrix U  = createMatrix(A->rows, A->cols);
  copyVector(A->as_vec, U->as_vec);
  for (int i=0;i<U->rows;++i)
    for (int j=0;j<=i;++j)
      U->data[j][i] = 0.0;
  copyVector(u, b);
  fillVector(u, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(u, e);
    copyVector(b, u);
    MxV(u, U, e, -1.0, 1.0);
    lutsolve(A, u, 'L');
    axpy(u, e, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  printf("number of iterations %i %f\n", it, max);
  freeVector(b);
  freeVector(e);
  freeVector(v);
  freeMatrix(U);
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
  double L=1.0;
  if (argc > 2)
    L = atof(argv[2]);

  double h = L/N;

  Matrix A = createPoisson2D(M, 0.0);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  scaleVector(u, h*h);

  double time = WallTime();
  GShog(A, u, 1e-6, 1000);

  evalMesh2(u, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeVector(u);
  freeVector(grid);
  freeMatrix(A);

  close_app();
  return 0;
}
