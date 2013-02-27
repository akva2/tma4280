#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void GJ(Matrix A, Vector u, double tolerance, int maxit)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  copyVector(u, b);
  fillVector(u, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(u, e);
    copyVector(b, u);
#pragma omp parallel for schedule(static)
    for (int i=0;i<A->rows;++i) {
      for (int j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*e->data[j];
      }
      u->data[i] /= A->data[i][i];
    }
    axpy(u, e, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  printf("number of iterations %i %f\n", it, max);
  freeVector(b);
  freeVector(e);
}

void GJhog(Matrix A, Vector u, double tolerance, int maxit)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Matrix* LpU = malloc(max_threads()*sizeof(Matrix));
  int* sizes, *displ;
  splitVector(A->rows, max_threads(), &sizes, &displ);
#pragma omp parallel
  {
    LpU[get_thread()] = subMatrix(A, displ[get_thread()], sizes[get_thread()],
                                  0, A->cols);
    for (int i=0;i<sizes[get_thread()];++i)
      LpU[get_thread()]->data[i+displ[get_thread()]][i] = 0.0;
  }
  copyVector(u, b);
  fillVector(u, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(u, e);
    copyVector(b, u);
#pragma omp parallel
    {
      MxVdispl(u, LpU[get_thread()], e, -1.0, 1.0,
               sizes[get_thread()], displ[get_thread()]);
      for (int i=0;i<sizes[get_thread()];++i)
        u->data[i+displ[get_thread()]] /= A->data[i+displ[get_thread()]]
                                                 [i+displ[get_thread()]];
    }
    axpy(u, e, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  free(sizes);
  free(displ);
  freeVector(e);
  freeVector(b);
  for (int i=0;i<max_threads();++i)
    freeMatrix(LpU[i]);
  free(LpU);
  printf("number of iterations %i %f\n", it, max);
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
  double L=1.0;
  if (argc > 2)
    L = atof(argv[2]);
  double mu=0.01;
  if (argc > 3)
    mu = atof(argv[3]);

  double h = L/N;

  Matrix A = createPoisson2D(M, h*h*mu);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  evalMesh2(u, grid, grid, exact_solution, mu);
  scaleVector(u, h*h);

  double time = WallTime();
  GJhog(A, u, 1e-6, 1000);

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
