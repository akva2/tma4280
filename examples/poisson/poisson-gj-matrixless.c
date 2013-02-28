#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void GJ(Vector u, double tolerance, int maxit, double mu)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tolerance+1;
  int M = sqrt(u->len);
  while (max > tolerance && ++it < maxit) {
    copyVector(e, u);
    copyVector(v, u);
    copyVector(u, b);
#pragma omp parallel for schedule(static)
    for (int i=0;i<M;++i) {
      int cnt = i*M;
      for (int j=0;j<M;++j, ++cnt) {
        if (j > 0)
          u->data[cnt] += v->data[cnt-1];
        if (j < M-1)
          u->data[cnt] += v->data[cnt+1];
        if (i > 0)
          u->data[cnt] += v->data[cnt-M];
        if (i < M-1)
          u->data[cnt] += v->data[cnt+M];
        u->data[cnt] /= (4.0+mu);
      }
    }
    axpy(e, u, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  printf("number of iterations %i %f\n", it, max);
  freeVector(b);
  freeVector(v);
  freeVector(e);
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

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  evalMesh2(u, grid, grid, exact_solution, mu);
  scaleVector(u, h*h);

  double time = WallTime();
  GJ(u, 1e-6, 10000, h*h*mu);

  evalMesh2(u, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeVector(u);
  freeVector(grid);

  close_app();
  return 0;
}
