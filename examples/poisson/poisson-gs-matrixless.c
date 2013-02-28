#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void GS(Vector u, double tolerance, int maxit)
{
  int it=0;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector v = createVector(u->len);
  int M = sqrt(u->len);
  int* sizes, *displ;
  splitVector(M, 2*max_threads(), &sizes, &displ);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
    for (int color=0;color<2;++color) {
      for (int i=0;i<M;++i) {
#pragma omp parallel
        {
          int cnt=i*M+displ[get_thread()*2+color];
          for (int j=0;j<sizes[get_thread()*2+color];++j, ++cnt) {
            if (j+displ[get_thread()*2+color] > 0)
              u->data[cnt] += v->data[cnt-1];
            if (j+displ[get_thread()*2+color] < M-1)
              u->data[cnt] += v->data[cnt+1];
            if (i > 0)
              u->data[cnt] += v->data[cnt-M];
            if (i < M-1)
              u->data[cnt] += v->data[cnt+M];
            u->data[cnt] /= 4.0;
            v->data[cnt] = u->data[cnt];
          }
        }
      }
    }
    axpy(e, u, -1.0);
    max = sqrt(innerproduct(e, e));
  }
  printf("number of iterations %i %f\n", it, max);
  freeVector(b);
  freeVector(e);
  freeVector(v);
  free(sizes);
  free(displ);
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

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  scaleVector(u, h*h);

  double time = WallTime();
  GS(u, 1e-6, 5000);

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
