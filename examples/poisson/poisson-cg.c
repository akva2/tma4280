#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <memory.h>

#include "common.h"

void cg(Matrix A, Vector b, double tolerance)
{
  Vector r = createVector(b->len);
  Vector p = createVector(b->len);
  Vector buffer = createVector(b->len);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r,b);
  fillVector(b, 0.0);
  int i=0;
  while (i < b->len && rdr > tolerance) {
    ++i;
    if (i == 1) {
      copyVector(p,r);
      dotp = innerproduct(r,r);
    } else {
      double dotp2 = innerproduct(r,r);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(p,r,1.0);
    }
    MxV(buffer, A, p, 1.0, 0.0);
    double alpha = dotp/innerproduct(p,buffer);
    axpy(b,p,alpha);
    axpy(r,buffer,-alpha);
    rdr = sqrt(innerproduct(r,r));
  }
  printf("%i iterations\n",i);
  freeVector(r);
  freeVector(p);
  freeVector(buffer);
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

#ifdef HAVE_MPI
  Matrix A = createPoisson2DMPI(M, 0.0);
  Vector u = createVectorMPI(M*M, 1, &WorldComm);
#else
  Matrix A = createPoisson2D(M, 0.0);
  Vector u = createVector(M*M);
#endif
  Vector grid = createVector(M);

  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  evalMesh(u, grid, grid, poisson_source);
  scaleVector(u, h*h);

  double time = WallTime();
  cg(A, u, 1.e-6);

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
