#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

double source(double x)
{
  return 4*M_PI*M_PI*sin(2*M_PI*x);
}

double exact(double x)
{
  return sin(2*M_PI*x);
}

int main(int argc, char** argv)
{
  int i, j, N, flag;
  Matrix A;
  Vector b, grid, e;
  double time, sum;

  if (argc < 3) {
    printf("need two parameters, N and flag\n");
    return 1;
  }
  N=atoi(argv[1]);
  flag=atoi(argv[2]);
  if (N < 0) {
    printf("invalid problem size given\n");
    return 2;
  }

  if (flag < 0 || flag > 2) {
    printf("invalid flag given\n");
    return 3;
  }

  A = createMatrix(N-1,N-1);
  diag(A, -1, -1);
  diag(A, 0, 2);
  diag(A, 1, -1);

  grid = equidistantMesh(0.0, 1.0, N);
  b = createVector(N-1);
  e = createVector(N-1);
  evalMeshInternal(b, grid, source);
  scaleVector(b, pow(grid->data[1]-grid->data[0], 2));

  time = WallTime();

  if (flag == 1) {
    int* ipiv=NULL;
    lusolve(A, b, &ipiv);
    free(ipiv);
  } else
    llsolve(A,b);

  printf("elapsed: %f\n", WallTime()-time);

  evalMeshInternal(e, grid, exact);
  axpy(b,e,-1.0);

  printf("max error: %e\n", maxNorm(b));
  
  freeMatrix(A);
  freeVector(grid);
  freeVector(b);
  return 0;
}
