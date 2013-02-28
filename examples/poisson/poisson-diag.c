#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <memory.h>

#include "common.h"

Vector createEigenValues(int m)
{ 
  Vector diag = createVector(m);
  for (int i=0; i < m; i++) 
    diag->data[i] = 2.0*(1.0-cos((i+1)*M_PI/(m+1)));

  return diag;
}

Matrix createEigenMatrix(int m)
{ 
  Matrix Q = createMatrix(m,m);
  for (int i=0;i<m;++i)
    for (int j=0;j<m;++j)
      Q->data[j][i] = sqrt(2.0/(m+1))*sin((i+1)*(j+1)*M_PI/(m+1));

  return Q;
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
  Vector lambda = createEigenValues(M);
  Matrix Q = createEigenMatrix(M);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Matrix u = createMatrix(M, M);
  Matrix ut = createMatrix(M, M);
  evalMesh(u->as_vec, grid, grid, poisson_source);
  scaleVector(u->as_vec, h*h);

  double time = WallTime();
  MxM(ut, u, Q, 1.0, 0.0);
  MxM(u, Q, ut, 1.0, 0.0);

  for (int j=0; j < M; j++)
    for (int i=0; i < M; i++)
      u->data[j][i] /= lambda->data[i]+lambda->data[j];

  MxM(ut, u, Q, 1.0, 0.0);
  MxM(u, Q, ut, 1.0, 0.0);

  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeMatrix(ut);
  freeVector(grid);
  freeVector(lambda);
  freeMatrix(Q);

  close_app();
  return 0;
}
