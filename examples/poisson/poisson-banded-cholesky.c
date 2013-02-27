#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

int dpbtrf_(char *uplo, int *n, int* ku, double *a, int* lda, int *info);
int dpbtrs_(char *uplo, int *n, int* ku, int *nrhs, double*a, int* lda,
            double* b, int* ldb, int* info);

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

Matrix makeBanded(Matrix A, int m, int kl, int ku)
{
  Matrix AB = createMatrix(kl+kl+ku+1,m);
  for (int i=0;i<m;++i) {
    for (int j=0;j<m;++j) {
      int lb = max(0,j-ku);
      int ub = min(m-1,j+kl);
      if (i >= lb && i <= ub)
        AB->data[j][kl+ku+i-j] = A->data[j][i];
    }
  }

  return AB;
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

  Matrix AF = createMatrix(M*M,M*M);
  for (int i=0;i<M;++i) {
    for (int j=0;j<M;++j) {
      AF->data[i*M+j][i*M+j] = 4.0;
      if (j > 0)
        AF->data[i*M+j][i*M+j-1] = -1.0;
      if (j < M-1)
        AF->data[i*M+j][i*M+j+1] = -1.0;
      if (i < M-1)
        AF->data[i*M+j][(i+1)*M+j] = -1.0;
      if (i > 0)
        AF->data[i*M+j][(i-1)*M+j] = -1.0;
    }
  }

  Matrix A = makeBanded(AF, M*M, 0, M);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  scaleVector(u, h*h);

  double time = WallTime();
  int MM=M*M;
  int* ipiv = malloc(MM*sizeof(int));
  int one=1;
  int info;
  int lda=M+1;
  char uplo='U';
  dpbtrf_(&uplo, &MM, &M, A->data[0], &lda, &info);
  dpbtrs_(&uplo, &MM, &M, &one, A->data[0], &lda, u->data, &MM, &info);

  evalMesh2(u, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeVector(u);
  freeVector(grid);
  freeMatrix(A);
  freeMatrix(AF);
  free(ipiv);

  close_app();
  return 0;
}
