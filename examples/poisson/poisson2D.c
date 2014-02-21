#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"
#include "solvers.h"
#include "poissoncommon.h"

double source(double x, double y)
{
  return -30.0*pow(y,4)*x*(pow(x,5.0)-1)-30.0*pow(x,4)*y*(pow(y,5)-1);
}

double exact(double x, double y)
{
  return x*(pow(x,5)-1.0)*y*(pow(y,5)-1.0);
}

int GaussJacobiPoisson2DVec(Vector u, double tol, int maxit)
{
  int it=0, i, j, k;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  int M = sqrt(e->len);
  while (max > tol && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
#pragma omp parallel for schedule(static) private(j, k)
    for (i=0;i<M;++i) {
      k = i*M;
      for (j=0;j<M;++j, ++k) {
        if (j > 0)
          u->data[k] += e->data[k-1];
        if (j < M-1)
          u->data[k] += e->data[k+1];
        if (i > 0)
          u->data[k] += e->data[k-M];
        if (i < M-1)
          u->data[k] += e->data[k+M];
        u->data[k] /= 4.0;
      }
    }
    axpy(e, u, -1.0);
    max = maxNorm(e);
  }
  freeVector(b);
  freeVector(e);

  return it;
}

int GaussJacobiPoisson2DMat(Matrix u, double tol, int maxit)
{
  int it=0, i, j, k;
  Matrix b = createMatrix(u->rows, u->cols);
  Matrix e = createMatrix(u->rows, u->cols);
  copyVector(b->as_vec, u->as_vec);
  fillVector(u->as_vec, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e->as_vec, u->as_vec);
    copyVector(u->as_vec, b->as_vec);
#pragma omp parallel for schedule(static) private(i)
    for (j=1;j<u->cols-1;++j) {
      for (i=1;i<u->rows-1;++i) {
        u->data[j][i] += e->data[j][i-1];
        u->data[j][i] += e->data[j][i+1];
        u->data[j][i] += e->data[j+1][i];
        u->data[j][i] += e->data[j-1][i];
        u->data[j][i] /= 4.0;
      }
    }
    axpy(e->as_vec, u->as_vec, -1.0);
    max = maxNorm(e->as_vec);
  }
  freeMatrix(b);
  freeMatrix(e);

  return it;
}

void DiagonalizationPoisson2D(Matrix b, const Vector lambda, const Matrix Q)
{
  int i,j;
  Matrix ut = createMatrix(b->rows, b->cols);
  MxM(ut, b, Q, 1.0, 0.0, 'T', 'N');
  MxM(b, Q, ut, 1.0, 0.0, 'N', 'N');
  for (j=0;j<b->cols;++j)
    for (i=0;i<b->rows;++i)
      b->data[j][i] /= lambda->data[i]+lambda->data[j];
  MxM(ut, b, Q, 1.0, 0.0, 'N', 'T');
  MxM(b, Q, ut, 1.0, 0.0, 'N', 'N');
  freeMatrix(ut);
}

void DiagonalizationPoisson2Dfst(Matrix b, const Vector lambda)
{
  int i,j;
  Matrix ut = createMatrix(b->rows, b->cols);
  Vector buf = createVector(4*(b->rows+1));
  int N=b->rows+1;
  int NN=4*N;

  for (i=0;i<b->cols;++i)
    fst(b->data[i], &N, buf->data, &NN);
  transposeMatrix(ut, b);
  for (i=0;i<ut->cols;++i)
    fstinv(ut->data[i], &N, buf->data, &NN);

  for (j=0;j<b->cols;++j)
    for (i=0;i<b->rows;++i)
      ut->data[j][i] /= lambda->data[i]+lambda->data[j];

  for (i=0;i<b->cols;++i)
    fst(ut->data[i], &N, buf->data, &NN);
  transposeMatrix(b, ut);
  for (i=0;i<ut->cols;++i)
    fstinv(b->data[i], &N, buf->data, &NN);

  freeMatrix(ut);
  freeVector(buf);
}

void Poisson2D(Vector u, const Vector v)
{
  int M=sqrt(v->len);
  int i, j, k;
#pragma omp parallel for schedule(static) private(j, k)
    for (i=0;i<M;++i) {
      k = i*M;
      for (j=0;j<M;++j, ++k) {
        u->data[k] = 4.0*v->data[k];
        if (j > 0)
          u->data[k] -= v->data[k-1];
        if (j < M-1)
          u->data[k] -= v->data[k+1];
        if (i > 0)
          u->data[k] -= v->data[k-M];
        if (i < M-1)
          u->data[k] -= v->data[k+M];
      }
    }
}

int main(int argc, char** argv)
{
  int i, j, N, flag, local;
  Matrix A=NULL, Q=NULL;
  Matrix b, e;
  Vector grid, lambda=NULL;
  double time, sum, h;

  if (argc < 3) {
    printf("need two parameters, N and flag\n");
    printf(" - N is the problem size (in each direction\n");
    printf(" - flag = 1  -> Dense LU\n");
    printf(" - flag = 2  -> Dense Cholesky\n");
    printf(" - flag = 3  -> Full Gauss-Jacobi iterations\n");
    printf(" - flag = 4  -> Full Gauss-Jacobi iterations using BLAS\n");
    printf(" - flag = 5  -> Full Gauss-Seidel iterations\n");
    printf(" - flag = 6  -> Full Gauss-Seidel iterations using BLAS\n");
    printf(" - flag = 7  -> Full CG iterations using BLAS\n");
    printf(" - flag = 8  -> Matrix-less Gauss-Jacobi iterations\n");
    printf(" - flag = 9  -> Matrix-less Gauss-Jacobi iterations, local data structure\n");
    printf(" - flag = 10 -> Matrix-less Red-Black Gauss-Jacobi iterations, local data structure\n");
    printf(" - flag = 11 -> Diagonalization\n");
    printf(" - flag = 12 -> Diagonalization, fst based\n");
    printf(" - flag = 13  -> Matrix-free CG iterations\n");
    return 1;
  }
  N=atoi(argv[1]);
  flag=atoi(argv[2]);
  if (N < 0) {
    printf("invalid problem size given\n");
    return 2;
  }

  if (flag < 0 || flag > 13) {
    printf("invalid flag given\n");
    return 3;
  }

  local = (flag==9 || flag == 10);

  grid = equidistantMesh(0.0, 1.0, N);
  if (local) {
    b = createMatrix(N+1,N+1);
    e = createMatrix(N+1,N+1);
  } else {
    b = createMatrix(N-1,N-1);
    e = createMatrix(N-1,N-1);
  }
  evalMeshInternal2(b, grid, source, local);
  h = grid->data[1]-grid->data[0];
  scaleVector(b->as_vec, pow(h, 2));

  if (flag < 8) {
    A = createMatrix((N-1)*(N-1),(N-1)*(N-1));
    diag(A, -1, -1);
    diag(A, 0, 4.0);
    diag(A, 1, -1);
    diag(A, N-1, -1);
    diag(A, -(N-1), -1);
    for (i=N-2;i<(N-1)*(N-1)-1;i+=N-1) {
      A->data[i+1][i] = 0.0;
      A->data[i][i+1] = 0.0;
    }
  }

  if (flag >= 11 && flag < 13)
    lambda = generateEigenValuesP1D(N-1);
  if (flag == 11)
    Q = generateEigenMatrixP1D(N-1);

  time = WallTime();

  if (flag == 1) {
    int* ipiv=NULL;
    lusolve(A, b->as_vec, &ipiv);
    free(ipiv);
  } else if (flag == 2)
    llsolve(A,b->as_vec,0);
  else if (flag == 3)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobi(A, b->as_vec, 1e-8, 1000000));
  else if (flag == 4)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobiBlas(A, b->as_vec, 1e-8, 1000000));
  else if (flag == 5)
    printf("Gauss-Seidel used %i iterations\n",
           GaussSeidel(A, b->as_vec, 1e-8, 1000000));
  else if (flag == 6)
    printf("Gauss-Seidel used %i iterations\n",
           GaussSeidelBlas(A, b->as_vec, 1e-8, 1000000));
  else if (flag == 7)
    printf("CG used %i iterations\n", cg(A, b->as_vec, 1e-8));
  else if (flag == 8)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobiPoisson2DVec(b->as_vec, 1e-8, 1000000));
  else if (flag == 9)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobiPoisson2DMat(b, 1e-8, 1000000));
  else if (flag == 11)
           DiagonalizationPoisson2D(b, lambda, Q);
  else if (flag == 12)
           DiagonalizationPoisson2Dfst(b, lambda);
  else if (flag == 13)
    printf("CG used %i iterations\n", cgMatrixFree(Poisson2D, b->as_vec, 1e-8));

  printf("elapsed: %f\n", WallTime()-time);

  evalMeshInternal2(e, grid, exact, local);
  axpy(b->as_vec,e->as_vec,-1.0);

  printf("max error: %e\n", maxNorm(b->as_vec));

  if (A)
    freeMatrix(A);
  if (Q)
    freeMatrix(Q);
  freeMatrix(b);
  freeMatrix(e);
  freeVector(grid);
  if (lambda)
    freeVector(lambda);
  return 0;
}
