#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"
#include "solvers.h"
#include "poissoncommon.h"

double alpha=0.1;

double exact(double x)
{
  return x*(pow(x,5.0)-1.0);
}

double source(double x)
{
  return -30*pow(x,4.0);
}


int GaussJacobiPoisson1D(Vector u, double tol, int maxit)
{
  int it=0, i;
  double rl;
  double max = tol+1;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
#pragma omp parallel for schedule(static)
    for (i=0;i<e->len;++i) {
      if (i > 0)
        u->data[i] += e->data[i-1];
      if (i < e->len-1)
        u->data[i] += e->data[i+1];
      u->data[i] /= (2.0+alpha);
    }
    axpy(e, u, -1.0);
    max = maxNorm(e);
  }
  freeVector(b);
  freeVector(e);

  return it;
}

int GaussSeidelPoisson1D(Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double max = tol+1;
  double rl = maxNorm(u);
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (max > tol*rl && ++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (i=0;i<r->len;++i) {
      if (i > 0)
        u->data[i] += v->data[i-1];
      if (i < r->len-1)
        u->data[i] += v->data[i+1];
      r->data[i] = u->data[i]-(2.0+alpha)*v->data[i];
      u->data[i] /= (2.0+alpha);
      v->data[i] = u->data[i];
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(r);
  freeVector(v);

  return it;
}

int GaussSeidelPoisson1Drb(Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double max = tol+1;
  double rl = maxNorm(u);
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (max > tol && ++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (j=0;j<2;++j) {
#pragma omp parallel for schedule(static)
      for (i=j;i<r->len;i+=2) {
        if (i > 0)
          u->data[i] += v->data[i-1];
        if (i < r->len-1)
          u->data[i] += v->data[i+1];
        r->data[i] = u->data[i]-(2.0+alpha)*v->data[i];
        u->data[i] /= (2.0+alpha);
        v->data[i] = u->data[i];
      }
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(r);
  freeVector(v);

  return it;
}

void DiagonalizationPoisson1D(Vector u, const Vector lambda, const Matrix Q)
{
  Vector btilde = createVector(u->len);
  int i;
  MxV(btilde, Q, u, 1.0, 0.0, 'T');
  for (i=0;i<btilde->len;++i)
    btilde->data[i] /= lambda->data[i];
  MxV(u, Q, btilde, 1.0, 0.0, 'N');
  freeVector(btilde);
}

void DiagonalizationPoisson1Dfst(Vector u, const Vector lambda)
{
  Vector btilde = createVector(u->len);
  Vector buf = createVector(4*(u->len+1));
  int i;
  int N=u->len+1;
  int NN=4*N;
  copyVector(btilde, u);
  fst(btilde->data, &N, buf->data, &NN);
  for (i=0;i<btilde->len;++i)
    btilde->data[i] /= (lambda->data[i]+alpha);
  fstinv(btilde->data, &N, buf->data, &NN);
  copyVector(u, btilde);
  freeVector(btilde);
  freeVector(buf);
}

void Poisson1D(Vector u, const Vector v)
{
  int i;
#pragma omp parallel for schedule(static)
  for (i=0;i<u->len;++i) {
    u->data[i] = (2.0+alpha)*v->data[i];
    if (i > 0)
      u->data[i] -= v->data[i-1];
    if (i < u->len-1)
      u->data[i] -= v->data[i+1];
  }
}

int main(int argc, char** argv)
{
  int i, j, N, flag;
  Matrix A=NULL, Q=NULL;
  Vector b, grid, e, lambda=NULL;
  double time, sum, h, tol=1e-4;

  if (argc < 3) {
    printf("need two parameters, N and flag [and tolerance]\n");
    printf(" - N is the problem size (in each direction\n");
    printf(" - flag = 1  -> Dense LU\n");
    printf(" - flag = 2  -> Dense Cholesky\n");
    printf(" - flag = 3  -> Full Gauss-Jacobi iterations\n");
    printf(" - flag = 4  -> Full Gauss-Jacobi iterations using BLAS\n");
    printf(" - flag = 5  -> Full Gauss-Seidel iterations\n");
    printf(" - flag = 6  -> Full Gauss-Seidel iterations using BLAS\n");
    printf(" - flag = 7  -> Full CG iterations\n");
    printf(" - flag = 8  -> Matrix-less Gauss-Jacobi iterations\n");
    printf(" - flag = 9  -> Matrix-less Gauss-Seidel iterations\n");
    printf(" - flag = 10 -> Matrix-less Red-Black Gauss-Seidel iterations\n");
    printf(" - flag = 11 -> Diagonalization\n");
    printf(" - flag = 12 -> Diagonalization - FST\n");
    printf(" - flag = 13 -> Matrix-less CG iterations\n");
    return 1;
  }
  N=atoi(argv[1]);
  flag=atoi(argv[2]);
  if (argc > 3)
    tol = atof(argv[3]);
  if (N < 0) {
    printf("invalid problem size given\n");
    return 2;
  }

  if (flag < 0 || flag > 13) {
    printf("invalid flag given\n");
    return 3;
  }

  if (flag == 10 && (N-1)%2 != 0) {
    printf("need an even size for red-black iterations\n");
    return 4;
  }
  if (flag == 12 && (N & (N-1)) != 0) {
    printf("need a power-of-two for fst-based diagonalization\n");
    return 5;
  }

  h = 1.0/N;

  grid = equidistantMesh(0.0, 1.0, N);
  b = createVector(N-1);
  e = createVector(N-1);
  evalMeshInternal(b, grid, source);
  evalMeshInternal(e, grid, exact);
  scaleVector(b, pow(h, 2));
  axpy(b, e, alpha);

  if (flag < 8) {
    A = createMatrix(N-1,N-1);
    diag(A, -1, -1.0);
    diag(A, 0, 2.0+alpha);
    diag(A, 1, -1.0);
  }

  if (flag >= 11 && flag < 13)
    lambda = generateEigenValuesP1D(N-1);
  if (flag == 11)
    Q = generateEigenMatrixP1D(N-1);

  time = WallTime();

  if (flag == 1) {
    int* ipiv=NULL;
    lusolve(A, b, &ipiv);
    free(ipiv);
  } else if (flag == 2)
    llsolve(A,b,0);
  else if (flag == 3)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobi(A, b, tol, 10000000));
  else if (flag == 4)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobiBlas(A, b, tol, 10000000));
  else if (flag == 5)
    printf("Gauss-Seidel used %i iterations\n",
           GaussSeidel(A, b, tol, 10000000));
  else if (flag == 6)
    printf("Gauss-Seidel used %i iterations\n",
           GaussSeidelBlas(A, b, tol, 10000000));
  else if (flag == 7)
    printf("CG used %i iterations\n", cg(A, b, 1e-8));
  else if (flag == 8)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussJacobiPoisson1D(b, tol, 10000000));
  else if (flag == 9)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussSeidelPoisson1D(b, tol, 10000000));
  else if (flag == 10)
    printf("Gauss-Jacobi used %i iterations\n",
           GaussSeidelPoisson1Drb(b, tol, 10000000));
  else if (flag == 11)
           DiagonalizationPoisson1D(b,lambda,Q);
  else if (flag == 12)
           DiagonalizationPoisson1Dfst(b,lambda);
  else if (flag == 13)
    printf("CG used %i iterations\n", cgMatrixFree(Poisson1D, b, tol));

  printf("elapsed: %f\n", WallTime()-time);

  evalMeshInternal(e, grid, exact);
  axpy(b,e,-1.0);

  printf("max error: %e\n", maxNorm(b));
  
  if (A)
    freeMatrix(A);
  if (Q)
    freeMatrix(Q);
  freeVector(grid);
  freeVector(b);
  freeVector(e);
  if (lambda)
    freeVector(lambda);
  return 0;
}
