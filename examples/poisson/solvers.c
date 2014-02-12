#include "solvers.h"
#include <math.h>
#include <stdlib.h>
#include "blaslapack.h"
#include <stdio.h>

void MxVdispl(Vector y, const Matrix A, const Vector x,
              double alpha, double beta, int ydispl)
{
  char trans='N';
  dgemv(&trans, &A->rows, &A->cols, &alpha, A->data[0], &A->rows, x->data,
        &x->stride, &beta, y->data+ydispl*y->stride, &y->stride);
}

int GaussJacobi(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector r = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
#pragma omp parallel for schedule(static) private(j)
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*e->data[j];
      }
      r->data[i] = u->data[i]-A->data[i][i]*e->data[i];
      u->data[i] /= A->data[i][i];
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(e);
  freeVector(r);

  return it;
}

int GaussJacobiBlas(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector r = createVector(u->len);
  Matrix* LpU = calloc(sizeof(Matrix), getMaxThreads());
  int* sizes, *displ;
  splitVector(A->rows, getMaxThreads(), &sizes, &displ);
#pragma omp parallel private(i)
  {
    LpU[getCurrentThread()] = subMatrix(A, displ[getCurrentThread()],
                                        sizes[getCurrentThread()], 0, A->cols);
    for (i=0;i<sizes[getCurrentThread()];++i)
      LpU[getCurrentThread()]->data[i+displ[getCurrentThread()]][i] = 0.0;
  }
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
#pragma omp parallel private(i)
    {
      MxVdispl(u, LpU[getCurrentThread()], e, -1.0, 1.0, displ[getCurrentThread()]);
      for (i=0;i<sizes[getCurrentThread()];++i) {
        r->data[i+displ[getCurrentThread()]] = u->data[i+displ[getCurrentThread()]]-
                                                A->data[i+displ[getCurrentThread()]]
                                                       [i+displ[getCurrentThread()]]*
                                               e->data[i+displ[getCurrentThread()]];
        u->data[i+displ[getCurrentThread()]] /= A->data[i+displ[getCurrentThread()]]
                                                       [i+displ[getCurrentThread()]];
      }
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(e);
  freeVector(r);
  for (i=0;i<getMaxThreads();++i)
    freeMatrix(LpU[i]);
  free(LpU);

  return it;
}

int GaussSeidel(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double max = tol+1;
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (max > tol && ++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*v->data[j];
      }
      r->data[i] = u->data[i]-A->data[i][i]*v->data[i];
      u->data[i] /= A->data[i][i];
      v->data[i] = u->data[i];
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(r);
  freeVector(v);

  return it;
}

int GaussSeidelBlas(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector v = createVector(u->len);
  Matrix U  = createMatrix(A->rows, A->cols);
  copyVector(U->as_vec, A->as_vec);
  for (i=0;i<U->rows;++i)
    for (j=0;j<=i;++j)
      U->data[j][i] = 0.0;
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
    MxV(u, U, e, -1.0, 1.0, 'N');
    lutsolve(A, u, 'L');
    copyVector(e, b);
    MxV(e, A, u, -1.0, 1.0, 'N');
    max = maxNorm(e);
  }
  freeVector(b);
  freeVector(e);
  freeVector(v);
  freeMatrix(U);

  return it;
}
