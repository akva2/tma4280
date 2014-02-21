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
  double rl;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector r = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
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
  double rl;
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
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
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
  double rl;
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
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
  double rl;
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
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
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

int cg(Matrix A, Vector b, double tolerance)
{
  int i=0, j;
  double rl;
  Vector r = createVector(b->len);
  Vector p = createVector(b->len);
  Vector buffer = createVector(b->len);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r,b);
  fillVector(b, 0.0);
  int* sizes, *displ;
  splitVector(A->rows, getMaxThreads(), &sizes, &displ);
  Matrix* Ablock = malloc(getMaxThreads()*sizeof(Matrix));
#pragma omp parallel
  {
    Ablock[getCurrentThread()] = subMatrix(A, displ[getCurrentThread()],
                                           sizes[getCurrentThread()], 0, A->cols);
  }
  rl = sqrt(dotproduct(r,r));
  while (i < b->len && rdr > tolerance*rl) {
    ++i;
    if (i == 1) {
      copyVector(p,r);
      dotp = dotproduct(r,r);
    } else {
      double dotp2 = dotproduct(r,r);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(p,r,1.0);
    }
#pragma omp parallel
    {
      MxVdispl(buffer, Ablock[getCurrentThread()], p, 1.0, 0.0,
               displ[getCurrentThread()]);
    }
    double alpha = dotp/dotproduct(p,buffer);
    axpy(b,p,alpha);
    axpy(r,buffer,-alpha);
    rdr = sqrt(dotproduct(r,r));
  }
  freeVector(r);
  freeVector(p);
  freeVector(buffer);
  for (j=0;j<getMaxThreads();++j)
    freeMatrix(Ablock[j]);
  free(Ablock);

  return i;
}

int cgMatrixFree(MatVecFunc A, Vector b, double tolerance)
{
  int it=0;
  double rl;
  Vector r = cloneVector(b);
  Vector p = cloneVector(b);
  Vector buffer = cloneVector(b);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r,b);
  fillVector(b, 0.0);
  rl = sqrt(dotproduct(r,r));
  while (it < b->glob_len && rdr > tolerance*rl) {
    ++it;
    if (it == 1) {
      copyVector(p,r);
      dotp = dotproduct(r,r);
    } else {
      double dotp2 = dotproduct(r,r);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(p,r,1.0);
    }
    A(buffer, p);
    double alpha = dotp/dotproduct(p,buffer);
    axpy(b,p,alpha);
    axpy(r,buffer,-alpha);
    rdr = sqrt(dotproduct(r,r));
  }
  freeVector(r);
  freeVector(p);
  freeVector(buffer);

  return it;
}

int cgMatrixFreeMat(MatMatFunc A, Matrix b, double tolerance)
{
  int it=0;
  double rl;
  Matrix r = cloneMatrix(b);
  Matrix p = cloneMatrix(b);
  Matrix buffer = cloneMatrix(b);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r->as_vec, b->as_vec);
  fillVector(b->as_vec, 0.0);
  rl = sqrt(dotproduct(r->as_vec, r->as_vec));
  while (it < b->as_vec->glob_len && rdr > tolerance*rl) {
    ++it;
    if (it == 1) {
      copyVector(p->as_vec, r->as_vec);
      dotp = dotproduct(r->as_vec, r->as_vec);
    } else {
      double dotp2 = dotproduct(r->as_vec, r->as_vec);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p->as_vec, beta);
      axpy(p->as_vec, r->as_vec, 1.0);
    }
    A(buffer, p);
    double alpha = dotp/dotproduct(p->as_vec, buffer->as_vec);
    axpy(b->as_vec, p->as_vec, alpha);
    axpy(r->as_vec, buffer->as_vec, -alpha);
    rdr = sqrt(dotproduct(r->as_vec, r->as_vec));
  }
  freeMatrix(r);
  freeMatrix(p);
  freeMatrix(buffer);

  return it;
}

int pcgMatrixFree(MatVecFunc A, MatVecFunc pre, Vector b, double tolerance)
{
  int it=0;
  double rl;
  Vector r = cloneVector(b);
  Vector p = cloneVector(b);
  Vector z = cloneVector(b);
  Vector buffer = cloneVector(b);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r,b);
  fillVector(b, 0.0);
  rl = sqrt(dotproduct(r,r));
  while (it < b->glob_len && rdr > tolerance*rl) {
    pre(z,r);
    ++it;
    if (it == 1) {
      copyVector(p,z);
      dotp = dotproduct(r,z);
    } else {
      double dotp2 = dotproduct(r,z);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(p,z,1.0);
    }
    A(buffer, p);
    double alpha = dotp/dotproduct(p,buffer);
    axpy(b,p,alpha);
    axpy(r,buffer,-alpha);
    rdr = sqrt(dotproduct(r,r));
  }
  freeVector(r);
  freeVector(p);
  freeVector(z);
  freeVector(buffer);

  return it;
}

int pcgMatrixFreeMat(MatMatFunc A, MatMatFunc pre, Matrix b, double tolerance)
{
  int it=0;
  double rl;
  Matrix r = cloneMatrix(b);
  Matrix p = cloneMatrix(b);
  Matrix z = cloneMatrix(b);
  Matrix buffer = cloneMatrix(b);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r->as_vec, b->as_vec);
  fillVector(b->as_vec, 0.0);
  rl = sqrt(dotproduct(r->as_vec, r->as_vec));
  while (it < b->as_vec->glob_len && rdr > tolerance*rl) {
    pre(z,r);
    ++it;
    if (it == 1) {
      copyVector(p->as_vec, z->as_vec);
      dotp = dotproduct(r->as_vec, z->as_vec);
    } else {
      double dotp2 = dotproduct(r->as_vec, z->as_vec);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p->as_vec, beta);
      axpy(p->as_vec, z->as_vec, 1.0);
    }
    A(buffer, p);
    double alpha = dotp/dotproduct(p->as_vec, buffer->as_vec);
    axpy(b->as_vec, p->as_vec, alpha);
    axpy(r->as_vec, buffer->as_vec, -alpha);
    rdr = sqrt(dotproduct(r->as_vec, r->as_vec));
  }
  freeMatrix(r);
  freeMatrix(p);
  freeMatrix(z);
  freeMatrix(buffer);

  return it;
}
