#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <memory.h>

#include "common.h"

typedef struct {
  Matrix A;
} poisson_info_t;

Matrix createPoisson1D(int M)
{
  Matrix result = createMatrix(M, M);
  result->data[0][0] = 2.0;
  result->data[1][0] = -1.0;
  for (int i=1;i<M-1;++i) {
    result->data[i][i] = 2.0;
    result->data[i+1][i] = -1.0;
    result->data[i-1][i] = -1.0;
  }
  result->data[M-1][M-1] = 2.0;
  result->data[M-2][M-1] = -1.0;

  return result;
}

void evaluate(Matrix u, const Matrix v, void* ctx)
{
  poisson_info_t* info = ctx;
  MxM(u, info->A, v, 1.0, 0.0, 'N', 'N');
  MxM(u, v, info->A, 1.0, 1.0, 'N', 'N');
}

typedef void(*eval_t)(Matrix, const Matrix, void* ctx);

void cg(eval_t A, Matrix b, double tolerance, void* ctx)
{
  Matrix r = createMatrix(b->rows, b->cols);
  Matrix p = createMatrix(b->rows, b->cols);
  Matrix buffer = createMatrix(b->rows, b->cols);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r->as_vec,b->as_vec);
  fillVector(b->as_vec, 0.0);
  int i=0;
  while (i < b->as_vec->len && rdr > tolerance) {
    ++i;
    if (i == 1) {
      copyVector(p->as_vec,r->as_vec);
      dotp = innerproduct(r->as_vec,r->as_vec);
    } else {
      double dotp2 = innerproduct(r->as_vec,r->as_vec);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p->as_vec,beta);
      axpy(p->as_vec,r->as_vec,1.0);
    }
    A(buffer,p,ctx);
    double alpha = dotp/innerproduct(p->as_vec,buffer->as_vec);
    axpy(b->as_vec,p->as_vec,alpha);
    axpy(r->as_vec,buffer->as_vec,-alpha);
    rdr = sqrt(innerproduct(r->as_vec,r->as_vec));
  }
  printf("%i iterations\n",i);
  freeMatrix(r);
  freeMatrix(p);
  freeMatrix(buffer);
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
  poisson_info_t ctx;
  ctx.A = createPoisson1D(M);

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Matrix u = createMatrix(M, M);
  evalMesh(u->as_vec, grid, grid, poisson_source);
  scaleVector(u->as_vec, h*h);

  double time = WallTime();
  cg(evaluate, u, 1.e-6, &ctx);

  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeVector(grid);
  freeMatrix(ctx.A);

  close_app();
  return 0;
}
