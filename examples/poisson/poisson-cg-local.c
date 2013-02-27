#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

void mask(Matrix u)
{
  fillVector(u->col[0], 0.0);
  fillVector(u->col[u->cols-1], 0.0);
  fillVector(u->row[0], 0.0);
  fillVector(u->row[u->rows-1], 0.0);
}

void evaluate(const Matrix v, Matrix u)
{
#pragma omp parallel for schedule(static)
  for (int i=1;i<v->cols-1;++i) {
    for (int j=1;j<v->rows-1;++j) {
      u->data[i][j]  = 4.0*v->data[i][j];
      u->data[i][j] -= v->data[i][j-1];
      u->data[i][j] -= v->data[i][j+1];
      u->data[i][j] -= v->data[i-1][j];
      u->data[i][j] -= v->data[i+1][j];
    }
  }
  mask(u);
}

typedef void(*eval_t)(const Matrix, Matrix);

void cg(eval_t A, Matrix b, double tolerance)
{
  Matrix r = createMatrix(b->rows, b->cols);
  Matrix p = createMatrix(b->rows, b->cols);
  Matrix buffer = createMatrix(b->rows, b->cols);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(b->as_vec,r->as_vec);
  fillVector(b->as_vec, 0.0);
  int i=0;
  while (i < b->as_vec->len && rdr > tolerance) {
    ++i;
    if (i == 1) {
      copyVector(r->as_vec,p->as_vec);
      dotp = innerproduct(r->as_vec,r->as_vec);
    } else {
      double dotp2 = innerproduct(r->as_vec,r->as_vec);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p->as_vec,beta);
      axpy(r->as_vec,p->as_vec,1.0);
    }
    A(p,buffer);
    double alpha = dotp/innerproduct(p->as_vec,buffer->as_vec);
    axpy(p->as_vec,b->as_vec,alpha);
    axpy(buffer->as_vec,r->as_vec,-alpha);
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

  Vector grid = createVector(M+2);
  for (int i=0;i<M+2;++i)
    grid->data[i] = i*h;

  Matrix u = createMatrix(M+2, M+2);
  evalMesh(u->as_vec, grid, grid, poisson_source);
  scaleVector(u->as_vec, h*h);
  mask(u);

  double time = WallTime();
  cg(evaluate, u, 1.e-6);

  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeVector(grid);

  close_app();
  return 0;
}
