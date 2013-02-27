#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

typedef struct {
  double mu;
} poisson_info_t;

void precondition(const Vector v, Vector u, void* ctx)
{
  poisson_info_t* info = ctx;
  fillVector(u, 0.0);
  axpy(v, u, 1.0/(4.0+info->mu));
}

void evaluate(const Vector v, Vector u, void* ctx)
{
  poisson_info_t* info = ctx;
  int M = sqrt(v->len);
#pragma omp parallel for schedule(static)
  for (int i=0;i<M;++i) {
    int cnt=i*M;
    for (int j=0;j<M;++j, ++cnt) {
      u->data[cnt] = (4.0+info->mu)*v->data[cnt];
      if (j > 0)
        u->data[cnt] -= v->data[cnt-1];
      if (j < M-1)
        u->data[cnt] -= v->data[cnt+1];
      if (i > 0)
        u->data[cnt] -= v->data[cnt-M];
      if (i < M-1)
        u->data[cnt] -= v->data[cnt+M];
    }
  }
}

typedef void(*eval_t)(const Vector, Vector, void* ctx);

void pcg(eval_t A, eval_t pre, Vector b, double tolerance, void* ctx)
{
  Vector r = createVector(b->len);
  Vector p = createVector(b->len);
  Vector z = createVector(b->len);
  Vector buffer = createVector(b->len);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(b,r);
  double rl = innerproduct(b,r);
  fillVector(b, 0.0);
  int i=0;
  while (i < b->len && rdr/rl > tolerance) {
    pre(r,z,ctx);
    ++i;
    if (i == 1) {
      copyVector(z,p);
      dotp = innerproduct(r,z);
    } else {
      double dotp2 = innerproduct(r,z);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(z,p,1.0);
    }
    A(p, buffer, ctx);
    double alpha = dotp/innerproduct(p,buffer);
    axpy(p,b,alpha);
    axpy(buffer,r,-alpha);
    rdr = sqrt(innerproduct(r,r));
  }
  printf("%i iterations\n",i);
  freeVector(r);
  freeVector(p);
  freeVector(z);
  freeVector(buffer);
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 2) {
    printf("usage: %s <N> [L] [mu]\n",argv[0]);
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
  double mu=0.01;
  if (argc > 3)
    mu = atof(argv[3]);

  double h = L/N;

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  Vector u = createVector(M*M);
  evalMesh(u, grid, grid, poisson_source);
  evalMesh2(u, grid, grid, exact_solution, mu);
  scaleVector(u, h*h);

  poisson_info_t ctx;
  ctx.mu = h*h*mu;

  double time = WallTime();
  pcg(evaluate, precondition, u, 1.e-6, &ctx);

  evalMesh2(u, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeVector(u);
  freeVector(grid);

  close_app();
  return 0;
}
