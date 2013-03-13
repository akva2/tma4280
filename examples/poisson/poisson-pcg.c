#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

typedef struct {
  Vector lambda1;
  Matrix Q1;
  Vector lambda2;
  Matrix Q2;
  int subs;
  int* sizes;
  int* displ;
} poisson_info_t;

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

void mask(Matrix u)
{
  fillVector(u->col[0], 0.0);
  fillVector(u->col[u->cols-1], 0.0);
  fillVector(u->row[0], 0.0);
  fillVector(u->row[u->rows-1], 0.0);
}

void precondition(Matrix u, const Matrix v, void* ctx)
{
  poisson_info_t* info = ctx;
  Matrix* sub = malloc(info->subs*info->subs*sizeof(Matrix));
  fillVector(u->as_vec,0.0);
  for (int i=0;i<info->subs;++i) {
    for (int j=0;j<info->subs;++j) {
      sub[i*info->subs+j] = subMatrix(v, 1+info->displ[i], info->sizes[i],
                                      1+info->displ[j], info->sizes[j]);
    }
  }
#pragma omp parallel for schedule(static)
  for (int l=0;l<info->subs*info->subs;++l) {
    Matrix Qx = (l/info->subs==info->subs-1?info->Q2:info->Q1);
    Matrix Qy = (l%info->subs==info->subs-1?info->Q2:info->Q1);
    Matrix ut = createMatrix(sub[l]->rows, Qy->cols);
    Vector lambdax = (l/info->subs==info->subs-1?info->lambda2:info->lambda1);
    Vector lambday = (l%info->subs==info->subs-1?info->lambda2:info->lambda1);
    MxM(ut, sub[l], Qy, 1.0, 0.0, 'N', 'N');
    MxM(sub[l], Qx, ut, 1.0, 0.0, 'T', 'N');

    for (int j=0; j < sub[l]->cols; j++)
      for (int i=0; i < sub[l]->rows; i++)
        sub[l]->data[j][i] /= lambdax->data[i]+lambday->data[j];

    MxM(ut, sub[l], Qy, 1.0, 0.0, 'N', 'T');
    MxM(sub[l], Qx, ut, 1.0, 0.0, 'N', 'N');
    freeMatrix(ut);
  }
  for (int i=0;i<info->subs;++i) {
    for (int j=0;j<info->subs;++j) {
      for (int c=0;c<sub[i*info->subs+j]->cols;++c)
        for (int r=0;r<sub[i*info->subs+j]->rows;++r)
          u->data[1+info->displ[j]+c][1+info->displ[i]+r] += 
              sub[i*info->subs+j]->data[c][r];
      freeMatrix(sub[i*info->subs+j]);
    }
  }

  free(sub);
  mask(u);
}

void evaluate(Matrix u, const Matrix v, void* ctx)
{
  poisson_info_t* info = ctx;
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

typedef void(*eval_t)(Matrix, const Matrix, void*);

void pcg(eval_t A, eval_t pre, Matrix b, double tolerance, void* ctx)
{
  Matrix r = cloneMatrix(b);
  Matrix p = cloneMatrix(b);
  Matrix z = cloneMatrix(b);
  Matrix buffer = cloneMatrix(b);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r->as_vec,b->as_vec);
  double rl = innerproduct(r->as_vec,r->as_vec);
  fillVector(b->as_vec, 0.0);
  int i=0;
  while (i < b->as_vec->len && rdr/rl > tolerance) {
    pre(z,r,ctx);
    ++i;
    if (i == 1) {
      copyVector(p->as_vec,z->as_vec);
      dotp = innerproduct(r->as_vec,z->as_vec);
    } else {
      double dotp2 = innerproduct(r->as_vec,z->as_vec);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p->as_vec,beta);
      axpy(p->as_vec,z->as_vec,1.0);
    }
    A(buffer, p, ctx);
    double alpha = dotp/innerproduct(p->as_vec,buffer->as_vec);
    axpy(b->as_vec,p->as_vec,alpha);
    axpy(r->as_vec,buffer->as_vec,-alpha);
    rdr = sqrt(innerproduct(r->as_vec,r->as_vec));
  }
  printf("%i iterations\n",i);
  freeMatrix(r);
  freeMatrix(p);
  freeMatrix(z);
  freeMatrix(buffer);
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 2) {
    printf("usage: %s <N> [subdomains] [L]\n",argv[0]);
    close_app();
    return 1;
  }

  /* the total number of grid points in each spatial direction is (N+1) */
  /* the total number of degrees-of-freedom in each spatial direction is (N-1) */
  int N  = atoi(argv[1]);
  int M  = N-1;
  poisson_info_t ctx;
  ctx.subs = 2;
  if (argc > 2)
    ctx.subs = atoi(argv[2]);
  double L=1.0;
  if (argc > 3)
    L = atof(argv[3]);

  double h = L/N;

  Vector grid = createVector(N+1);
  for (int i=0;i<N+1;++i)
    grid->data[i] = i*h;

  Matrix u = createMatrix(N+1, N+1);
  evalMesh(u->as_vec, grid, grid, poisson_source);
  fillVector(u->as_vec, 1.0);
  scaleVector(u->as_vec, h*h);

  splitVector(u->rows-2, ctx.subs, &ctx.sizes, &ctx.displ);
  if (ctx.sizes[0] == ctx.sizes[ctx.subs-1]) {
    for (int i=0;i<ctx.subs-1;++i)
      ctx.sizes[i]++;
  } else {
    for (int i=0;i<ctx.subs-1;++i)
      ctx.displ[i+1] = ctx.displ[i]+ctx.sizes[0];
    for (int i=0;i<ctx.subs-1;++i)
      ctx.sizes[i] = ctx.sizes[ctx.subs-1];
    ctx.sizes[ctx.subs-1] = u->rows-2-ctx.displ[ctx.subs-1];
  }

  ctx.lambda1 = createEigenValues(ctx.sizes[0]);
  ctx.Q1 = createEigenMatrix(ctx.sizes[0]);
  ctx.lambda2 = createEigenValues(ctx.sizes[ctx.subs-1]);
  ctx.Q2 = createEigenMatrix(ctx.sizes[ctx.subs-1]);

  double time = WallTime();
  mask(u);
  pcg(evaluate, precondition, u, 1.e-6, &ctx);

  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeVector(grid);
  free(ctx.displ);
  free(ctx.sizes);
  freeMatrix(ctx.Q1);
  freeVector(ctx.lambda1);
  freeMatrix(ctx.Q2);
  freeVector(ctx.lambda2);

  close_app();
  return 0;
}
