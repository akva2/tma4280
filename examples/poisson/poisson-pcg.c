#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

typedef struct {
  Vector lambda1x;
  Matrix Q1x;
  Vector lambda2x;
  Matrix Q2x;
  Vector lambda1y;
  Matrix Q1y;
  Vector lambda2y;
  Matrix Q2y;
  int subs;
  int* xsizes;
  int* xdispl;
  int* ysizes;
  int* ydispl;
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
      sub[i*info->subs+j] = subMatrix(v, 1+info->xdispl[i], info->xsizes[i],
                                         1+info->ydispl[j], info->ysizes[j]);
    }
  }
/*#pragma omp parallel for schedule(static)*/
  for (int l=0;l<info->subs*info->subs;++l) {
    Matrix Qx = (l/info->subs==info->subs-1?info->Q2x:info->Q1x);
    Matrix Qy = (l%info->subs==info->subs-1?info->Q2y:info->Q1y);
    Vector lambdax = (l/info->subs==info->subs-1?info->lambda2x:info->lambda1x);
    Vector lambday = (l%info->subs==info->subs-1?info->lambda2y:info->lambda1y);

    Matrix ut = createMatrix(sub[l]->rows, Qy->cols);
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
          u->data[1+info->ydispl[j]+c][1+info->xdispl[i]+r] += 
              sub[i*info->subs+j]->data[c][r];
      freeMatrix(sub[i*info->subs+j]);
    }
  }

  free(sub);
  ddsum(u);
  mask(u);
}

void evaluate(Matrix u, const Matrix v, void* ctx)
{
  Matrix t = cloneMatrix(v);
  copyVector(t->as_vec, v->as_vec);
  collectMatrix(t);
/*#pragma omp parallel for schedule(static)*/
  for (int i=1;i<v->cols-1;++i) {
    for (int j=1;j<v->rows-1;++j) {
      u->data[i][j]  = 4.0*t->data[i][j];
      u->data[i][j] -= t->data[i][j-1];
      u->data[i][j] -= t->data[i][j+1];
      u->data[i][j] -= t->data[i-1][j];
      u->data[i][j] -= t->data[i+1][j];
    }
  }
  mask(u);
  freeMatrix(t);
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
    printf("rdr %f\n",rdr);
  }
  if (b->as_vec->comm_rank == 0)
    printf("%i iterations\n",i);
  freeMatrix(r);
  freeMatrix(p);
  freeMatrix(z);
  freeMatrix(buffer);
  printf("gonyo\n");
}

void setupOverlap(int* sizes, int* displ, int subs, int total)
{
  if (sizes[0] == sizes[subs-1]) {
    for (int i=0;i<subs-1;++i)
      sizes[i]++;
  } else {
    for (int i=0;i<subs-1;++i)
      displ[i+1] = displ[i]+sizes[0];
    for (int i=0;i<subs-1;++i)
      sizes[i] = sizes[subs-1];
    sizes[subs-1] = total-displ[subs-1];
  }
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
  ctx.subs = 1;
  if (argc > 2)
    ctx.subs = atoi(argv[2]);
  double L=1.0;
  if (argc > 3)
    L = atof(argv[3]);

  double h = L/N;

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  int coords[2] = {0,0};
  int sizes[2] = {1,1};
#ifdef HAVE_MPI
  sizes[0] = sizes[1] = 0;
  MPI_Dims_create(size,2,sizes);
  int periodic[2];
  periodic[0] = periodic[1] = 0;
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD,2,sizes,periodic,0,&comm);
  MPI_Cart_coords(comm,rank,2,coords);
#endif

  int* len[2];
  int* displ[2];
  splitVector(M, sizes[0], &len[0], &displ[0]);
  splitVector(M, sizes[1], &len[1], &displ[1]);

#ifdef HAVE_MPI
  Matrix u = createMatrixMPI(len[0][coords[0]]+2, len[1][coords[1]]+2, M, M, &comm);
/*  evalMeshDispl(u, grid, grid, poisson_source,*/
/*                displ[0][coords[0]], displ[1][coords[1]]);*/
#else
  Matrix u = createMatrix(N+1, N+1);
  evalMesh(u->as_vec, grid, grid, poisson_source);
#endif
  fillVector(u->as_vec, 1.0);
  scaleVector(u->as_vec, h*h);

  printf("r %i c %i\n", u->rows, u->cols);

  splitVector(u->rows-2, ctx.subs, &ctx.xsizes, &ctx.xdispl);
  splitVector(u->cols-2, ctx.subs, &ctx.ysizes, &ctx.ydispl);
  setupOverlap(ctx.xsizes, ctx.xdispl, ctx.subs, u->rows-2);
  setupOverlap(ctx.ysizes, ctx.ydispl, ctx.subs, u->cols-2);

  ctx.lambda1x = createEigenValues(ctx.xsizes[0]);
  ctx.Q1x = createEigenMatrix(ctx.xsizes[0]);
  ctx.lambda2x = createEigenValues(ctx.xsizes[ctx.subs-1]);
  ctx.Q2x = createEigenMatrix(ctx.xsizes[ctx.subs-1]);

  ctx.lambda1y = createEigenValues(ctx.ysizes[0]);
  ctx.Q1y = createEigenMatrix(ctx.ysizes[0]);
  ctx.lambda2y = createEigenValues(ctx.ysizes[ctx.subs-1]);
  ctx.Q2y = createEigenMatrix(ctx.ysizes[ctx.subs-1]);

  double time = WallTime();
  mask(u);
  pcg(evaluate, precondition, u, 1.e-6, &ctx);

#if HAVE_MPI
/*  evalMesh2Displ(u, grid, grid, exact_solution, -1.0,*/
/*                 displ[0][coords[0]], displ[1][coords[1]]);*/
#else
  evalMesh2(u->as_vec, grid, grid, exact_solution, -1.0);
#endif

  printf("maxyo\n");
  double max = maxNorm(u->as_vec);
  printf("maxed yo\n");

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  printf("hereyo\n");
  freeMatrix(u);
  freeVector(grid);
  free(ctx.xdispl);
  free(ctx.xsizes);
  free(ctx.ydispl);
  free(ctx.ysizes);
  free(len[0]);
  free(len[1]);
  free(displ[0]);
  free(displ[1]);
  freeMatrix(ctx.Q1x);
  freeVector(ctx.lambda1x);
  freeMatrix(ctx.Q2x);
  freeVector(ctx.lambda2x);
  freeMatrix(ctx.Q1y);
  freeVector(ctx.lambda1y);
  freeMatrix(ctx.Q2y);
  freeVector(ctx.lambda2y);

  close_app();
  return 0;
}
