#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "blaslapack.h"
#include "common.h"
#include "solvers.h"
#include "poissoncommon.h"

double alpha=0.0;
double tol=1e-6;

double source(double x, double y)
{
  return -30.0*pow(y,4)*x*(pow(x,5.0)-1)-30.0*pow(x,4)*y*(pow(y,5)-1);
}

double exact(double x, double y)
{
  return x*(pow(x,5)-1.0)*y*(pow(y,5)-1.0);
}

void mask(Matrix u)
{
  fillVector(u->col[0], 0.0);
  fillVector(u->col[u->cols-1], 0.0);
  fillVector(u->row[0], 0.0);
  fillVector(u->row[u->rows-1], 0.0);
}

int GaussJacobiPoisson2D(Matrix u, double tol, int maxit)
{
  int it=0, i, j;
  Matrix b = cloneMatrix(u);
  Matrix e = cloneMatrix(u);
  copyVector(b->as_vec, u->as_vec);
  fillVector(u->as_vec, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e->as_vec, u->as_vec);
    collectMatrix(e);
    copyVector(u->as_vec, b->as_vec);
#pragma omp parallel for schedule(static) private(j)
    for (i=1;i<e->cols-1;++i) {
      for (j=1;j<e->rows-1;++j) {
        u->data[i][j] += e->data[i-1][j];
        u->data[i][j] += e->data[i+1][j];
        u->data[i][j] += e->data[i][j-1];
        u->data[i][j] += e->data[i][j+1];
        u->data[i][j] /= (4.0+alpha);
      }
    }
    axpy(e->as_vec, u->as_vec, -1.0);
    mask(e);
    max = maxNorm(e->as_vec);
  }
  freeMatrix(b);
  freeMatrix(e);

  return it;
}

int GaussSeidelPoisson2DMatrb(Matrix u, double tol, int maxit)
{
  int it=0, i, j, k;
  double max = tol+1;
  double rl = maxNorm(u->as_vec);
  Matrix b = cloneMatrix(u);
  Matrix r = cloneMatrix(u);
  Matrix v = cloneMatrix(u);
  copyVector(b->as_vec, u->as_vec);
  fillVector(u->as_vec, 0.0);
  while (max > tol && ++it < maxit) {
    copyVector(v->as_vec, u->as_vec);
    copyVector(u->as_vec, b->as_vec);
    collectMatrix(v);
    for (k=0;k<2;++k) {
#pragma omp parallel for schedule(static) private(j)
      for (i=1;i<r->cols-1;++i) {
        for (j=1+(i-1+k)%2;j<r->rows-1;j+=2) {
          u->data[i][j] += v->data[i-1][j];
          u->data[i][j] += v->data[i+1][j];
          u->data[i][j] += v->data[i][j-1];
          u->data[i][j] += v->data[i][j+1];
          r->data[i][j] = u->data[i][j]-(4.0+alpha)*v->data[i][j];
          u->data[i][j] /= (4.0+alpha);
          v->data[i][j] = u->data[i][j];
        }
      }
      if (k == 0)
        collectMatrix(v);
    }
    max = maxNorm(r->as_vec);
  }
  freeMatrix(b);
  freeMatrix(r);
  freeMatrix(v);

  return it;
}

void Poisson2D(Matrix u, Matrix v)
{
  int i,j;
  collectMatrix(v);
#pragma omp parallel for schedule(static) private(j)
  for (i=1;i<u->cols-1;++i) {
    for (j=1;j<u->rows-1;++j) {
      u->data[i][j] = (4.0+alpha)*v->data[i][j];
      u->data[i][j] -= v->data[i-1][j];
      u->data[i][j] -= v->data[i+1][j];
      u->data[i][j] -= v->data[i][j-1];
      u->data[i][j] -= v->data[i][j+1];
    }
  }
  mask(u);
  mask(v);
}

void DiagonalizationPoisson2D(Matrix b, const Vector lambda1, const Matrix Q1,
                              const Vector lambda2, const Matrix Q2)
{
  int i,j;
  Matrix ut = createMatrix(b->rows, Q1->rows);
  MxM(ut, b, Q1, 1.0, 0.0, 'N', 'T');
  MxM(b, Q2, ut, 1.0, 0.0, 'N', 'N');
  for (j=0;j<b->cols;++j)
    for (i=0;i<b->rows;++i)
      b->data[j][i] /= (lambda2->data[i]+lambda1->data[j]+alpha);
  MxM(ut, Q2, b, 1.0, 0.0, 'T', 'N');
  MxM(b, ut, Q1, 1.0, 0.0, 'N', 'N');
  freeMatrix(ut);
}

//! \brief A struct describing a subdomain in additive schwarz
typedef struct {
  int size[2];             //!< Size of subdomain
  int from_disp[2];        //!< Left displacement when creating subdomain vector
  int to_source_disp[2];   //!< Left displacement from source recreating full vector
  int to_dest_disp[2];     //!< Left displacement in destination recreating full vector
  int to_dest_size[2];     //!< Datas to copy to destinatino recreating full vector
  Matrix Q[2];             //!< Eigenvectors of Poisson operator
  Vector lambda[2];        //!< Eigenvalues of Poisson operator
} SchwarzSubdomain;

SchwarzSubdomain subdomain;

Matrix collectBeforePre(Matrix u)
{
  int j;
  collectMatrix(u);
  Matrix result=createMatrix(subdomain.size[1], subdomain.size[0]);
  for (j=0;j<result->cols;++j) {
    dcopy(&subdomain.size[1], u->data[j+subdomain.from_disp[0]]+subdomain.from_disp[1],
          &u->col[0]->stride, result->data[j], &result->col[0]->stride);
  }

  return result;
}

void collectAfterPre(Matrix u, const Matrix v)
{
  int i,j;
  int source, dest, source2, dest2;

  for (j=0;j<subdomain.to_dest_size[0];++j) {
    dcopy(&subdomain.to_dest_size[1],
          v->data[j+subdomain.to_source_disp[0]]+subdomain.to_source_disp[1],
          &v->col[j]->stride,
          u->data[j+subdomain.to_dest_disp[0]]+subdomain.to_dest_disp[1],
          &u->col[j]->stride);
  }
  // east
  MPI_Cart_shift(*u->as_vec->comm, 0,   1, &source, &dest);
  MPI_Sendrecv(v->data[v->cols-1], v->rows, MPI_DOUBLE, dest,   0,
               u->data[0]+1, v->rows, MPI_DOUBLE, source, 0,
               *u->as_vec->comm, MPI_STATUS_IGNORE);

  if (source != MPI_PROC_NULL)
    axpy(u->col[1], u->col[0], 1.0);

  Vector sendBuf = createVector(v->cols);
  Vector recvBuf = createVector(v->cols);

  // south
  MPI_Cart_shift(*u->as_vec->comm, 1, 1, &source2, &dest2);
  if (dest2 != MPI_PROC_NULL)
    copyVector(sendBuf, v->row[v->rows-1]);

  MPI_Sendrecv(sendBuf->data, sendBuf->len, MPI_DOUBLE, dest2, 3,
               recvBuf->data, recvBuf->len, MPI_DOUBLE, source2, 3,
               *u->as_vec->comm, MPI_STATUS_IGNORE);
  if (source2 != MPI_PROC_NULL) {
    double alpha=1.0;
    int ext=(source!=MPI_PROC_NULL?1:0);
    int len=recvBuf->len-1;
    daxpy(&len, &alpha, recvBuf->data+ext, &recvBuf->stride, u->row[1]->data+(1+ext)*u->row[1]->stride, &u->row[1]->stride);
  }

  mask(u);
}

void Poisson2DPre(Matrix u, Matrix v)
{
  Matrix tmp = collectBeforePre(v);
  mask(v);
  if (subdomain.Q[0])
    DiagonalizationPoisson2D(tmp, subdomain.lambda[0], subdomain.Q[0],
                                  subdomain.lambda[1], subdomain.Q[1]);

  collectAfterPre(u, tmp);
  freeMatrix(tmp);
}

int main(int argc, char** argv)
{
  int i, j, N, flag;
  Vector grid;
  Matrix b, e;
  double time, sum, h;
  int rank, size;
  int mpi_top_coords[2];
  int mpi_top_sizes[2];

  init_app(argc, argv, &rank, &size);

  if (argc < 3) {
    printf("need two parameters, N and flag [alpha] [tolerance]\n");
    printf(" - N is the problem size (in each direction\n");
    printf(" - flag = 1  -> Matrix-free Gauss-Jacobi iterations\n");
    printf(" - flag = 2  -> Matrix-free red-black Gauss-Seidel iterations\n");
    printf(" - flag = 3  -> Matrix-free CG iterations\n");
    printf(" - flag = 4  -> Matrix-free additive schwarz preconditioned+diagonalization CG iterations\n");
    printf(" - alpha is the Helmholtz scaling factor\n");
    printf(" - tolerance is the residual error tolerance in the iterative scheme\n");
    return 1;
  }
  N=atoi(argv[1]);
  flag=atoi(argv[2]);
  if (argc > 3)
    alpha = atof(argv[3]);
  if (argc > 4)
    tol=atof(argv[4]);

  if (N < 0) {
    if (rank == 0)
      printf("invalid problem size given\n");
    close_app();
    return 2;
  }

  if (flag < 0 || flag > 4) {
    if (rank == 0)
      printf("invalid flag given\n");
    close_app();
    return 3;
  }

  if (flag == 2 && (N-1)%2 != 0 && ((N-1)/size) % 2 != 0) {
    if (rank == 0)
      printf("need an even size (per process) for red-black iterations\n");
    close_app();
    return 4;
  }

  // setup topology
  mpi_top_sizes[0] = mpi_top_sizes[1] = 0;
  MPI_Dims_create(size, 2, mpi_top_sizes);
  int periodic[2] = {0, 0};
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_top_sizes, periodic, 0, &comm);
  MPI_Cart_coords(comm, rank, 2, mpi_top_coords);

  b = createMatrixMPICart(N+1, N+1, &comm, 1);
  e = createMatrixMPICart(N+1, N+1, &comm, 1);

  // setup subdomain
  subdomain.Q[0] = subdomain.Q[1] = NULL;
  subdomain.lambda[0] = subdomain.lambda[1] = NULL;
  for (i=0;i<2;++i) {
    if (mpi_top_coords[i] == 0) {
      int dec=1;
      if (mpi_top_sizes[i] == 1)
        dec=2;
      subdomain.size[i] = subdomain.to_dest_size[i] = (i==0?b->cols:b->rows)-dec;
      if (mpi_top_sizes[i] != 1)
        subdomain.to_dest_size[i]--;
      subdomain.from_disp[i] = 1;
      subdomain.to_source_disp[i] = 0;
      subdomain.to_dest_disp[i] = 1;
    } else {
      subdomain.size[i] = (i==0?b->cols:b->rows)-(1+(mpi_top_coords[i] == mpi_top_sizes[i-1]?1:0));
      subdomain.to_dest_size[i] = subdomain.size[i];
      subdomain.from_disp[i] = 1;
      subdomain.to_source_disp[i] = 0; subdomain.to_dest_disp[i] = 1;
    }
  }

  grid = equidistantMesh(0.0, 1.0, N);
  h = 1.0/N;

  evalMesh2Displ(b, grid, source, mpi_top_coords);
  scaleVector(b->as_vec, pow(h, 2));
  evalMesh2Displ(e, grid, exact, mpi_top_coords);
  axpy(b->as_vec, e->as_vec, alpha);
  mask(b);

  if (flag == 4) {
    subdomain.lambda[0] = generateEigenValuesP1D(subdomain.size[0]);
    subdomain.lambda[1] = generateEigenValuesP1D(subdomain.size[1]);
    subdomain.Q[0] = generateEigenMatrixP1D(subdomain.size[0]);;
    subdomain.Q[1] = generateEigenMatrixP1D(subdomain.size[1]);;
  }

  int its=-1;
  char method[128];
  time = WallTime();
  if (flag == 1) {
    its=GaussJacobiPoisson2D(b, tol, 1000000);
    sprintf(method,"Gauss-Jacobi");
  }
  if (flag == 2) {
    its=GaussSeidelPoisson2DMatrb(b, tol, 1000000);
    sprintf(method,"Gauss-Seidel");
  }
  if (flag == 3) {
    its=cgMatrixFreeMat(Poisson2D, b, tol);
    sprintf(method,"CG");
  }
  if (flag == 4 || flag == 5) {
    its=pcgMatrixFreeMat(Poisson2D, Poisson2DPre, b, tol);
    sprintf(method,"PCG");
  }
  if (rank == 0) {
    printf("%s used %i iterations\n", method, its);
    printf("elapsed: %f\n", WallTime()-time);
  }

  axpy(b->as_vec, e->as_vec,-1.0);
  mask(b);

  h = maxNorm(b->as_vec);
  if (rank == 0)
    printf("max error: %e\n", h);
  
  freeVector(grid);
  freeMatrix(b);
  freeMatrix(e);

  MPI_Comm_free(&comm);

  close_app();
  return 0;
}
