#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include "blaslapack.h"

#ifdef HAVE_MPI
MPI_Comm WorldComm;
MPI_Comm SelfComm;
#endif

void init_app(int argc, char** argv, int* rank, int* size)
{
#ifdef HAVE_MPI
#ifdef HAVE_OPENMP
  int aquired;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &aquired);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  if (*rank == 0) {
    printf("aquired MPI threading level: ");
    if (aquired == MPI_THREAD_SINGLE)
      printf("MPI_THREAD_SINGLE\n");
    if (aquired == MPI_THREAD_FUNNELED)
      printf("MPI_THREAD_FUNNELED\n");
    if (aquired == MPI_THREAD_SERIALIZED)
      printf("MPI_THREAD_SERIALIZED\n");
    if (aquired == MPI_THREAD_MULTIPLE)
      printf("MPI_THREAD_MULTIPLE\n");
  }
#else
  MPI_Init(&argc, &argv);
#endif
  MPI_Comm_size(MPI_COMM_WORLD, size);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_dup(MPI_COMM_WORLD, &WorldComm);
  MPI_Comm_dup(MPI_COMM_SELF, &SelfComm);
#else
  *rank = 0;
  *size = 1;
#endif
}

void close_app()
{
#ifdef HAVE_MPI
  MPI_Comm_free(&WorldComm);
  MPI_Comm_free(&SelfComm);
  MPI_Finalize();
#endif
}

Vector createVector(int len)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->data = calloc(len, sizeof(double));
  result->len = result->glob_len = len;
  result->stride = 1;
#ifdef HAVE_MPI
  result->comm = &SelfComm;
#endif
  result->comm_size = 1;
  result->comm_rank = 0;
  result->displ = NULL;
  result->sizes = NULL;

  return result;
}

#ifdef HAVE_MPI
Vector createVectorMPI(int glob_len, MPI_Comm* comm, int allocdata)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->comm = comm;
  MPI_Comm_size(*comm, &result->comm_size);
  MPI_Comm_rank(*comm, &result->comm_rank);
  splitVector(glob_len, result->comm_size, &result->sizes, &result->displ);
  result->len = result->sizes[result->comm_rank];
  if (allocdata)
    result->data = calloc(result->len, sizeof(double));
  else
    result->data = NULL;
  result->glob_len = glob_len;
  result->stride = 1;

  return result;
}
#endif

void splitVector(int globLen, int size, int** len, int** displ)
{
  int i;
  *len = calloc(size,sizeof(int));
  *displ = calloc(size,sizeof(int));
  for (i=0;i<size;++i) {
    (*len)[i] = globLen/size;
    if (globLen % size && i >= (size - globLen % size))
      (*len)[i]++;
    if (i < size-1)
      (*displ)[i+1] = (*displ)[i]+(*len)[i];
  }
}

void freeVector(Vector vec)
{
  free(vec->data);
  free(vec->sizes);
  free(vec->displ);
  free(vec);
}

Matrix createMatrix(int n1, int n2)
{
  int i;
  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));
  result->rows = n1;
  result->cols = n2;
  result->data = (double **)calloc(n2   ,sizeof(double *));
  result->data[0] = (double  *)calloc(n1*n2,sizeof(double));
  for (i=1; i < n2; i++)
    result->data[i] = result->data[i-1] + n1;
  result->as_vec = (Vector)malloc(sizeof(vector_t));
  result->as_vec->data = result->data[0];
  result->as_vec->len = n1*n2;
  result->as_vec->stride = 1;
  result->as_vec->comm_size = 1;
  result->as_vec->comm_rank = 0;
  result->col = malloc(n2*sizeof(Vector));
  for (i=0;i<n2;++i) {
    result->col[i] = malloc(sizeof(vector_t));
    result->col[i]->len = n1;
    result->col[i]->stride = 1;
    result->col[i]->data = result->data[i];
  }
  result->row = malloc(n1*sizeof(Vector));
  for (i=0;i<n1;++i) {
    result->row[i] = malloc(sizeof(vector_t));
    result->row[i]->len = n2;
    result->row[i]->stride = n1;
    result->row[i]->data = result->data[0]+i;
  }

  return result;
}

#ifdef HAVE_MPI
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm)
{
  int i, n12;

  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));
  result->as_vec = createVectorMPI(N1*N2, comm, 0);
  n12 = n1;
  if (n1 == -1)
    n1 = result->as_vec->len/N2;
  else
    n2 = result->as_vec->len/N1;

  result->rows = n1;
  result->cols = n2;
  result->glob_rows = N1;
  result->glob_cols = N2;
  result->data = (double **)calloc(n2   ,sizeof(double *));
  result->data[0] = (double  *)calloc(n1*n2,sizeof(double));
  result->as_vec->data = result->data[0];
  for (i=1; i < n2; i++)
    result->data[i] = result->data[i-1] + n1;
  result->col = malloc(n2*sizeof(Vector));
  for (i=0;i<n2;++i) {
    if (n12 == N1)
      result->col[i] = createVectorMPI(N1, &SelfComm, 0);
    else
      result->col[i] = createVectorMPI(N1, comm, 0);
    result->col[i]->data = result->data[i];
  }
  result->row = malloc(n1*sizeof(Vector));
  for (i=0;i<n1;++i) {
    if (n12 == N1)
      result->row[i] = createVectorMPI(N2, comm, 0);
    else
      result->row[i] = createVectorMPI(N2, &SelfComm, 0);
    result->row[i]->data = result->data[0]+i;
    result->row[i]->stride = n1;
  }

  return result;
}
#endif

void freeMatrix(Matrix A)
{
  int i;
  for (i=0;i<A->cols;++i)
    free(A->col[i]);
  for (i=0;i<A->rows;++i)
    free(A->row[i]);
  free(A->col);
  free(A->row);
  free(A->as_vec);
  free(A->data[0]);
  free(A->data);
  free(A);
}

int getMaxThreads()
{
#ifdef HAVE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int getCurrentThread()
{
#ifdef HAVE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

double WallTime ()
{
#ifdef HAVE_OPENMP
  return omp_get_wtime();
#else
  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
#endif
}

double dotproduct(Vector u, Vector v)
{
  double locres, res;
  locres = ddot(&u->len, u->data, &u->stride, v->data, &v->stride);

#ifdef HAVE_MPI
  if (u->comm_size > 1) {
    MPI_Allreduce(&locres, &res, 1, MPI_DOUBLE, MPI_SUM, *u->comm);
    return res;
  }
#endif

  return locres;
}

void MxV(Vector u, Matrix A, Vector v, double alpha, double beta, char trans)
{
  dgemv(&trans, &A->rows, &A->cols, &alpha, A->data[0], &A->rows, v->data,
        &v->stride, &beta, u->data, &u->stride);
}

void MxM(Matrix U, Matrix A, Matrix V, double alpha, double beta,
         char transA, char transV)
{
  dgemm(&transA, &transV, &A->rows, &V->cols, &A->cols, &alpha,
        A->data[0], &A->rows, V->data[0], &A->cols, &beta, U->data[0], &U->rows);
}

void diag(Matrix A, int diag, double value)
{
  int i;
  for (i=0;i<A->rows;++i) {
    if (i+diag >= 0 && i+diag < A->cols)
      A->data[i+diag][i] = value;
  }
}

Vector equidistantMesh(double x0, double x1, int N)
{
  double h = (x1-x0)/N;
  Vector result = createVector(N+1);
  int i;

  for (i=0;i<N+1;++i)
    result->data[i] = i*h;

  return result;
}

void evalMeshInternal(Vector u, Vector grid, function1D func)
{
  int i;
  for (i=1;i<grid->len-1;++i)
    u->data[i-1] = func(grid->data[i]);
}

void evalMeshInternal2(Matrix u, Vector grid, function2D func, int boundary)
{
  int i, j;
  for (i=1;i<grid->len-1;++i)
    for (j=1;j<grid->len-1;++j)
      u->data[j-!boundary][i-!boundary] = func(grid->data[i], grid->data[j]);
}

void scaleVector(Vector u, double alpha)
{
  dscal(&u->len, &alpha, u->data, &u->stride);
}

void axpy(Vector y, const Vector x, double alpha)
{
  daxpy(&x->len, &alpha, x->data, &x->stride, y->data, &y->stride);
}


void lusolve(Matrix A, Vector x, int** ipiv)
{
  int one=1;
  int info;
  if (*ipiv == NULL) {
    *ipiv = malloc(x->len*sizeof(int));
    dgesv(&x->len,&one,A->data[0],&x->len,*ipiv,x->data,&x->len,&info);
  } else {
    char trans='N';
    dgetrs(trans,&x->len,&one,A->data[0],&x->len,*ipiv,x->data,&x->len,&info);
  }
  if (info < 0)
    printf("error solving linear system [%i]\n", info);
}

void llsolve(Matrix A, Vector x, int prefactored)
{
  int one=1;
  int info;
  char uplo='L';
  if (prefactored)
    dpotrs(&uplo,&x->len,&one,A->data[0],&x->len,x->data,&x->len,&info);
  else
    dposv(&uplo,&x->len,&one,A->data[0],&x->len,x->data,&x->len,&info);
  if (info < 0)
    printf("error solving linear system [%i]\n", info);
}

void lutsolve(Matrix A, Vector x, char uplo)
{
  char trans='N';
  char diag='N';
  int one=1;
  int info;
  dtrtrs(&uplo, &trans, &diag, &A->rows, &one,
         A->data[0], &A->rows, x->data, &A->rows, &info);
}

double maxNorm(const Vector x)
{
  // idamax is a fortran function, and the first index is 1
  // since indices in C are 0 based, we have to decrease it 
  double result = fabs(x->data[idamax(&x->len, x->data, &x->stride)-1]);
#ifdef HAVE_MPI
  if (x->comm_size > 1) {
    double r2=result;
    MPI_Allreduce(&r2, &result, 1, MPI_DOUBLE, MPI_MAX, *x->comm);
  }
#endif

  return result;
}

void copyVector(Vector y, const Vector x)
{
  dcopy(&x->len, x->data, &x->stride, y->data, &y->stride);
}

void fillVector(Vector x, double alpha)
{
  int i;
  for (i=0;i<x->len;++i)
    x->data[i*x->stride] = alpha;
}

Matrix subMatrix(const Matrix A, int r_ofs, int r,
                 int c_ofs, int c)
{
  int i, j;
  Matrix result = createMatrix(r, c);
  for (i=0;i<c;++i)
    for (j=0;j<r;++j)
      result->data[i][j] = A->data[i+c_ofs][j+r_ofs];

  return result;
}

void transposeMatrix(Matrix A, const Matrix B)
{
  int i,j;
  for (i=0;i<B->rows;++i)
    for (j=0;j<B->cols;++j)
      A->data[i][j] = B->data[j][i];
}
