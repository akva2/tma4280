#include "common.h"

#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#include "blaslapack.h"

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>

MPI_Comm WorldComm;
MPI_Comm SelfComm;
#endif

void init_app(int argc, char** argv, int* rank, int* size)
{
#if HAVE_MPI
#ifdef HAVE_OPENMP
  int aquired;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &aquired);
  printf("aq %i %i %i %i %i\n", aquired, MPI_THREAD_SINGLE,
                                MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED,
                                MPI_THREAD_MULTIPLE);
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

int get_thread()
{
#ifdef HAVE_OPENMP
  return omp_get_thread_num();
#endif

  return 0;
}

int num_threads()
{
#ifdef HAVE_OPENMP
  return omp_get_num_threads();
#endif
  return 1;
}

int max_threads()
{
#ifdef HAVE_OPENMP
  return omp_get_max_threads();
#endif
  return 1;
}

Vector createVector(int len)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->data = calloc(len, sizeof(double));
  result->len = len;
  result->stride = 1;

  return result;
}

void splitVector(int globLen, int size, int** len, int** displ)
{
  *len = calloc(size,sizeof(int));
  *displ = calloc(size,sizeof(int));
  for (int i=0;i<size;++i) {
    (*len)[i] = globLen/size;
    if (globLen % size && i >= (size - globLen % size))
      (*len)[i]++;
    if (i < size-1)
      (*displ)[i+1] = (*displ)[i]+(*len)[i];
  }
}

#ifdef HAVE_MPI
Vector createVectorMPI(int globLen, int allocdata, MPI_Comm* comm)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->comm = comm;
  MPI_Comm_size(*comm, &result->comm_size);
  MPI_Comm_rank(*comm, &result->comm_rank);
  splitVector(globLen, result->comm_size, &result->sizes, &result->displ);
  result->len = result->sizes[result->comm_rank];
  if (allocdata)
    result->data = calloc(result->len, sizeof(double));
  else
    result->data = NULL;
  result->globLen = globLen;

  return result;
}
#endif

void freeVector(Vector vec)
{
  free(vec->data);
#ifdef HAVE_MPI
  if (vec->comm)
    MPI_Comm_free(vec->comm);
#endif
  free(vec->displ);
  free(vec->sizes);
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
  result->as_vec->len = result->as_vec->globLen = n1*n2;
  result->as_vec->stride = 1;
  result->col = malloc(n2*sizeof(Vector));
  for (int i=0;i<n2;++i) {
    result->col[i] = malloc(sizeof(vector_t));
    result->col[i]->len = n1;
    result->col[i]->data = result->data[i];
    result->col[i]->stride = 1;
  }
  result->row = malloc(n1*sizeof(Vector));
  for (int i=0;i<n1;++i) {
    result->row[i] = malloc(sizeof(vector_t));
    result->row[i]->len = n2;
    result->row[i]->data = result->data[0]+i;
    result->row[i]->stride = n1;
  }

  return result;
}

#ifdef HAVE_MPI
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm)
{
  int i;
  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));

  result->as_vec = createVectorMPI(N1*N2, 0, comm);
  int n12 = n1;
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
  for (int i=0;i<n2;++i) {
    if (n12 == N1) {
      result->col[i] = createVectorMPI(N1, 0, &SelfComm);
    } else {
      result->col[i] = createVectorMPI(N1, 0, comm);
    }
    result->col[i]->data = result->data[i];
  }
  result->row = malloc(n1*sizeof(Vector));
  for (int i=0;i<n1;++i) {
    if (n12 == N1)
      result->row[i] = createVectorMPI(N2, 0, comm);
    else
      result->row[i] = createVectorMPI(N2, 0, &SelfComm);
    result->row[i]->data = result->data[0]+i;
    result->row[i]->stride = n1;
  }

  return result;
}
#endif

Matrix subMatrix(const Matrix A, int r_ofs, int r,
                 int c_ofs, int c)
{
  Matrix result = createMatrix(r, c);
  for (int i=0;i<c;++i)
    for (int j=0;j<r;++j)
      result->data[i][j] = A->data[i+c_ofs][j+r_ofs];

  return result;
}

void freeMatrix(Matrix A)
{
  for (int i=0;i<A->cols;++i)
    free(A->col[i]);
  free(A->col);
  for (int i=0;i<A->rows;++i)
    free(A->row[i]);
  free(A->row);
  free(A->as_vec);
  free(A->data[0]);
  free(A->data);
  free(A);
}

void MxV(Vector y, const Matrix A, const Vector x, double alpha, double beta)
{
  char trans='N';
  int one=1;
#ifdef HAVE_MPI
  Vector temp = createVector(A->rows);
  copyVector(y, x);
#else
  Vector temp=y;
#endif
  dgemv(&trans, &A->rows, &A->cols, &alpha, A->data[0], &A->rows, x->data,
        &x->stride, &beta, temp->data, &one);
#ifdef HAVE_MPI
  for (int i=0;i<x->comm_size;++i) {
    MPI_Reduce(temp->data+x->displ[i], y->data, x->sizes[i],
               MPI_DOUBLE, MPI_SUM, i, *x->comm);
  }
  freeVector(temp);
#endif
}

void MxVdispl(Vector y, const Matrix A, const Vector x, 
              double alpha, double beta, int ydispl)
{
  char trans='N';
  dgemv(&trans, &A->rows, &A->cols, &alpha, A->data[0], &A->rows, x->data,
        &x->stride, &beta, y->data+ydispl*y->stride, &y->stride);
}

void MxM(Matrix C, const Matrix A, const Matrix B, double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &B->cols, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[0], &A->cols, &beta, C->data[0], &C->rows);
}

void MxM2(Matrix C, const Matrix A, const Matrix B, int b_ofs, int b_col, 
          int c_ofs, double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &b_col, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[b_ofs], &A->cols, &beta, C->data[c_ofs],
        &C->rows);
}

void transposeMatrix(Matrix A, const Matrix B)
{
  for (int i=0;i<B->cols;++i)
    for (int j=0;j<B->rows;++j)
      A->data[j][i] = B->data[i][j];
}

double innerproduct(const Vector x, const Vector y)
{
  return ddot(&x->len, x->data, &x->stride, y->data, &y->stride);
}

double innerproduct2(const Vector x, const Vector y, int xdispl, int len)
{
  return ddot(&len, x->data+xdispl*x->stride, &x->stride, y->data, &y->stride);
}

double WallTime ()
{
#ifdef HAVE_MPI
  return MPI_Wtime();
#endif
#ifdef HAVE_OPENMP
  return omp_get_wtime();
#endif

  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}

void saveVectorSerial(char* name, const Vector x)
{
  FILE* f = fopen(name,"wb");
  for (int i=0;i<x->len;++i)
    fprintf(f,"%f ",x->data[i]);
  fclose(f);
}

void saveMatrixSerial(char* name, const Matrix x)
{
  FILE* f = fopen(name,"wb");
  for (int i=0;i<x->rows;++i)
    for(int j=0;j<x->cols;++j)
      fprintf(f,"%f%c",x->data[j][i], j==x->cols-1?'\n':' ');
  fclose(f);
}

#ifdef HAVE_MPI
void saveVectorMPI(char* name, const Vector x)
{
  MPI_File f;
  MPI_File_open(*x->comm, name, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &f);
  MPI_File_seek(f, 17*x->displ[x->comm_rank], MPI_SEEK_SET);
  for (int i=0;i<x->len;++i) {
    char num[21];
    sprintf(num,"%016f ",x->data[i]);
    MPI_File_write(f, num, 17, MPI_CHAR, MPI_STATUS_IGNORE);
  }
  MPI_File_close(&f);
}
#endif

void lusolve(Matrix A, Vector x, int** ipiv)
{
  if (*ipiv == NULL)
    *ipiv = malloc(x->len*sizeof(int));
  int one=1;
  int info;
  dgesv(&x->len,&one,A->data[0],&x->len,*ipiv,x->data,&x->len,&info);
}

void llsolve(Matrix A, Vector x)
{
  int one=1;
  int info;
  char uplo='U';
  dpotrf(&uplo,&x->len,A->data[0],&x->len,&info);
  dpotrs(&uplo,&x->len,&one,A->data[0],&x->len,x->data,&x->len,&info);
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

void fillVector(Vector x, double alpha)
{
  for (int i=0;i<x->len;++i)
    x->data[i*x->stride] = alpha;
}

void copyVector(Vector y, const Vector x)
{
  dcopy(&x->len, x->data, &x->stride, y->data, &y->stride);
}

void copyVectorDispl(Vector y, const Vector x, int ysize, int xdispl)
{
  dcopy(&ysize, x->data+xdispl*x->stride, &x->stride, y->data, &y->stride);
}

void scaleVector(Vector x, double alpha)
{
  dscal(&x->len, &alpha, x->data, &x->stride);
}

double maxNorm(const Vector x)
{
  // idamax is a fortran function, and the first index is 1
  // since indices in C are 0 based, we have to decrease it 
  return fabs(x->data[idamax(&x->len, x->data, &x->stride)-1]);
}

void axpy(Vector y, const Vector x, double alpha)
{
  daxpy(&x->len, &alpha, x->data, &x->stride, y->data, &y->stride);
}

void evalMesh(Vector u, const Vector x, const Vector y, function2 f)
{
#pragma omp parallel for schedule(static)
  for (int j=0;j<y->len;++j)
    for (int i=0;i<x->len;++i)
      u->data[j*x->len+i] = f(x->data[i],y->data[j]);
}

void evalMesh2(Vector u, const Vector x, const Vector y,
               function2 f, double alpha)
{
#pragma omp parallel for schedule(static)
  for (int j=0;j<y->len;++j)
    for (int i=0;i<x->len;++i)
      u->data[j*x->len+i] += alpha*f(x->data[i],y->data[j]);
}

double poisson_source(double x, double y)
{
  return -(2.0-4*M_PI*M_PI*x*(x-1.0))*sin(2.0*M_PI*y);
}

double exact_solution(double x, double y)
{
  return x*(x-1.0)*sin(2.0*M_PI*y);
}

Matrix createPoisson2D(int M, double mu)
{
  Matrix A = createMatrix(M*M, M*M);
  for (int i=0;i<M;++i) {
    for (int j=0;j<M;++j) {
      A->data[i*M+j][i*M+j] = 4.0+mu;
      if (j > 0)
        A->data[i*M+j-1][i*M+j] = -1.0;
      if (j < M-1)
        A->data[i*M+j+1][i*M+j] = -1.0;
      if (i < M-1)
        A->data[(i+1)*M+j][i*M+j] = -1.0;
      if (i > 0)
        A->data[(i-1)*M+j][i*M+j] = -1.0;
    }
  }

  return A;
}
