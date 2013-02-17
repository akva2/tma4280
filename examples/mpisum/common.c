#include "common.h"
#include <stdlib.h>
#include <sys/time.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

void init_app(int argc, char** argv, int* rank, int* size)
{
#if HAVE_MPI
#ifdef HAVE_OPENMP
  int aquired;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &aquired);
#else
  MPI_Init(&argc, &argv);
#endif
  MPI_Comm_size(MPI_COMM_WORLD, size);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
#else
  *rank = 0;
  *size = 1;
#endif
}

void close_app()
{
#ifdef HAVE_MPI
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
Vector createVectorMPI(int globLen, MPI_Comm* comm)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  MPI_Comm_dup(*comm, &result->comm);
  MPI_Comm_size(*comm, &result->comm_size);
  MPI_Comm_rank(*comm, &result->comm_rank);
  splitVector(globLen, result->comm_size, &result->sizes, &result->displ);
  result->len = result->sizes[result->comm_rank];
  result->data = calloc(result->len, sizeof(double));
  result->globLen = globLen;
}
#endif

void freeVector(Vector vec)
{
  free(vec->data);
#ifdef HAVE_MPI
  if (vec->comm)
    MPI_Comm_free(&vec->comm);
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
  result->as_vec->len = n1*n2;
  result->col = malloc(n2*sizeof(Vector));
  for (int i=0;i<n2;++i) {
    result->col[i] = malloc(sizeof(vector_t));
    result->col[i]->len = n1;
    result->col[i]->data = result->data[i];
  }

  return result;
}

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
  free(A->as_vec);
  free(A->data[0]);
  free(A->data);
  free(A);
}

void MxV(Vector u, Matrix A, Vector v)
{
  char trans='N';
  double onef=1.0;
  double zerof=0.0;
  int one=1;
  dgemv(&trans, &A->rows, &A->cols, &onef, A->data[0], &A->rows, v->data,
        &one, &zerof, u->data, &one);
}

void MxM(Matrix A, Matrix B, Matrix C, double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &B->cols, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[0], &A->cols, &beta, C->data[0], &C->rows);
}

void MxM2(Matrix A, Matrix B, Matrix C, int b_ofs, int b_col, 
          int c_ofs, double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &b_col, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[b_ofs], &A->cols, &beta, C->data[c_ofs],
        &C->rows);
}

double innerproduct(Vector u, Vector v)
{
  int one=1;
  return ddot(&u->len, u->data, &one, v->data, &one);
}

double innerproduct2(Vector u, int ofs, int len, Vector v)
{
  int one=1;
  return ddot(&len, u->data+ofs, &one, v->data, &one);
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
