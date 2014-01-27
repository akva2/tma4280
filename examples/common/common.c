#include "common.h"
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

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
    result->row[i]->data = result->data[0]+n1;
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
#ifdef HAVE_MPI
  return MPI_Wtime();
#elif defined(HAVE_OPENMP)
  return omp_get_wtime();
#else
  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
#endif
}
