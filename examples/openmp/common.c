#include "common.h"
#include <stdlib.h>
#include <sys/time.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

Vector createVector(int len)
{
  Vector result = (Vector)malloc(sizeof(vector_t));
  result->data = calloc(len, sizeof(double));
  result->len = len;
  result->stride = 1;

  return result;
}

void freeVector(Vector vec)
{
  free(vec->data);
  free(vec);
}

Matrix createMatrix(int n1, int n2)
{
  int i;
  Matrix result = (Matrix)malloc(sizeof(matrix_t));
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
    result->row[i]->stride = n2;
  }

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

void MxV(Vector u, const Matrix A, const Vector v)
{
  char trans='N';
  double onef=1.0;
  double zerof=0.0;
  dgemv(&trans, &A->rows, &A->cols, &onef, A->data[0], &A->rows, v->data,
        &v->stride, &zerof, u->data, &u->stride);
}

void MxM(const Matrix A, const Matrix B, Matrix C, double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &B->cols, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[0], &A->cols, &beta, C->data[0], &C->rows);
}

void MxM2(const Matrix A, const Matrix B, Matrix C, int b_ofs, int b_col, 
          double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &A->rows, &b_col, &A->cols, &alpha,
        A->data[0], &A->rows, B->data[b_ofs], &A->cols, &beta, C->data[b_ofs],
        &C->rows);
}

double innerproduct(const Vector u, const Vector v)
{
  return ddot(&u->len, u->data, &u->stride, v->data, &v->stride);
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
