#include "common.h"
#include <stdlib.h>
#include <sys/time.h>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

double** createMatrix(int n1, int n2)
{
  int i;
  double **a;
  a    = (double **)calloc(n2   ,sizeof(double *));
  a[0] = (double  *)calloc(n1*n2,sizeof(double));
  for (i=1; i < n2; i++)
    a[i] = a[i-1] + n1;

  return (a);
}

void MxV(double* u, double** A, double* v, int N)
{
  char trans='N';
  double onef=1.0;
  double zerof=0.0;
  int one=1;
  dgemv(&trans, &N, &N, &onef, A[0], &N, v, &one, &zerof, u, &one);
}

void MxM2(double** A, double** B, double** C, int M, int N, int K,
         double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &M, &N, &K, &alpha, A[0], &M, B[0], &K, &beta, C[0], &M);
}

void MxM(double* A, double* B, double* C, int M, int N, int K,
         double alpha, double beta)
{
  char trans='N';
  dgemm(&trans, &trans, &M, &N, &K, &alpha, A, &M, B, &K, &beta, C, &M);
}

double innerproduct(double* u, double* v, int N)
{
  int one=1;
  return ddot(&N, u, &one, v, &one);
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
