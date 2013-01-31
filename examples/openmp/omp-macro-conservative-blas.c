#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#ifndef USE_MKL
  #define ddot ddot_
  #define dgemv dgemv_
#endif

double ddot(int* N, double* dx, int* incx, double* dy, int* incy);
double dgemv(char* trans, int* M, int* N, double* alpha, double* A,
             int* LDA, double* x, int* incx,
             double* beta, double* y, int* incy);


// allocate a n1xn2 matrix in fortran format
// note that reversed index order is assumed
double** createMatrix(int n1, int n2)
{
  int i;
  double **a;
  a    = (double **)calloc(n1   ,sizeof(double *));
  a[0] = (double  *)calloc(n1*n2,sizeof(double));
  for (i=1; i < n1; i++)
    a[i] = a[i-1] + n2;

  return (a);
}

// perform a matrix-vector product
void MxV(double* u, double** A, double* v, int N)
{
  char trans='N';
  double onef=1.0;
  double zerof=0.0;
  int one=1;
  dgemv(&trans, &N, &N, &onef, A[0], &N, v, &one, &zerof, u, &one);
}

// perform an innerproduct
double innerproduct(double* u, double* v, int N)
{
  int one=1;
  return ddot(&N, u, &one, v, &one);
}

// calculates \sum_K v_i'*A*v_i
double dosum(double** A, double** v, int K, int N)
{
  double alpha=0;
  int t = omp_get_max_threads();
  double** temp = createMatrix(t,N);
#pragma omp parallel for schedule(static) reduction(+:alpha)
  for( int i=0;i<K;++i ) {
    MxV(temp[omp_get_thread_num()],A,v[i],N);
    alpha += innerproduct(temp[omp_get_thread_num()],v[i],N);
  }

  return alpha;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
  }
  int N=atoi(argv[1]);
  int K=atoi(argv[2]);

  double** A = createMatrix(N,N);
  // identity matrix
  for (int i=0;i<K;++i)
    A[i][i] = 1.0;

  double** v = createMatrix(K,N);
  // fill with row number
  for (int i=0;i<K;++i)
    for (int j=0;j<N;++j)
      v[j][i] = i;

  double sum = dosum(A,v,K,N);

  printf("sum: %f\n", sum);

  return 0;
}
