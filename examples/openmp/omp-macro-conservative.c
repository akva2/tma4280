#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

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
  for( int i=0;i<N;++i) {
    u[i] = 0;
    for( int j=0;j<N;++j )
      u[i] += A[i][j]*v[j];
  }
}

// perform an innerproduct
double innerproduct(double* u, double* v, int N)
{
  double result=0;
  for( int i=0;i<N;++i )
    result += u[i]*v[i];

  return result;
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
