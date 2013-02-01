#include <stdlib.h>
#include <stdio.h>

// allocate a n1xn2 matrix in fortran format
// note that reversed index order is assumed
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

// perform a matrix-vector product
void MxV(double* u, double** A, double* v, int N)
{
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static) reduction(+:result)
  for( int i=0;i<N;++i )
    result += u[i]*v[i];

  return result;
}

// calculates \sum_K v_i'*A*v_i
double dosum(double** A, double** v, int K, int N)
{
  double alpha=0;
  double temp[N];
  for( int i=0;i<K;++i ) {
    MxV(temp,A,v[i],N);
    alpha += innerproduct(temp,v[i],N);
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
  for (int i=0;i<N;++i)
    A[i][i] = 1.0;

  double** v = createMatrix(N,K);
  // fill with row number
  for (int i=0;i<K;++i)
    for (int j=0;j<N;++j)
      v[i][j] = i;

  double sum = dosum(A,v,K,N);
  printf("sum: %f\n", sum);

  return 0;
}
