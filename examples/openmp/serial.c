#include <stdlib.h>
#include <stdio.h>

#include "common.h"

// perform a matrix-vector product
void myMxV(double* u, double** A, double* v, int N)
{
  for( int i=0;i<N;++i) {
    u[i] = 0;
    for( int j=0;j<N;++j )
      u[i] += A[j][i]*v[j];
  }
}

// perform an innerproduct
double myinnerproduct(double* u, double* v, int N)
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
  double temp[N];
  for( int i=0;i<K;++i ) {
    myMxV(temp,A,v[i],N);
    alpha += myinnerproduct(temp,v[i],N);
  }

  return alpha;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
    return 1;
  }
  int N=atoi(argv[1]);
  int K=atoi(argv[2]);

  double** A = createMatrix(N,N);
  // identity matrix
  for (int i=0;i<N;++i)
    A[i][i] = 1.0;

  double** v = createMatrix(N,K);
  // fill with column number
  for (int i=0;i<K;++i)
    for (int j=0;j<N;++j)
      v[i][j] = i;

  double time = WallTime();
  double sum = dosum(A,v,K,N);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  return 0;
}
