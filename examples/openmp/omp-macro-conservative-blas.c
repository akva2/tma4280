#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#include "common.h"

// calculates \sum_K v_i'*A*v_i
double dosum(double** A, double** v, int K, int N)
{
  double alpha=0;
  int t = omp_get_max_threads();
  double** temp = createMatrix(N,t);
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
