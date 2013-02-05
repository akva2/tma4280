#include <stdlib.h>
#include <stdio.h>

#include "common.h"

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

  double** v2 = createMatrix(N,K);
  double time = WallTime();
  MxM2(A, v, v2, N, K, N, 1.0, 0.0);
  double sum = innerproduct(v[0], v2[0], N*K);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  return 0;
}
