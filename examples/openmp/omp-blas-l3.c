#include <stdlib.h>
#include <stdio.h>

#include "common.h"

#include <omp.h>

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
    return 1;
  }
  int N=atoi(argv[1]);
  int K=atoi(argv[2]);

  Matrix A = createMatrix(N,N);
  // identity matrix
  for (int i=0;i<N;++i)
    A->data[i][i] = 1.0;

  Matrix v = createMatrix(N,K);
  // fill with column number
  for (int i=0;i<K;++i)
    for (int j=0;j<N;++j)
      v->data[i][j] = i;

  Matrix v2 = createMatrix(N,K);
  double time = WallTime();
  int t = omp_get_max_threads();
#pragma omp parallel for schedule(static)
  for (int i=0; i<t;++i)
    MxM2(A, v, v2, i*K/t, K/t, 1.0, 0.0);

  double sum = innerproduct(v->as_vec, v2->as_vec);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  freeMatrix(v);
  freeMatrix(v2);
  freeMatrix(A);
  return 0;
}
