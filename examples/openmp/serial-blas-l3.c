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
  MxM(A, v, v2, 1.0, 0.0);
  double sum = innerproduct(v->as_vec, v2->as_vec);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  freeMatrix(v2);
  freeMatrix(v);
  freeMatrix(A);
  return 0;
}
