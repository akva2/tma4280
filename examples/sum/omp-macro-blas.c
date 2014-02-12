#include <stdlib.h>
#include <stdio.h>

#include "common.h"

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v)
{
  double alpha=0.0;
  Matrix temp;
  int i, t;

  t = getMaxThreads();

  temp = createMatrix(A->rows, t);
#pragma omp parallel for schedule(static) reduction(+:alpha)
  for(i=0;i<v->cols;++i) {
    MxV(temp->col[getCurrentThread()],A,v->col[i], 1.0, 0.0);
    alpha += dotproduct(temp->col[getCurrentThread()],v->col[i]);
  }
  freeMatrix(temp);

  return alpha;
}

int main(int argc, char** argv)
{
  int N, K, i, j;
  Matrix A,v;
  double time, sum;

  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
    return 1;
  }

  N=atoi(argv[1]);
  K=atoi(argv[2]);

  A = createMatrix(N,N);
  // identity matrix
  for (i=0;i<N;++i)
    A->data[i][i] = 1.0;

  v = createMatrix(N,K);

  // fill with column number
  for (i=0;i<K;++i)
    for (j=0;j<N;++j)
      v->data[i][j] = i;

  time = WallTime();
  sum = dosum(A,v);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  freeMatrix(v);
  freeMatrix(A);

  return 0;
}
