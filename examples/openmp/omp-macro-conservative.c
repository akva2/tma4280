#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#include "common.h"

// perform a matrix-vector product
void myMxV(Vector u, Matrix A, Vector v)
{
  for( int i=0;i<A->rows;++i) {
    u->data[i] = 0;
    for( int j=0;j<A->cols;++j )
      u->data[i] += A->data[j][i]*v->data[j];
  }
}

// perform an innerproduct
double myinnerproduct(Vector u, Vector v)
{
  double result=0;
  for( int i=0;i<u->len;++i )
    result += u->data[i]*v->data[i];

  return result;
}

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v)
{
  double alpha=0;
  int t = omp_get_max_threads();
  Matrix temp = createMatrix(A->rows,t);
#pragma omp parallel for schedule(static) reduction(+:alpha)
  for( int i=0;i<v->cols;++i ) {
    myMxV(temp->col[omp_get_thread_num()],A,v->col[i]);
    alpha += myinnerproduct(temp->col[omp_get_thread_num()],v->col[i]);
  }
  freeMatrix(temp);

  return alpha;
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
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

  double time = WallTime();
  double sum = dosum(A,v);

  printf("sum: %f\n", sum);
  printf("elapsed: %f\n", WallTime()-time);
  freeMatrix(v);
  freeMatrix(A);

  return 0;
}
