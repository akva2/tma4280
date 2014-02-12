#include <stdlib.h>
#include <stdio.h>

#include "common.h"

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v)
{
  double alpha=0.0, s2;
  int i;
  Vector temp = createVector(A->rows);
  for (i=0;i<v->cols;++i) {
    MxV(temp,A,v->col[i], 1.0, 0.0);
    alpha += dotproduct(temp,v->col[i]);
  }
  freeVector(temp);
#ifdef HAVE_MPI
  s2=alpha;
  MPI_Allreduce(&s2, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  return alpha;
}

int main(int argc, char** argv)
{
  int i, j, N, K;
  Matrix A, v;
  double time, sum;
  int rank, size;
  int *displ, *cols;

  init_app(argc, argv, &rank, &size);

  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
    close_app();
    return 1;
  }
  N=atoi(argv[1]);
  K=atoi(argv[2]);

  A = createMatrix(N,N);
  // identity matrix
  for (i=0;i<N;++i)
    A->data[i][i] = 1.0;
  
  splitVector(K, size, &cols, &displ);
  v = createMatrix(N,cols[rank]);
  // fill with column number
  for (i=0;i<cols[rank];++i)
    for (j=0;j<N;++j)
      v->data[i][j] = i+displ[rank];

  time = WallTime();
  sum = dosum(A,v);

  if (rank == 0) {
    printf("sum: %f\n", sum);
    printf("elapsed: %f\n", WallTime()-time);
  }

  freeMatrix(v);
  freeMatrix(A);
  free(displ);
  free(cols);

  close_app();
  return 0;
}
