#include <stdlib.h>
#include <stdio.h>

#include "common.h"

// perform a matrix-vector product
void myMxV(Vector u, Matrix A, Vector v)
{
  int i, j;
  Vector temp;
#ifdef HAVE_MPI
  temp = createVector(A->rows);
#else
  temp = u;
#endif
  for (i=0;i<A->rows;++i) {
    temp->data[i] = 0;
    for (j=0;j<A->cols;++j )
      temp->data[i] += A->data[j][i]*v->data[j];
  }
#ifdef HAVE_MPI
  for (i=0;i<v->comm_size;++i) {
    MPI_Reduce(temp->data+v->displ[i], u->data, v->sizes[i],
               MPI_DOUBLE, MPI_SUM, i, *v->comm);
  }
  freeVector(temp);
#endif
}

// perform an innerproduct
double myinnerproduct(Vector u, Vector v)
{
  int i;
  double result=0.0, r2;
  for (i=0;i<u->len;++i)
    result += u->data[i]*v->data[i];
#ifdef HAVE_MPI
  r2=result;
  MPI_Allreduce(&r2, &result, 1, MPI_DOUBLE, MPI_SUM, *u->comm);
#endif
  return result;
}

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v)
{
  int i;
  double alpha=0.0;
  for (i=0;i<v->cols;++i) {
    Vector temp = createVector(A->rows);
    myMxV(temp,A,v->col[i]);
    alpha += myinnerproduct(v->col[i], temp);
    freeVector(temp);
  }

  return alpha;
}

int main(int argc, char** argv)
{
  int rank, size, N, K, i, j;
  Matrix A,v;
  double time, sum;

  init_app(argc, argv, &rank, &size);

  if (argc < 3) {
    if (rank == 0)
      printf("need two parameters, the matrix size and the number of vectors\n");
    close_app();
    return 1;
  }
  N=atoi(argv[1]);
  K=atoi(argv[2]);

  A = createMatrixMPI(N, -1, N, N, &WorldComm);
  // identity matrix
  for (i=0;i<A->cols;++i)
    A->data[i][i] = 1.0;
  
  v = createMatrixMPI(-1, K, N, K, &WorldComm);
  // fill with column number
  for (i=0;i<v->rows;++i)
    for (j=0;j<v->cols;++j)
      v->data[j][i] = j;
        
  time = WallTime();
  sum = dosum(A,v);

  if (rank == 0) {
    printf("sum: %f\n", sum);
    printf("elapsed: %f\n", WallTime()-time);
  }

  freeMatrix(v);
  freeMatrix(A);
  close_app();
  return 0;
}
