#include <stdlib.h>
#include <stdio.h>

#include "common.h"

// perform a matrix-vector product
void myMxV(Vector u, Matrix A, Vector v)
{
  Vector temp = createVector(A->rows);
  for( int i=0;i<A->rows;++i) {
    temp->data[i] = 0;
    for( int j=0;j<A->cols;++j )
      temp->data[i] += A->data[j][i]*v->data[j];
  }
#ifdef HAVE_MPI
  for (int i=0;i<v->comm_size;++i) {
    MPI_Reduce(temp->data+v->displ[i], u->data, v->sizes[i],
               MPI_DOUBLE, MPI_SUM, i, *v->comm);
  }
#else
  memcpy(u->data, temp->data, u->len*sizeof(double));
#endif
  freeVector(temp);
}

// perform an innerproduct
double myinnerproduct(Vector u, Vector v)
{
  double result=0;
  for( int i=0;i<u->len;++i )
    result += u->data[i]*v->data[i];
#ifdef HAVE_MPI
  double r2=result;
  MPI_Allreduce(&r2, &result, 1, MPI_DOUBLE, MPI_SUM, *u->comm);
#endif
  return result;
}

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v)
{
  double alpha=0;
  for( int i=0;i<v->cols;++i ) {
    Vector temp = createVector(A->rows);
    myMxV(temp,A,v->col[i]);
    alpha += myinnerproduct(v->col[i], temp);
    freeVector(temp);
  }

  return alpha;
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 3) {
    printf("need two parameters, the matrix size and the number of vectors\n");
    close_app();
    return 1;
  }
  int N=atoi(argv[1]);
  int K=atoi(argv[2]);

  Matrix A = createMatrixMPI(N, -1, N, N, &WorldComm);
  // identity matrix
  for (int i=0;i<A->cols;++i)
    A->data[i][i] = 1.0;
  
  Matrix v = createMatrixMPI(-1, K, N, K, &WorldComm);
  // fill with column number
  for (int i=0;i<v->rows;++i)
    for (int j=0;j<v->cols;++j)
      v->data[j][i] = j;
        
  double time = WallTime();
  double sum = dosum(A,v);

  if (rank == 0) {
    printf("sum: %f\n", sum);
    printf("elapsed: %f\n", WallTime()-time);
  }

/*  freeMatrix(v);*/
/*  freeMatrix(A);*/
  close_app();
  return 0;
}
