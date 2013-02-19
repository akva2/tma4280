#include <stdlib.h>
#include <stdio.h>

#include "common.h"

void myMxM(Matrix A, Matrix v, Matrix u, int* cols, 
           int* displ, int size)
{
  Matrix temp = createMatrix(A->rows, v->cols);
#pragma omp parallel
  {
    int* displ, *cols;
    splitVector(v->cols, num_threads(), &cols, &displ);
    MxM2(A, v, temp, displ[get_thread()], cols[get_thread()],
         displ[get_thread()], 1.0, 0.0);
    free(cols);
    free(displ);
  }
#ifdef HAVE_MPI
  for (int i=0;i<size;++i) {
    Matrix t = subMatrix(temp, displ[i], cols[i], 0, v->cols);
    MPI_Reduce(t->data[0], u->data[0], cols[i]*v->cols,
               MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
    freeMatrix(t);
  }
#else
  memcpy(u->data[0], temp->data[0], u->as_vec->len*sizeof(double));
#endif
  freeMatrix(temp);
}

// perform an innerproduct
double myinnerproduct(Vector u, Vector v)
{
  double result = innerproduct(u, v);
#ifdef HAVE_MPI
  double r2=result;
  MPI_Allreduce(&r2, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return result;
}

// calculates \sum_K v_i'*A*v_i
double dosum(Matrix A, Matrix v, int* cols, int* displ, int size)
{
  Matrix v2 = createMatrix(v->rows, v->cols);
  myMxM(A, v, v2, cols, displ, size);
  double alpha = myinnerproduct(v->as_vec, v2->as_vec);
  freeMatrix(v2);

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

  int *displ, *cols;
  splitVector(K, size, &cols, &displ);

  Matrix A = createMatrix(N,cols[rank]);
  // identity matrix
  for (int i=0;i<cols[rank];++i)
    A->data[i][i+displ[rank]] = 1.0;
  
  Matrix v = createMatrix(cols[rank], N);
  // fill with column number
  for (int i=0;i<N;++i)
    for (int j=0;j<cols[rank];++j)
      v->data[i][j] = i;

  double time = WallTime();
  double sum = dosum(A,v,cols,displ,size);

  if (rank == 0) {
    printf("sum: %f\n", sum);
    printf("elapsed: %f\n", WallTime()-time);
  }

  freeMatrix(v);
  freeMatrix(A);
  close_app();
  return 0;
}
