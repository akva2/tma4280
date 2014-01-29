#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <string.h>

#include "blaslapack.h"
#include "common.h"

void mxm_std (Matrix a, Matrix b, Matrix c)
{
  int i, j, l;
  for (i=0; i <  a->rows; i++) {
    for (j=0; j  < b->cols; j++) {
      c->data[i][j] =  0.0;
      for (l=0;  l < a->cols; l++) {
        c->data[j][i] += a->data[l][i]*b->data[j][l];
      }
    }
  }
}

void mxm_unr (Matrix a, Matrix b, Matrix c)
{
  int i, j;
  for (i=0; i < a->rows; i++) {
    for (j=0; j < b->cols; j++) {
      c->data[i][j] = a->data[i][0]*b->data[0][j]
        +a->data[1][i]*b->data[j][1]
        +a->data[2][i]*b->data[j][2]
        +a->data[3][i]*b->data[j][3]
        +a->data[4][i]*b->data[j][4]
        +a->data[5][i]*b->data[j][5]
        +a->data[6][i]*b->data[j][6]
        +a->data[7][i]*b->data[j][7]
        +a->data[8][i]*b->data[j][8]
        +a->data[9][i]*b->data[j][9];
    }
  }
}

void mxm_blas (Matrix a, Matrix b, Matrix c)
{
  char    trans = 'N';
  double alpha = 1.0, beta = 0.0;
  dgemm(&trans, &trans, &a->rows, &b->cols, &a->cols, &alpha,
        a->data[0], &a->cols, b->data[0], &a->rows, &beta, c->data[0], &a->rows);
}

void mxm (Matrix a, Matrix b, Matrix c, int flag)
{
  if (flag == 0)
    mxm_std(a, b, c); 
  else if (flag == 1 && a->cols == 10)
    mxm_unr(a, b, c); 
  else if (flag == 2)
    mxm_blas(a, b, c);
  else {
    printf (" Incorrect setting of variable flag in mxm \n");
    close_app();
    exit(1);
  }
}

int main(int argc, char **argv )
{
  int size, rank;
  long i, j, l, m, n, k, iter, loop;
  Matrix a, b, c;
  double t1, t2, dt, dt1, r;

  init_app(argc, argv, &rank, &size);

  if( argc < 3 ) {
    if (rank == 0)
      printf("need atleast 2 arguments - n & flag\n");
    close_app();
    return 1;
  }

  k = m = n = atoi(argv[1]);
  a = createMatrix(m, k);
  b = createMatrix(k, n);
  c = createMatrix(m, n);

  for (i=0; i  < m; i++) {
    for (l=0; l < k; l++) {
      a->data[l][i] = i+1.0;
    }
  }
  for (l=0; l  < k; l++) {
    for (j=0; j < n; j++) {
      b->data[j][l] = j+1.0;
    }
  }
  loop = 5;

  t1 = WallTime(); 
  for (iter=0; iter < loop; iter++)
    mxm (a,b,c,atoi(argv[2]));
  t2 = WallTime();

  dt = t2 - t1;
  dt1 = dt/(m*2*k*n);
  dt1 = dt1/loop;
  r    = 1.e-6/dt1;
  printf (" matrix-matrix : (m,k,n)= (%ld,%ld,%ld)    dt= %lf (s) dt1= %le r= %lf\n" ,m, k, n, dt, dt1, r);

  freeMatrix(a);
  freeMatrix(b);
  freeMatrix(c);

  close_app();

  return 0;
}
