#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "blaslapack.h"
#include "common.h"

void daxpy_blas(Vector a, double alpha, Vector b)
{
  daxpy(&a->len, &alpha, a->data, &a->stride, b->data, &b->stride);
}

void daxpy_std(Vector a, double alpha, Vector b)
{
  int i;
  for (i=0;i<a->len;++i)
    a->data[i] += alpha*b->data[i];
}

double dot_std(Vector a, Vector b)
{
  double result=0.0;
  int i;
  for (i=0;i<a->len;++i)
    result += a->data[i]*b->data[i];

  return result;
}

double dot_blas(Vector a, Vector b)
{
  return ddot(&a->len, a->data, &a->stride, b->data, &b->stride);
}

void doop(Vector a, double alpha, Vector b, int flag)
{
  switch(flag) {
    case 0:
      daxpy_std(a, alpha, b);
      break;
    case 1:
      daxpy_blas(a, alpha, b);
      break;
    case 2:
      dot_std(a, a);
      break;
    case 3:
      dot_std(a, b);
      break;
    case 4:
      dot_blas(a, a);
      break;
    case 5:
      dot_blas(a, b);
      break;
    default:
      printf("invalid flag given\n");
      close_app();
      exit(1);
  }
}

int main(int argc, char **argv )
{
  int size, rank;
  long i, n, iter, loop, flag;
  Vector a, b;
  double t1, t2, dt, dt1, r;

  init_app(argc, argv, &rank, &size);

  if( argc < 3 ) {
    if (rank == 0)
      printf("need atleast 2 arguments - n & flag\n");
    close_app();
    return 1;
  }

  n = atoi(argv[1]);
  flag = atoi(argv[2]);
  a = createVector(n);
  b = createVector(n);
  for (i=0;i<n;++i)
    a->data[i] = b->data[i] = i+1;

  loop = 5;
  t1 = WallTime();
  for (iter=0; iter < loop; iter++)
    doop(a, 1.0 ,b, flag);

  t2 = WallTime();
  dt = t2 - t1;
  dt1 = dt/(2*n);
  dt1 = dt1/loop;
  r    = 1.e-6/dt1;
  printf (" daxpy : (n)= (%ld)    dt= %e (s) dt1= %e r= %e\n",n, dt, dt1, r);
  close_app();

  return 0;
}

