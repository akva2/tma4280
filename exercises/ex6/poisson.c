/*
  C-program to solve the two-dimensional Poisson equation on 
  a unit square using one-dimensional eigenvalue decompositions
  and fast sine transforms

  einar m. ronquist
  ntnu, october 2000
  revised, october 2001
*/

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

typedef double Real;

/* function prototypes */
Real *createRealArray (int n);
Real **createReal2DArray (int m, int n);
void transpose (Real **bt, Real **b, int m);
void fst_(Real *v, int *n, Real *w, int *nn);
void fstinv_(Real *v, int *n, Real *w, int *nn);


int main(int argc, char **argv )
{
  Real *diag, **b, **bt, *z;
  Real pi, h, umax;
  int i, j, n, m, nn;

  /* the total number of grid points in each spatial direction is (n+1) */
  /* the total number of degrees-of-freedom in each spatial direction is (n-1) */
  /* this version requires n to be a power of 2 */

 if( argc < 2 ) {
    printf("need a problem size\n");
    return 1;
  }

  n  = atoi(argv[1]);
  m  = n-1;
  nn = 4*n;

  diag = createRealArray (m);
  b    = createReal2DArray (m,m);
  bt   = createReal2DArray (m,m);
  z    = createRealArray (nn);

  h    = 1./(Real)n;
  pi   = 4.*atan(1.);

  for (i=0; i < m; i++) {
    diag[i] = 2.*(1.-cos((i+1)*pi/(Real)n));
  }
  for (j=0; j < m; j++) {
    for (i=0; i < m; i++) {
      b[j][i] = h*h;
    }
  }
  for (j=0; j < m; j++) {
    fst_(b[j], &n, z, &nn);
  }

  transpose (bt,b,m);

  for (i=0; i < m; i++) {
    fstinv_(bt[i], &n, z, &nn);
  }
  
  for (j=0; j < m; j++) {
    for (i=0; i < m; i++) {
      bt[j][i] = bt[j][i]/(diag[i]+diag[j]);
    }
  }
  
  for (i=0; i < m; i++) {
    fst_(bt[i], &n, z, &nn);
  }

  transpose (b,bt,m);

  for (j=0; j < m; j++) {
    fstinv_(b[j], &n, z, &nn);
  }

  umax = 0.0;
  for (j=0; j < m; j++) {
    for (i=0; i < m; i++) {
      if (b[j][i] > umax) umax = b[j][i];
    }
  }
  printf (" umax = %e \n",umax);

  return 0;
}

void transpose (Real **bt, Real **b, int m)
{
  int i, j;
  for (j=0; j < m; j++) {
    for (i=0; i < m; i++) {
      bt[j][i] = b[i][j];
    }
  }
}

Real *createRealArray (int n)
{
  Real *a;
  int i;
  a = (Real *)malloc(n*sizeof(Real));
  for (i=0; i < n; i++) {
    a[i] = 0.0;
  }
  return (a);
}

Real **createReal2DArray (int n1, int n2)
{
  int i, n;
  Real **a;
  a    = (Real **)malloc(n1   *sizeof(Real *));
  a[0] = (Real  *)malloc(n1*n2*sizeof(Real));
  for (i=1; i < n1; i++) {
    a[i] = a[i-1] + n2;
  }
  n = n1*n2;
  memset(a[0],0,n*sizeof(Real));
  return (a);
}
