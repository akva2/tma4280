#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"

typedef double(*function_t)(double x);

double integrate(double x0, double x1, int n, function_t f)
{
  double sum=0.0;
  double h = 1.0 / (double)n;
  double x;
  int i;

#pragma omp parallel for schedule(static) reduction(+:sum) private(x)
  for (i = 0; i < n; ++i) {
    x = h * ((double)i + 0.5);
    sum += f(x);
  }

  return sum*h;
}

double myf(double x)
{
  return 4.0/(1.0+x*x);
}

int main(int argc, char **argv)
{
  double pi, h, sum, x, piref, error;
  double t1, t2, dt;
  int n, myid, nproc, i;

  if (argc < 2) {
    printf("need at least one parameter, the number of intervals\n");
    return 1;
  }

  n = atoi(argv[1]);

  if (n <= 0) {
    printf("Error, %i intervals make no sense, bailing\n", n);
    return 2;
  }

  t1 = WallTime();
  pi = integrate(0.0, 1.0, n, myf);
  t2 = WallTime();
  dt = t2 - t1;

  piref = 4.0 * atan(1.0);
  error = fabs(pi-piref);
  printf ("pi=%e error=%e dt=%e \n", pi, error, dt);

  return 0;
}
