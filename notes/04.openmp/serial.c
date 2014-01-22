#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

typedef double(*function_t)(double x);

double integrate(double x0, double x1, int n, function_t f)
{
  double h = (x1-x0)/n;
  double result=0.0;
#pragma omp parallel for schedule(static) reduction(+:result)
  for (int i=0;i<n;++i) {
    double x = x0 + (i+0.5)*h;
    result += h*f(x);
  }

  return result;
}

double myf(double x)
{
  return 4.0/(1.0+x*x);
}

int main(int argc, char** argv)
{
  int n;
  double mypi;

  n = atoi(argv[1]);
  if (n <= 0) {
    printf("Error, %i intervals make no sense, bailing\n",n);
    exit(1);
  }

  double start = omp_get_wtime();
  mypi = integrate(0,1,n,myf);
  printf("elapsed: %f\n",omp_get_wtime()-start);

  printf("%1.16f\n",fabs(mypi-4.0*atan(1.0)));

  return 0;
}
