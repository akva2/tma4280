#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

typedef double(*function_t)(double x);

double integrate(double x0, double x1, int n, function_t f)
{
  double h = (x1-x0)/n;
  double result=0.f;
#pragma omp parallel for schedule(static) reduction(+:result)
  for (int i=0;i<n;++i) {
    double x = x0 + (i+.5f)*h;
    result += h*f(x);
  }

  return result;
}

double myf(double x)
{
  return 4.f/(1.f+x*x);
}

int main(int argc, char** argv)
{
  int n;
  double mypi;

/*  printf("Enter the number of intervals:\n");*/
/*  scanf("%d", &n);*/
  n = atoi(argv[1]);
  if (n <= 0) {
    printf("Error, %i intervals make no sense, bailing\n",n);
    exit(1);
  }

  double start = omp_get_wtime();
  mypi = integrate(0,1,n,myf);
  printf("elapsed: %f\n",omp_get_wtime()-start);

/*  printf("pi = %1.16f, error= %1.16f\n",mypi,fabs(mypi-4.f*atan(1.f)));*/
  printf("%1.16f\n",fabs(mypi-4.f*atan(1.f)));

  return 0;
}
