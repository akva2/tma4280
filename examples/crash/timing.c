#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
double WallTime ()
{
  struct timeval tmpTime; gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
}
double dot(double* u, double* v, int len)
{
  double result=0;
  int i;
  for (i=0;i<len;++i) result += u[i]*v[i];
  return result;
}
int main(int argc, char** argv)
{
  int times, N, i;
  double* u;
  if (argc < 2) {
    printf("need two parameter, the number of times to loop and the vector length\n");
    return 1;
  }
  times = atoi(argv[1]);
  N = atoi(argv[2]);
  u = (double*)malloc(N*sizeof(double));
  srand(WallTime());
  for (i=0;i<N;++i) u[i] = (double)rand() / RAND_MAX;
  printf("Performing %i loops\n", times);
  printf("Vector length: %i\n", N);
  double now = WallTime();
  double b=0;
  for (i=0;i<times;++i) b += dot(u, u, N);
  printf("Used %f seconds, sum %f\n", WallTime()-now, b);
  free(u);
  return 0;
}
