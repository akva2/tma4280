#include <stdio.h>

const double A[][3] = {{0.3, 0.4, 0.3},
                       {0.7, 0.1, 0.2},
                       {0.5, 0.5, 0.0}};

const double x[3] = {1.0, 1.0, 1.0};

int main(int argc, char** argv)
{
  double y[3];
  int i, j;
  for (i=0; i < 3; ++i) {
    y[i] = 0.0;
    for (j=0; j < 3; ++j)
      y[i] += A[i][j]*x[j];
  }
  printf("result: y = %f %f %f\n", y[0], y[1], y[2]);
  return 0;
}
