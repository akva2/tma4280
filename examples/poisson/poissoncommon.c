#include "poissoncommon.h"
#include <math.h>

Vector generateEigenValuesP1D(int m)
{
  Vector result = createVector(m);
  int i;
  for (i=0;i<m;++i)
    result->data[i] = 2.0*(1.0-cos((i+1)*M_PI/(m+1)));

  return result;
}

Matrix generateEigenMatrixP1D(int m)
{ 
  Matrix Q = createMatrix(m,m);
  int i,j;
  for (i=0;i<m;++i)
    for (j=0;j<m;++j)
      Q->data[j][i] = sqrt(2.0/(m+1))*sin((i+1)*(j+1)*M_PI/(m+1));

  return Q;
}
