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
