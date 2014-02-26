int GaussJacobiPoisson1D(Matrix A, Vector u, double tol, int maxit)
{
  ...
#pragma omp parallel for schedule(static)
    for (i=0;i<e->len;++i) {
      if (i > 0)
        u->data[i] += e->data[i-1];
      if (i < e->len-1)
        u->data[i] += e->data[i+1];
      u->data[i] /= (2.0+alpha);
    }
  ...
}
