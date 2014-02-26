int GaussJacobiPoisson1D(Vector u, double tol, int maxit)
{
  int it=0, i;
  Vector b = cloneVector(u);
  Vector e = cloneVector(u);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  while (max > tol && ++it < maxit) {
    copyVector(e, u);
    collectVector(e); // only change
    copyVector(u, b);
#pragma omp parallel for schedule(static)
    for (i=1;i<e->len-1;++i) {
      u->data[i] += e->data[i-1];
      u->data[i] += e->data[i+1];
      u->data[i] /= (2.0+alpha);
    }
    axpy(e, u, -1.0);
    e->data[0] = e->data[e->len-1] = 0.0;
    max = maxNorm(e);
  }
  freeVector(b);
  freeVector(e);

  return it;
}
