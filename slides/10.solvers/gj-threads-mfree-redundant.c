int GaussJacobiPoisson1D(Vector u, double tol, int maxit)
{
  int it=0, i;
  double rl;
  double max = tol+1;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
#pragma omp parallel for schedule(static)
    for (i=1;i<e->len-1;++i) {
      u->data[i] += e->data[i-1];
      u->data[i] += e->data[i+1];
      u->data[i] /= 2.0;
    }
    axpy(e, u, -1.0);
    max = maxNorm(e);
  }
  freeVector(b);
  freeVector(e);

  return it;
}
