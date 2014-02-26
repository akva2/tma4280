int GaussSeidelPoisson1Drb(Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double max = tol+1;
  double rl = maxNorm(u);
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (max > tol && ++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (j=0;j<2;++j) {
#pragma omp parallel for schedule(static)
      for (i=j;i<r->len;i+=2) {
        if (i > 0)
          u->data[i] += v->data[i-1];
        if (i < r->len-1)
          u->data[i] += v->data[i+1];
        r->data[i] = u->data[i]-2.0*v->data[i];
        u->data[i] /= 2.0;
        v->data[i] = u->data[i];
      }
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(r);
  freeVector(v);
  return it;
}
