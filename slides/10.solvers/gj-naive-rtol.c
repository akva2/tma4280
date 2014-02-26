int GaussJacobi(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double rl;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  Vector r = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  double max = tol+1;
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
    copyVector(e, u);
    copyVector(u, b);
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*e->data[j];
      }
      r->data[i] = u->data[i]-A->data[i][i]*e->data[i];
      u->data[i] /= A->data[i][i];
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(e);
  freeVector(r);

  return it;
}
