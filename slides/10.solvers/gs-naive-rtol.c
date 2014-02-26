int GaussSeidel(Matrix A, Vector u, double tol, int maxit)
{
  int it=0, i, j;
  double max = tol+1;
  double rl;
  Vector b = createVector(u->len);
  Vector r = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  rl = maxNorm(b);
  while (max > tol*rl && ++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*v->data[j];
      }
      r->data[i] = u->data[i]-A->data[i][i]*v->data[i];
      u->data[i] /= A->data[i][i];
      v->data[i] = u->data[i];
    }
    max = maxNorm(r);
  }
  freeVector(b);
  freeVector(r);
  freeVector(v);

  return it;
}
