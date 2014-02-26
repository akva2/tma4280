int GaussJacobi(Matrix A, Vector u, int maxit)
{
  int it=0, i, j;
  Vector b = createVector(u->len);
  Vector e = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (++it < maxit) {
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
  }
  freeVector(b);
  freeVector(e);

  return it;
}
