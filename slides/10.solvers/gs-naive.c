int GaussSeidel(Matrix A, Vector u, int maxit)
{
  int it=0, i, j;
  Vector b = createVector(u->len);
  Vector v = createVector(u->len);
  copyVector(b, u);
  fillVector(u, 0.0);
  while (++it < maxit) {
    copyVector(v, u);
    copyVector(u, b);
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*v->data[j];
      }
      u->data[i] /= A->data[i][i];
      v->data[i] = u->data[i];
    }
  }
  freeVector(b);
  freeVector(v);

  return it;
}
