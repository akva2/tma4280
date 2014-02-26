int GaussJacobi(Matrix A, Vector u, double tol, int maxit)
{
  ...
#pragma omp parallel for schedule(static)
    for (i=0;i<A->rows;++i) {
      for (j=0;j<A->cols;++j) {
        if (j != i)
          u->data[i] -= A->data[j][i]*e->data[j];
      }
      r->data[i] = u->data[i]-A->data[i][i]*e->data[i];
      u->data[i] /= A->data[i][i];
    }
  ...
}
