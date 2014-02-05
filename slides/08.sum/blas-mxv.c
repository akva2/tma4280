double dotproduct(Vector u, Vector v)
{
  double locres;
  locres = ddot(&u->len, u->data, &u->stride, v->data, &v->stride);
  return locres;

#ifdef HAVE_MPI
  if (u->comm_size > 1) {
    MPI_Allreduce(&locres, &res, 1, MPI_DOUBLE, MPI_SUM, *u->comm);
    return res;
  }
#endif
}

void MxV(Vector u, Matrix A, Vector v)
{
  char trans='N';
  double onef=1.0;
  double zerof=0.0;
  int one=1;
  dgemv(&trans, &A->rows, &A->cols, &onef, A->data[0], &A->rows, v->data,
        &one, &zerof, u->data, &one);
}
