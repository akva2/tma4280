void lusolve(Matrix A, Vector x)
{
  int* ipiv = malloc(x->len*sizeof(int));
  int one=1;
  int info;
  dgesv(&x->len,&one,A->data[0],&x->len,
        *ipiv,x->data,&x->len,&info);
  free(ipiv);
}
