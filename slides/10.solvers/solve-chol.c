void llsolve(Matrix A, Vector x)
{
  int one=1;
  int info;
  char uplo='L';
  dposv(&uplo,&x->len,&one,A->data[0],&x->len,
        x->data,&x->len,&info);
}


