int cg(Matrix A, Vector b, double tolerance)
{
  int i=0, j;
  double rl;
  Vector r = createVector(b->len);
  Vector p = createVector(b->len);
  Vector buffer = createVector(b->len);
  double dotp = 1000;
  double rdr = dotp;
  copyVector(r,b);
  fillVector(b, 0.0);
  rl = sqrt(dotproduct(r,r));
  while (i < b->len && rdr > tolerance*rl) {
    ++i;
    if (i == 1) {
      copyVector(p,r);
      dotp = dotproduct(r,r);
    } else {
      double dotp2 = dotproduct(r,r);
      double beta = dotp2/dotp;
      dotp = dotp2;
      scaleVector(p,beta);
      axpy(p,r,1.0);
    }
    MxV(buffer, p);
    double alpha = dotp/dotproduct(p,buffer);
    axpy(b,p,alpha);
    axpy(r,buffer,-alpha);
    rdr = sqrt(dotproduct(r,r));
  }
  freeVector(r);
  freeVector(p);
  freeVector(buffer);
  return i;
}
