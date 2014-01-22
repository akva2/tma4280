double dosum(double** A, double** v, int K, int N)
{
    double alpha=0;
    double** temp = createMatrix(K,N);
#pragma omp parallel for schedule(static) \
	reduction(+:alpha)
    for( int i=0;i<K;++i ) {
        MxV(temp[i],A,v[i],N);
        alpha += innerproduct(temp[i],v[i],N);
    }

    return alpha;
}

double dosumMPI(double** A, double** v, int myK, int N)
{
  /* assumes we have divided the data */
  /* in vectors per proc */
  double myalpha = dosum(A,V,myK,N);
  double alpha;
  MPI_Allreduce(&myalpha,&alpha,1,MPI_DOUBLE,
		  		MPI_SUM,MPI_COMM_WORLD);

  return alpha;
}

