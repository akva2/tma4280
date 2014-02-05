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
