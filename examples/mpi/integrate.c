#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv)
{
  double mypi, pi, h, sum, x, piref, error;
  double t1, t2, dt;
  int n, myid, nproc, i;

  MPI_Init (&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) {
    printf (" Enter the number of intervals:\n");
    i  = scanf ("%d",&n);
    t1 = MPI_Wtime();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n > 0) {
    h = 1.0 / (double)n;
    sum = 0.0;
    for (i = myid+1; i <= n; i += nproc) {
      x = h * ((double)i - 0.5);
      sum = sum + (4.0 / (1.0 + x*x));
    }
    mypi = h * sum;
    MPI_Reduce (&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  if (myid == 0) {
    t2 = MPI_Wtime();
    dt = t2 - t1;
    piref = 4.0 * atan(1.0);
    error = fabs(pi-piref);
    printf (" pi= %e error= %e dt= %e \n", pi, error, dt);
  }

  MPI_Finalize();
  return 0;
}
