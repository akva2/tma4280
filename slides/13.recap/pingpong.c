#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
  double t1, *elap;
  int trials, rank, size, i;

  MPI_Init(&argc, &argv); 

  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if (size != 2) {
    printf("only meant to run with 2 processes!\n");
    return 1;
  }

  if (argc < 2) {
    if (rank == 0)
      printf("need one argument, the number of trials\n");
    MPI_Finalize();
    return 2;
  }

  trials = atoi(argv[1]);
  elap = malloc(trials*sizeof(double));

  char send, recv;
  for (i=0;i<trials;++i) {
    if (rank == 0) {
      t1 = MPI_Wtime(); 
      MPI_Send(&send, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&recv, 1, MPI_CHAR, 1, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      elap[i] = MPI_Wtime()-t1;
    } else {
      MPI_Recv(&recv, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(&send, 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
  }

  if (rank == 0) {
    t1=0.0;
    for (i=0;i<trials;++i)
      t1 += elap[i];

    t1 /= 2*trials;
    printf("average latency: %1.10e\n", t1);
  }

  MPI_Finalize();
}
