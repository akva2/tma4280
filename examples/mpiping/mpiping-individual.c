// this give us access to the MPI library
#include <mpi.h>

// for printing etc
#include <stdio.h>

int main(int argc, char** argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int size, rank;

  // Figure out the number of processes and our rank
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size % 2 != 0) {
    printf("Need an even number of processes\n");
    MPI_Finalize();
    return 1;
  }

  int payload;

  if (rank % 2 == 0) {
    // determine rank of receiver
    int sendrank = rank+1;
    if (rank == size-1)
      sendrank = 0;

    // Send data to the process above
    MPI_Send(&rank, 1, MPI_INT, sendrank, 0, MPI_COMM_WORLD);

    // Receive from the process below. Exception for process 0
    // which receives from size-1
    int recvrank = rank-1;
    if (rank == 0)
      recvrank = size-1;

    MPI_Status status;

    MPI_Recv(&payload, 1, MPI_INT, recvrank, 0, MPI_COMM_WORLD, &status);
  } else {
    // Receive from the process below. Exception for process 0
    // which receives from size-1
    int recvrank = rank-1;
    if (rank == 0)
      recvrank = size-1;

    MPI_Status status;
    MPI_Recv(&payload, 1, MPI_INT, recvrank, 0, MPI_COMM_WORLD, &status);

    // determine rank of receiver
    int sendrank = rank+1;
    if (rank == size-1)
      sendrank = 0;

    // Send data to the process above
    MPI_Send(&rank, 1, MPI_INT, sendrank, 0, MPI_COMM_WORLD);
  }

  // print to tty
  printf("process %i: received %i\n", rank, payload);

  // close down MPI
  MPI_Finalize();

  // ay-oh-kay
  return 0;
}
