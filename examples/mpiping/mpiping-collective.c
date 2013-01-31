// this give us access to the MPI library
#include <mpi.h>

// for printing etc
#include <stdio.h>

int main(int argc, char** argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int size, rank;

  // Figure out the number of processes and our rank in the world group
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size % 2) {
    printf("Need an even number of processes\n");
    MPI_Finalize();
    return 1;
  }

  // setup new communicators
  MPI_Comm twocomm;
  MPI_Comm_split(MPI_COMM_WORLD, rank/2, rank%2, &twocomm);

  int senddata[2], recvdata[2];
  senddata[(rank+1)%2] = rank;
  senddata[rank%2] = 0;
  MPI_Alltoall(senddata, 1, MPI_INT, recvdata, 1, MPI_INT, twocomm);

  // print to tty
  printf("process %i: received %i\n", rank, recvdata[(rank+1)%2]);

  // close down MPI
  MPI_Finalize();

  // ay-oh-kay
  return 0;
}
