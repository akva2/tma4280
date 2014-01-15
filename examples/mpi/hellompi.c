#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv)
{
  int rank, size, i, tag;
  MPI_Status status;
  char message[20];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  tag = 100;

  if (rank == 0) {
    strcpy(message, "Hello world!");
    for (i=1; i < size; ++i)
      MPI_Send(message, 13, MPI_CHAR, i, tag, MPI_COMM_WORLD);
  } else
    MPI_Recv(message, 13, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);

  printf("process %d: %s\n", rank, message);

  MPI_Finalize();
  return 0;
}
