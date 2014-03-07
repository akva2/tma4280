MPI_File fh;
MPI_File_open(MPI_COMM_WORLD,"datafile",
              MPI_MODE_WRONLY|MPI_MODE_CREATE,
              info,&fh);
