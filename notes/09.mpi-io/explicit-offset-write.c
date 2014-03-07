MPI_File_write_at(fh,rank*mysize*sizeof(double),
                  vec,mysize,MPI_DOUBLE,
                  MPI_STATUS_IGNORE);
