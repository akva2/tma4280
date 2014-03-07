MPI_File_seek(fh,rank*mysize*sizeof(double),
              MPI_SEEK_SET);
MPI_File_write_all(fh,vec,mysize,
                   MPI_DOUBLE,MPI_STATUS_IGNORE);
