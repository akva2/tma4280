MPI_Offset mysize = N/size;

MPI_File_seek(fh,rank*mysize*sizeof(double),
              MPI_SEEK_SET);
MPI_File_write(fh,vec,mysize,MPI_DOUBLE,
               MPI_STATUS_IGNORE);
