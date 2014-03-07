MPI_File_set_view(fh,rank*sizeof(double),
                  MPI_DOUBLE,filetype,"native",
                  MPI_INFO_NULL);
