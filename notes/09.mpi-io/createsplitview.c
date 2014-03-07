MPI_Datatype filetype;
MPI_Type_create_resized(MPI_DOUBLE,0,
                        size*sizeof(double),
                        &filetype);
MPI_Type_commit(&filetype);
