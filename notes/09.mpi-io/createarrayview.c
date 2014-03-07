int gsizes[2], distribs[2], dargs[2];
gsizes[0] = gsizes[1] = N;
distribs[0] = MPI_DISTRIBUTE_BLOCK;
distribs[1] = MPI_DISTRIBUTE_BLOCK;
dargs[0] = dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;
MPI_Type_create_darray(size,rank,2,gsizes,
                       distribs,dargs,sizes,
                       MPI_ORDER_C,
                       MPI_DOUBLE,filetype);
MPI_Type_commit(&filetype);
