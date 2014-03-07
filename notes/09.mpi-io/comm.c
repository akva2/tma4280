int periodic[2]; periodic[0] = periodic[1] = 0;
MPI_Comm comm;
MPI_Cart_create(MPI_COMM_WORLD,2,
                dims,periodic,0,&comm);
