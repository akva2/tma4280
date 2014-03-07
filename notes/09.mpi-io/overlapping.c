MPI_Request req;
MPI_File_iwrite(f,vec,mysize,MPI_DOUBLE,&req);
doSomething();
MPI_Wait(&req,MPI_STATUS_IGNORE);
