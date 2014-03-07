#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* This program partions a vector of a specified length
 * over the available MPI processes in a cyclic fashion.
 *      |p0|p1|..|pn|p0|p1|..|pn|
 * A specified amount of repetitions of this vector 
 *  is then saved as "fileview.vec" using MPI-IO.
 */

void setupVector(double* V, int rank, int size, int N)
{
    int i;
    for( i=0;i<N;++i )
        V[i] = i*size+rank;
}

int main(int argc, char** argv)
{
    if( argc < 2 ) {
        printf("need atleast one parameter, N\n");
        exit(1);
    }
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int N = atoi(argv[1]);
    int perproc=N/size;
    int K;
    if( argc > 2 )
        K = atoi(argv[2]);
    else
        K = 1;

    double* vec = (double*)malloc(perproc*sizeof(double));
    setupVector(vec,rank,size,perproc);

    MPI_Datatype filetype;
    MPI_Type_create_resized(MPI_DOUBLE,0,
                            size*sizeof(double),
                            &filetype);
    MPI_Type_commit(&filetype);

    MPI_File f;

    MPI_File_open(MPI_COMM_WORLD,"fileview.vec",
                  MPI_MODE_CREATE|MPI_MODE_RDWR,
                  MPI_INFO_NULL,&f);

    /* NOTE: THIS TRIGGERS A BUG IN OPENMPI v1.3 */
    MPI_File_set_view(f,rank*sizeof(double),MPI_DOUBLE,
                      filetype,"native",MPI_INFO_NULL);
    int n;
    for( n=0;n<K;++n )
        MPI_File_write(f,vec,perproc,
                       MPI_DOUBLE,MPI_STATUS_IGNORE);

    MPI_File_close(&f);

    MPI_Finalize();

    return 0;
}
