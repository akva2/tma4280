#include <mpi.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>

/* This program partitions an array of a specified size
 * over the available MPI processes in either a strip
 * or a block fashion as specified.
 * A specified amount of repetitions of this array
 * is then saved as "combined.mat" using MPI-IO.
 */

void createFileView(MPI_Datatype* filetype, int* sizes,
                    int N, int rank, int size, int block)
{
    int gsizes[2], distribs[2], dargs[2];
    gsizes[0] = gsizes[1] = N;
    distribs[0] = MPI_DISTRIBUTE_BLOCK;
    sizes[0] = sizes[1] = 0;
    if( block ) {
        distribs[1] = MPI_DISTRIBUTE_BLOCK;
    }
    else {
        sizes[1] = 1;
        distribs[1] = MPI_DISTRIBUTE_NONE;
    }
    MPI_Dims_create(size,2,sizes);
    dargs[0] = dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;
    MPI_Type_create_darray(size,rank,2,gsizes,distribs,
                           dargs,sizes,MPI_ORDER_C,
                           MPI_DOUBLE,filetype);
    MPI_Type_commit(filetype);
}

void setupMatrix(double** A, int rows, int cols,
                 int startrow, int startcol, int N)
{
    for( int i=0;i<rows;++i)
        for( int j=0;j<cols;++j )
            A[i][j] = (i+startrow)*N+j+startcol;
}

double** createMatrix(int n1, int n2)
{
    int i, n;
    double **a;
    a    = (double **)calloc(n1   ,sizeof(double *));
    a[0] = (double  *)calloc(n1*n2,sizeof(double));

    for (i=1; i < n1; i++)
        a[i] = a[i-1] + n2;

    return (a);
}

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    int N = atoi(argv[1]);
    int blocked = 0;
    if(argc > 2 )
        blocked = atoi(argv[2]);
    int sizes[2];
    MPI_Datatype filetype;
    createFileView(&filetype,sizes,N,rank,size,blocked);

    int periodic[2]; periodic[0] = periodic[1] = 0;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD,2,sizes,
                    periodic,0,&comm);
    int coord[2];
    MPI_Cart_coords(comm,rank,2,coord);

    int rows = N/sizes[0];
    int cols = N/sizes[1];
    int bufsize = rows*cols;
    int K;
    if( argc > 3 )
        K = atoi(argv[3]);
    else
        K = 1;

    double** A = createMatrix(rows,cols);
    setupMatrix(A,rows,cols,coord[0]*rows,coord[1]*cols,N);

    MPI_File f;
    MPI_File_open(MPI_COMM_WORLD,"combined.mat",
                  MPI_MODE_CREATE|MPI_MODE_RDWR,
                  MPI_INFO_NULL,&f);
    MPI_File_set_view(f,0,MPI_DOUBLE,
                      filetype,"native",MPI_INFO_NULL);
    for( int n=0;n<K;++n )
        MPI_File_write(f,A[0],bufsize,
                       MPI_DOUBLE,MPI_STATUS_IGNORE);

    MPI_File_close(&f);

    MPI_Finalize();

    return 0;
}
