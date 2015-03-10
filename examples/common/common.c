#include "common.h"
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include "blaslapack.h"
#include <memory.h>

#ifdef HAVE_MPI
MPI_Comm WorldComm;
MPI_Comm SelfComm;
#endif

void init_app(int argc, char** argv, int* rank, int* size)
{
#ifdef HAVE_MPI
#ifdef HAVE_OPENMP
  int aquired;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &aquired);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  if (*rank == 0) {
    printf("aquired MPI threading level: ");
    if (aquired == MPI_THREAD_SINGLE)
      printf("MPI_THREAD_SINGLE\n");
    if (aquired == MPI_THREAD_FUNNELED)
      printf("MPI_THREAD_FUNNELED\n");
    if (aquired == MPI_THREAD_SERIALIZED)
      printf("MPI_THREAD_SERIALIZED\n");
    if (aquired == MPI_THREAD_MULTIPLE)
      printf("MPI_THREAD_MULTIPLE\n");
  }
#else
  MPI_Init(&argc, &argv);
#endif
  MPI_Comm_size(MPI_COMM_WORLD, size);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_dup(MPI_COMM_WORLD, &WorldComm);
  MPI_Comm_dup(MPI_COMM_SELF, &SelfComm);
#else
  *rank = 0;
  *size = 1;
#endif
}

void close_app()
{
#ifdef HAVE_MPI
  MPI_Comm_free(&WorldComm);
  MPI_Comm_free(&SelfComm);
  MPI_Finalize();
#endif
}

Vector createVector(int len)
{
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->data = calloc(len, sizeof(double));
  result->len = result->glob_len = len;
  result->stride = 1;
  result->comm = NULL;
  result->comm_size = 1;
  result->comm_rank = 0;
  result->displ = NULL;
  result->sizes = NULL;

  return result;
}

#ifdef HAVE_MPI
Vector createVectorMPI(int glob_len, MPI_Comm* comm, int allocdata, int pad)
{
  int padadd=2;
  Vector result = (Vector)calloc(1, sizeof(vector_t));
  result->comm = comm;
  MPI_Comm_size(*comm, &result->comm_size);
  MPI_Comm_rank(*comm, &result->comm_rank);
  splitVector(glob_len, result->comm_size, &result->sizes, &result->displ);
  if (result->comm_rank == 0)
    padadd--;
  if (result->comm_rank == result->comm_size-1)
    padadd--;
  result->len = result->sizes[result->comm_rank]+padadd;
  if (allocdata)
    result->data = calloc(result->len, sizeof(double));
  else
    result->data = NULL;
  result->glob_len = glob_len;
  result->stride = 1;

  return result;
}
#endif

void splitVector(int globLen, int size, int** len, int** displ)
{
  int i;
  *len = calloc(size,sizeof(int));
  *displ = calloc(size,sizeof(int));
  for (i=0;i<size;++i) {
    (*len)[i] = globLen/size;
    if (globLen % size && i >= (size - globLen % size))
      (*len)[i]++;
    if (i < size-1)
      (*displ)[i+1] = (*displ)[i]+(*len)[i];
  }
}

void freeVector(Vector vec)
{
  free(vec->data);
  free(vec->sizes);
  free(vec->displ);
  free(vec);
}

Matrix createMatrix(int n1, int n2)
{
  int i;
  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));
  result->rows = n1;
  result->cols = n2;
  result->data = (double **)calloc(n2   ,sizeof(double *));
  result->data[0] = (double  *)calloc(n1*n2,sizeof(double));
  for (i=1; i < n2; i++)
    result->data[i] = result->data[i-1] + n1;
  result->as_vec = (Vector)malloc(sizeof(vector_t));
  result->as_vec->data = result->data[0];
  result->as_vec->len = n1*n2;
  result->as_vec->stride = 1;
  result->as_vec->comm_size = 1;
  result->as_vec->comm_rank = 0;
  result->col = malloc(n2*sizeof(Vector));
  for (i=0;i<n2;++i) {
    result->col[i] = malloc(sizeof(vector_t));
    result->col[i]->len = n1;
    result->col[i]->stride = 1;
    result->col[i]->data = result->data[i];
  }
  result->row = malloc(n1*sizeof(Vector));
  for (i=0;i<n1;++i) {
    result->row[i] = malloc(sizeof(vector_t));
    result->row[i]->len = n2;
    result->row[i]->stride = n1;
    result->row[i]->data = result->data[0]+i;
  }

  return result;
}

#ifdef HAVE_MPI
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm)
{
  int i, n12;

  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));
  result->as_vec = createVectorMPI(N1*N2, comm, 0, 0);
  n12 = n1;
  if (n1 == -1)
    n1 = result->as_vec->len/N2;
  else
    n2 = result->as_vec->len/N1;

  result->rows = n1;
  result->cols = n2;
  result->glob_rows = N1;
  result->glob_cols = N2;
  result->data = (double **)calloc(n2   ,sizeof(double *));
  result->data[0] = (double  *)calloc(n1*n2,sizeof(double));
  result->as_vec->data = result->data[0];
  for (i=1; i < n2; i++)
    result->data[i] = result->data[i-1] + n1;
  result->col = malloc(n2*sizeof(Vector));
  for (i=0;i<n2;++i) {
    if (n12 == N1)
      result->col[i] = createVectorMPI(N1, &SelfComm, 0, 0);
    else
      result->col[i] = createVectorMPI(N1, comm, 0, 0);
    result->col[i]->data = result->data[i];
  }
  result->row = malloc(n1*sizeof(Vector));
  for (i=0;i<n1;++i) {
    if (n12 == N1)
      result->row[i] = createVectorMPI(N2, comm, 0, 0);
    else
      result->row[i] = createVectorMPI(N2, &SelfComm, 0, 0);
    result->row[i]->data = result->data[0]+i;
    result->row[i]->stride = n1;
  }

  return result;
}

Matrix createMatrixMPICart(int N1, int N2, MPI_Comm* comm, int pad)
{
  int pads[2]= {2*pad, 2*pad};  // padding in rows and cols
  int dims[2];                  // dimensions of cartesian topology
  int periods[2];               // periodicity in cartesian topology (ignored)
  int coords[2];                // coordinates in cartesian topology
  int* split[2];                // block sizes in both directions
  int* disp[2];                 // displacements in both directions
  int i;

  // sanity check communicator
  MPI_Topo_test(*comm,&dims[0]);
  if (dims[0] != MPI_CART) {
    printf("Error in createMatrixMPICart: Communicator has no cartesian topology\n");
    return NULL;
  }

  // get cartesian info
  MPI_Cart_get(*comm, 2, dims, periods, coords);

  // calculate padding
  if (pad) {
    for (i=0;i<2;++i) {
      if (coords[i] == 0)
        pads[i]--;
      if (coords[i] == dims[i]-1)
        pads[i]--;
    }
  }

  // calculate un-padded block sizes
  splitVector(N1, dims[0], &split[0], &disp[0]);
  splitVector(N2, dims[1], &split[1], &disp[1]);

  // allocate data
  Matrix result = (Matrix)calloc(1, sizeof(matrix_t));
  result->as_vec = createVectorMPI(N1*N2, comm, 0, 0);
  result->rows = split[1][coords[1]]+pads[1];
  result->cols = split[0][coords[0]]+pads[0];
  result->as_vec->len = result->rows*result->cols;
  MPI_Comm_rank(*comm,&result->as_vec->comm_rank);
  result->glob_rows = N1;
  result->glob_cols = N2;
  result->data = calloc(result->cols,sizeof(double*));
  result->as_vec->data = result->data[0] = calloc(result->as_vec->len,sizeof(double));
  for (i=1; i < result->cols; ++i)
    result->data[i] = result->data[i-1] + result->rows;

  // allocate column vectors
  result->col = calloc(result->cols,sizeof(Vector));
  for (i=0;i<result->cols;++i) {
    result->col[i] = createVectorMPI(N1,comm, 0, 0);
    result->col[i]->data = result->data[i];
    result->col[i]->len = result->rows;
    result->col[i]->stride = 1;
    result->col[i]->comm_rank = result->as_vec->comm_rank;
    result->col[i]->displ = calloc(dims[1],sizeof(int));
    memcpy(result->col[i]->displ, disp[1], dims[1]*sizeof(int));
    result->col[i]->sizes = calloc(dims[1],sizeof(int));
    memcpy(result->col[i]->sizes, split[1], dims[1]*sizeof(int));
  }

  // allocate row vectors
  result->row = calloc(result->rows,sizeof(Vector));
  for (i=0;i<result->rows;++i) {
    result->row[i] = createVectorMPI(N2,comm, 0, 0);
    result->row[i]->data = result->data[0]+i;
    result->row[i]->stride = result->rows;
    result->row[i]->len = result->cols;
    result->row[i]->comm_rank = result->as_vec->comm_rank;
    result->row[i]->displ = calloc(dims[0],sizeof(int));
    memcpy(result->row[i]->displ, disp[0], dims[0]*sizeof(int));
    result->row[i]->sizes = calloc(dims[0],sizeof(int));
    memcpy(result->row[i]->sizes, split[0], dims[0]*sizeof(int));
  }

  return result;
}
#endif

void freeMatrix(Matrix A)
{
  int i;
  for (i=0;i<A->cols;++i)
    free(A->col[i]);
  for (i=0;i<A->rows;++i)
    free(A->row[i]);
  free(A->col);
  free(A->row);
  free(A->as_vec);
  free(A->data[0]);
  free(A->data);
  free(A);
}

int getMaxThreads()
{
#ifdef HAVE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int getCurrentThread()
{
#ifdef HAVE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

double WallTime ()
{
#ifdef HAVE_OPENMP
  return omp_get_wtime();
#else
  struct timeval tmpTime;
  gettimeofday(&tmpTime,NULL);
  return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
#endif
}

double dotproduct(Vector u, Vector v)
{
  double locres, res;
  locres = ddot(&u->len, u->data, &u->stride, v->data, &v->stride);

#ifdef HAVE_MPI
  if (u->comm_size > 1) {
    MPI_Allreduce(&locres, &res, 1, MPI_DOUBLE, MPI_SUM, *u->comm);
    return res;
  }
#endif

  return locres;
}

void MxV(Vector u, Matrix A, Vector v, double alpha, double beta, char trans)
{
  dgemv(&trans, &A->rows, &A->cols, &alpha, A->data[0], &A->rows, v->data,
        &v->stride, &beta, u->data, &u->stride);
}

void MxM(Matrix U, Matrix A, Matrix V, double alpha, double beta,
         char transA, char transV)
{
  int M,N,K,lda,ldb;
  if( transA == 'N' ) {
    M = A->rows;
    lda = M;
  }
  else {
    M = A->cols;
  }
  if( transV == 'N' ) {
    N = V->cols;
    K = V->rows;
    ldb = K;
  }
  else {
    N = V->rows;
    K = V->cols;
    ldb = N;
  }
  if( transA != 'N')
    lda = K;
  dgemm(&transA, &transV, &M, &N, &K, &alpha,
        A->data[0], &lda, V->data[0], &ldb, &beta, U->data[0], &M);
}

void diag(Matrix A, int diag, double value)
{
  int i;
  for (i=0;i<A->rows;++i) {
    if (i+diag >= 0 && i+diag < A->cols)
      A->data[i+diag][i] = value;
  }
}

Vector equidistantMesh(double x0, double x1, int N)
{
  double h = (x1-x0)/N;
  Vector result = createVector(N+1);
  int i;

  for (i=0;i<N+1;++i)
    result->data[i] = x0+i*h;

  return result;
}

void evalMeshDispl(Vector u, Vector grid, function1D func)
{
  int i;
  int startofs=u->displ[u->comm_rank];
  if (u->comm_rank > 0)
    startofs--;
  for (i=0;i<u->len;++i)
    u->data[i] = func(grid->data[i+startofs]);
}

void evalMesh2Displ(Matrix b, Vector grid, function2D func, int* mpi_top_coords)
{
  int i, j;
  for (i=1;i<b->cols-1;++i) {
    int dispx=mpi_top_coords[0]==0?0:1;
    for (j=1;j<b->rows-1;++j) {
      int dispy=mpi_top_coords[1]==0?0:1;
      double x = grid->data[i+b->row[0]->displ[mpi_top_coords[0]]-dispx];
      double y = grid->data[j+b->col[0]->displ[mpi_top_coords[1]]-dispy];
      b->data[i][j] = func(x,y);
    }
  }
}

void evalMeshInternal(Vector u, Vector grid, function1D func)
{
  int i;
  for (i=1;i<grid->len-1;++i)
    u->data[i-1] = func(grid->data[i]);
}

void evalMeshInternal2(Matrix u, Vector grid, function2D func, int boundary)
{
  int i, j;
  for (i=1;i<grid->len-1;++i)
    for (j=1;j<grid->len-1;++j)
      u->data[j-!boundary][i-!boundary] = func(grid->data[i], grid->data[j]);
}

void scaleVector(Vector u, double alpha)
{
  dscal(&u->len, &alpha, u->data, &u->stride);
}

void axpy(Vector y, const Vector x, double alpha)
{
  daxpy(&x->len, &alpha, x->data, &x->stride, y->data, &y->stride);
}


void lusolve(Matrix A, Vector x, int** ipiv)
{
  int one=1;
  int info;
  if (*ipiv == NULL) {
    *ipiv = malloc(x->len*sizeof(int));
    dgesv(&x->len,&one,A->data[0],&x->len,*ipiv,x->data,&x->len,&info);
  } else {
    char trans='N';
    dgetrs(trans,&x->len,&one,A->data[0],&x->len,*ipiv,x->data,&x->len,&info);
  }
  if (info < 0)
    printf("error solving linear system [%i]\n", info);
}

void llsolve(Matrix A, Vector x, int prefactored)
{
  int one=1;
  int info;
  char uplo='L';
  if (prefactored)
    dpotrs(&uplo,&x->len,&one,A->data[0],&x->len,x->data,&x->len,&info);
  else
    dposv(&uplo,&x->len,&one,A->data[0],&x->len,x->data,&x->len,&info);
  if (info < 0)
    printf("error solving linear system [%i]\n", info);
}

void lutsolve(Matrix A, Vector x, char uplo)
{
  char trans='N';
  char diag='N';
  int one=1;
  int info;
  dtrtrs(&uplo, &trans, &diag, &A->rows, &one,
         A->data[0], &A->rows, x->data, &A->rows, &info);
}

double maxNorm(const Vector x)
{
  // idamax is a fortran function, and the first index is 1
  // since indices in C are 0 based, we have to decrease it 
  double result = fabs(x->data[idamax(&x->len, x->data, &x->stride)-1]);
#ifdef HAVE_MPI
  if (x->comm_size > 1) {
    double r2=result;
    MPI_Allreduce(&r2, &result, 1, MPI_DOUBLE, MPI_MAX, *x->comm);
  }
#endif

  return result;
}

void copyVector(Vector y, const Vector x)
{
  dcopy(&y->len, x->data, &x->stride, y->data, &y->stride);
}

void fillVector(Vector x, double alpha)
{
  int i;
  for (i=0;i<x->len;++i)
    x->data[i*x->stride] = alpha;
}

Matrix subMatrix(const Matrix A, int r_ofs, int r,
                 int c_ofs, int c)
{
  int i, j;
  Matrix result = createMatrix(r, c);
  for (i=0;i<c;++i)
    for (j=0;j<r;++j)
      result->data[i][j] = A->data[i+c_ofs][j+r_ofs];

  return result;
}

void transposeMatrix(Matrix A, const Matrix B)
{
  int i,j;
  for (i=0;i<B->rows;++i)
    for (j=0;j<B->cols;++j)
      A->data[i][j] = B->data[j][i];
}

void saveMatrix(const Matrix A, char* file)
{
  FILE* f = fopen(file,"wb");
  int i,j;
  for (i=0;i<A->rows;++i) {
    for (j=0;j<A->cols;++j)
      fprintf(f,"%1.16lf ",A->data[j][i]);
    fprintf(f,"\n");
  }
}

void saveMatrixMPI(const Matrix A, char* file)
{
  int rank, size, i, j;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  int sizes[2];
  MPI_Topo_test(*A->as_vec->comm,&sizes[0]);
  if (sizes[0] != MPI_CART) {
    printf("Expected a cartesian communicator, no data written\n");
    return;
  }

  int periods[2], coords[2];
  MPI_Cart_get(*A->as_vec->comm, 2, sizes, periods, coords);

  int* split1;
  int* displ1;
  int* split2;
  int* displ2;
  splitVector(A->glob_cols, sizes[0], &split1, &displ1);
  splitVector(A->glob_rows, sizes[1], &split2, &displ2);

  MPI_File f;
  MPI_File_open(*A->as_vec->comm,file,MPI_MODE_WRONLY|MPI_MODE_CREATE,
                MPI_INFO_NULL,&f);

  MPI_Datatype filetype;
  int gsizes[2], distribs[2], dargs[2];
  gsizes[0] = A->glob_cols; gsizes[1] = A->glob_rows;
  distribs[0] = MPI_DISTRIBUTE_BLOCK;
  distribs[1] = MPI_DISTRIBUTE_BLOCK;
  dargs[0] = dargs[1] = MPI_DISTRIBUTE_DFLT_DARG;
  MPI_Datatype datatype;
  MPI_Type_contiguous(13, MPI_CHAR, &datatype);
  MPI_Type_commit(&datatype);
  MPI_Type_create_darray(size,rank,2,gsizes,distribs,dargs,sizes,
                         MPI_ORDER_FORTRAN,datatype,&filetype);
  MPI_Type_commit(&filetype);
  MPI_File_set_view(f,0,datatype,filetype,"native",MPI_INFO_NULL);

  int startcoord_x = displ1[coords[0]];
  int startcoord_y = displ2[coords[1]];
  for (j=0;j<A->cols;++j) {
    for (i=0;i<A->rows;++i) {
      char num[20];
      sprintf(num,"%e ",A->data[j][i]);
      MPI_File_write(f,num,1,datatype,MPI_STATUS_IGNORE);
    }
  }
  MPI_File_close(&f);
}

void collectVector(Vector u)
{
#ifdef HAVE_MPI
  int source, dest;
  // west 
  MPI_Cart_shift(*u->comm, 0,   -1, &source, &dest);
  MPI_Sendrecv(u->data+1,        1, MPI_DOUBLE, dest,   0,
               u->data+u->len-1, 1, MPI_DOUBLE, source, 0, *u->comm, MPI_STATUS_IGNORE);

  // east
  MPI_Cart_shift(*u->comm,  0,   1, &source, &dest);
  MPI_Sendrecv(u->data+u->len-2, 1, MPI_DOUBLE, dest,   1,
               u->data,          1, MPI_DOUBLE, source, 1, *u->comm, MPI_STATUS_IGNORE);
#endif
}

void printMatrix(Matrix u)
{
  int i,j;
  for (i=0;i<u->rows;++i) {
    for (j=0;j<u->cols;++j)
      printf("%f ", u->data[j][i]);
    printf("\n");
  }
  printf("------\n");
}

void collectMatrix(Matrix u)
{
#ifdef HAVE_MPI
  int source, dest;

  // east
  MPI_Cart_shift(*u->as_vec->comm, 0, 1, &source, &dest);
  MPI_Sendrecv(u->data[u->cols-2], u->rows, MPI_DOUBLE, dest, 0,
               u->data[0], u->rows, MPI_DOUBLE, source, 0,
               *u->as_vec->comm, MPI_STATUS_IGNORE);

  // west
  MPI_Cart_shift(*u->as_vec->comm, 0, -1, &source, &dest);
  MPI_Sendrecv(u->data[1], u->rows, MPI_DOUBLE, dest, 1,
               u->data[u->cols-1], u->rows, MPI_DOUBLE, source, 1,
               *u->as_vec->comm, MPI_STATUS_IGNORE);

  Vector sendBuf = createVector(u->cols);
  Vector recvBuf = createVector(u->cols);

  // north
  MPI_Cart_shift(*u->as_vec->comm, 1, -1, &source, &dest);
  if (dest != MPI_PROC_NULL)
    copyVector(sendBuf,u->row[1]);
  MPI_Sendrecv(sendBuf->data, sendBuf->len, MPI_DOUBLE, dest, 2,
               recvBuf->data, recvBuf->len, MPI_DOUBLE, source, 2,
               *u->as_vec->comm, MPI_STATUS_IGNORE);
  if (source != MPI_PROC_NULL)
    copyVector(u->row[u->rows-1], recvBuf);

  // south
  MPI_Cart_shift(*u->as_vec->comm, 1, 1, &source, &dest);
  if (dest != MPI_PROC_NULL)
    copyVector(sendBuf, u->row[u->rows-2]);
  MPI_Sendrecv(sendBuf->data, sendBuf->len, MPI_DOUBLE, dest, 2,
               recvBuf->data, recvBuf->len, MPI_DOUBLE, source, 2,
               *u->as_vec->comm, MPI_STATUS_IGNORE);
  if (source != MPI_PROC_NULL)
    copyVector(u->row[0], recvBuf);

  freeVector(sendBuf);
  freeVector(recvBuf);
#endif
}

void printVector(const Vector u)
{
  int i,j;
  for (j=0;j<u->comm_size;++j) {
    if (j == u->comm_rank) {
      printf("rank %i: ", u->comm_rank);
      for (i=0;i<u->len;++i)
        printf("%f ", u->data[i]);
      printf("\n");
    }
#ifdef HAVE_MPI
    if (u->comm)
      MPI_Barrier(*u->comm);
#endif
  }
}

Vector cloneVector(const Vector u)
{
  Vector result = createVector(u->len);
  result->comm_size = u->comm_size;
  result->comm_rank = u->comm_rank;
  result->comm = u->comm;
  result->glob_len = u->glob_len;

  return result;
}

Matrix cloneMatrix(const Matrix u)
{ 
  Matrix result;
  int i;
  result = createMatrix(u->rows, u->cols);
  result->as_vec->glob_len = u->as_vec->glob_len;
  result->as_vec->comm = u->as_vec->comm;
  result->as_vec->comm_rank = u->as_vec->comm_rank;
  result->as_vec->comm_size = u->as_vec->comm_size;
  result->glob_rows = u->glob_rows;
  result->glob_cols = u->glob_cols;
  for (i=0;i<result->cols;++i) {
    result->col[i]->glob_len = u->col[i]->glob_len;
    result->col[i]->comm = u->col[i]->comm;
  }
  for (i=0;i<result->rows;++i) {
    result->row[i]->glob_len = u->row[i]->glob_len;
    result->row[i]->comm= u->row[i]->comm;
  }
  
  return result; 
} 
