#pragma once

#ifndef USE_MKL
  #define ddot ddot_
  #define dgemv dgemv_
  #define dgemm dgemm_
#endif

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm WorldComm;
extern MPI_Comm SelfComm;
#endif

// blas prototypes
double ddot(int* N, double* dx, int* incx, double* dy, int* incy);
double dgemv(char* trans, int* M, int* N, double* alpha, double* A,
             int* LDA, double* x, int* incx,
             double* beta, double* y, int* incy);
void dgemm(char* transA, char* transB, int* M, int* N, int* K, double* alpha,
           double* A, int* LDA, double* B, int* LDB,
           double* beta, double* C, int* LDC);

typedef struct {
  double* data;
  int len;
  int stride;
  int globLen;
#ifdef HAVE_MPI
  MPI_Comm* comm;
#endif
  int comm_size;
  int comm_rank;
  int* displ;
  int* sizes;
} vector_t;

typedef vector_t* Vector;

typedef struct {
  double** data;
  Vector as_vec;
  Vector* col;
  Vector* row;
  int rows;
  int cols;
  int glob_rows;
  int glob_cols;
} matrix_t;

typedef matrix_t* Matrix;

void init_app(int argc, char** argv, int* rank, int* size);
void close_app();

int get_thread();
int num_threads();
int max_threads();

void splitVector(int globLen, int size, int** len, int** displ);
Vector createVector(int len);
#ifdef HAVE_MPI
Vector createVectorMPI(int globLen, int allocdata, MPI_Comm* comm);
#endif
Matrix subMatrix(const Matrix A, int r_ofs, int r, int c_ofs, int c);

void freeVector(Vector vec);

// allocate a n1xn2 matrix in fortran format
// note that reversed index order is assumed
Matrix createMatrix(int n1, int n2);

Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm);

void freeMatrix(Matrix A);

// perform a matrix-vector product
void MxV(Vector u, Matrix A, Vector v);

// perform a matrix-matrix product
void MxM(Matrix A, Matrix B, Matrix C, double alpha, double beta);

// perform a matrix-matrix product
void MxM2(Matrix A, Matrix B, Matrix C, int b_ofs, int b_col,
          int c_ofs, double alpha, double beta);

// perform an innerproduct
double innerproduct(Vector u, Vector);

// perform an innerproduct v2
double innerproduct2(Vector u, int ofs, int len, Vector);

// get current time in msecs
double WallTime();

// io
void saveVectorSerial(char* name, Vector data);
void saveMatrixSerial(char* name, Matrix data);

void saveVectorMPI(char* name, Vector data);
