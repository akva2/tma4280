#pragma once

#ifndef USE_MKL
  #define idamax idamax_
  #define dcopy  dcopy_
  #define ddot   ddot_
  #define dscal  dscal_
  #define daxpy  daxpy_
  #define dgemv  dgemv_
  #define dgemm  dgemm_
  #define dtrmm  dtrmm_
  #define dgesv  dgesv_ 
  #define dpotrf dpotrf_
  #define dpotrs dpotrs_
  #define dtrtrs dtrtrs_
#endif

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm WorldComm;
extern MPI_Comm SelfComm;
#endif

// Handy defines
#define M_PI (4.0*atan(1))

// blas prototypes
int idamax(int* N, double* x, int* incx);
void dscal(int* N, double* alpha, double* x, int* incx);
void daxpy(int* N, double* alpha, double* x, int* incx, double*y, int* incy);
void dcopy(int* N, double* x, int* incx, double*y, int* incy);
double ddot(int* N, double* dx, int* incx, double* dy, int* incy);
double dgemv(char* trans, int* M, int* N, double* alpha, double* A,
             int* LDA, double* x, int* incx,
             double* beta, double* y, int* incy);
void dgemm(char* transA, char* transB, int* M, int* N, int* K, double* alpha,
           double* A, int* LDA, double* B, int* LDB,
           double* beta, double* C, int* LDC);

// lapack prototypes
void dgesv(const int* n, const int* nrhs,
           double* A, const int* lda, int* ipiv,
           double* B, const int* ldb, int* info);
int dpotrf_(char *uplo, int *n, double *a, int* lda, int *info);
int dpotrs_(char *uplo, int *n, int *nrhs, double*a, int* lda,
                double* b, int* ldb, int* info);
int dtrtrs_(char *uplo, char* trans, char*diag, int *n, int *nrhs,
            double*a, int* lda, double* b, int* ldb, int* info);

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

// l1 ops
void axpy(const Vector v, Vector u, double alpha);
void copyVector(const Vector v, Vector u);
void copyVectorDispl(const Vector v, Vector u, int size, int displ);
void scaleVector(Vector u, double alpha);
void fillVector(Vector u, double fill);
double maxNorm(const Vector u);

// allocate a n1xn2 matrix in fortran format
// note that reversed index order is assumed
Matrix createMatrix(int n1, int n2);

#ifdef HAVE_MPI
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm);
#endif

void freeMatrix(Matrix A);

// perform a matrix-vector product - u = alpha*A*v + beta*u
void MxV(Vector u, Matrix A, Vector v, double alpha, double beta);
void MxVdispl(Vector u, Matrix A, Vector v, double alpha, double beta,
              int len, int displ);

// perform a matrix-matrix product
void MxM(Matrix A, Matrix B, Matrix C, double alpha, double beta);

// perform a matrix-matrix product
void MxM2(Matrix A, Matrix B, Matrix C, int b_ofs, int b_col,
          int c_ofs, double alpha, double beta);

void transposeMatrix(const Matrix B, Matrix A);

// perform an innerproduct
double innerproduct(Vector u, Vector);

// perform an innerproduct v2
double innerproduct2(Vector u, int ofs, int len, Vector);

// lu factor and solve
void lusolve(Matrix A, Vector u);

// cholesky factor and solve
void llsolve(Matrix A, Vector u);

// triangular solve (forward/backward substitution)
void lutsolve(Matrix A, Vector u, char uplo);

// get current time in msecs
double WallTime();

// io
void saveVectorSerial(char* name, Vector data);
void saveMatrixSerial(char* name, Matrix data);

void saveVectorMPI(char* name, Vector data);

// problem definition
typedef double(*function2)(double x, double y);

void evalMesh(Vector u, Vector gridx, Vector gridy, function2 f);
void evalMesh2(Vector u, Vector gridx, Vector gridy, function2 f, double alpha);

double poisson_source(double x, double y);
double helmholtz_source(double x, double y);
double exact_solution(double x, double y);
Matrix createPoisson2D(int M, double mu);
