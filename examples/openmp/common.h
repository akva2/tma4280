#pragma once

#ifndef USE_MKL
  #define ddot ddot_
  #define dgemv dgemv_
  #define dgemm dgemm_
#endif

// blas prototypes
double ddot(int* N, double* dx, int* incx, double* dy, int* incy);
double dgemv(char* trans, int* M, int* N, double* alpha, double* A,
             int* LDA, double* x, int* incx,
             double* beta, double* y, int* incy);
void dgemm(char* transA, char* transB, int* M, int* N, int* K, double* alpha,
           double* A, int* LDA, double* B, int* LDB,
           double* beta, double* C, int* LDC);

// allocate a n1xn2 matrix in fortran format
// note that reversed index order is assumed
double** createMatrix(int n1, int n2);

// perform a matrix-vector product
void MxV(double* u, double** A, double* v, int N);

// perform a matrix-matrix product
void MxM2(double** A, double** B, double** C, int M, int N, int K,
         double alpha, double beta);

// perform a matrix-matrix product
void MxM(double* A, double* B, double* C, int M, int N, int K,
         double alpha, double beta);

// perform an innerproduct
double innerproduct(double* u, double* v, int N);

// get current time in msecs
double WallTime();
