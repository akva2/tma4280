#pragma once

#ifndef HAVE_MKL
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

//! \brief A struct representing a vector
typedef struct {
  double* data; //!< Array with data
  int len;      //!< The length of the vector
  int stride;   //!< The stride (distance) between vector elements
} vector_t;

typedef vector_t* Vector; //!< Convenience type definition

//! \brief A struct representing a matrix
typedef struct {
  double** data; //!< Array with data
  Vector as_vec; //!< A vector spanning the whole matrix.
  Vector* col;   //!< Column vectors pointing into the data array
  Vector* row;   //!< Row vectors pointing into the data array
  int rows;      //!< Number of rows in matrix
  int cols;      //!< Number of columns in matrix
} matrix_t;

typedef matrix_t* Matrix; //!< Convenience type defintion

//! \brief Create a vector
//! \param[in] len The length of the vector
//! \returns A vector with the requested length
Vector createVector(int len);

//! \brief Free up memory allocated to a vector
void freeVector(Vector vec);

//! \brief Allocate a n1xn2 matrix in fortran format
//! \param[in] n1 The number of rows
//! \param[in] n2 The number of columns
//! \details Matrix is allocated in Fortran format, so reversed indices must be used
Matrix createMatrix(int n1, int n2);

//! \brief Free up memory allocated to a matrix
void freeMatrix(Matrix A);

//! \brief Performs the matrix-vector product u = Av
//! \param[out] u Vector to store the result in
//! \param[in] A The matrix
//! \param[in] v The vector to apply the matrix to
void MxV(Vector u, const Matrix A, const Vector v);

//! \brief Performs the matrix-matrix product C = alpha*A*B + beta*C
void MxM(const Matrix A, const Matrix B, Matrix C, double alpha, double beta);

//! \brief Performs the matrix-matrix product C = alpha*A*B + beta*C for a subblock of B
//! \param[in] A The A matrix
//! \param[in] B The B matrix
//! \param[out] C The C matrix
//! \param[in] b_ofs The starting column of the subblock of B
//! \param[in] b_cols The number of columns in the subblock of B
void MxM2(const Matrix A, const Matrix B, Matrix C, int b_ofs, int b_col,
          double alpha, double beta);

//! \brief Performs a inner product u'*v
//! \param[in] u The u vector
//! \param[in] v The v vector
//! \returns The value of the inner product
double innerproduct(const Vector u, const Vector v);

//! \brief Get current wall-clock time
double WallTime();
