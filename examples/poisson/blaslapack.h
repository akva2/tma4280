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

// blas prototypes

/** @brief Find the element with the largest magnitude: \f$\max_{i} |x_i|\f$
*   @param[in] N The length of the vector
*   @param[in] x The vector data
*   @param[in] incx The distance in memory between vector elements
*   @return The index of the element
*   @details Since this is a fortran function, the index returned is 1-based
*/
int idamax(int* N, double* x, int* incx);

/** @brief Scale a vector: \f$x = \alpha x\f$
  * @param[in] N The vector length
  * @param[in] alpha The scale factor
  * @param[out] x The vector data
  * @param[in] incx The distance in memory between vector elements
  */
void dscal(int* N, double* alpha, double* x, int* incx);

/** @brief Perform an axpy operation: \f$y = \alpha x + y\f$
  * @param[in] N The vector length
  * @param[in] alpha The scale factor
  * @param[in] incx The distance in memory between vector elements in x
  * @param[in] x The x vector data
  * @param[in] incx The distance in memory between x vector elements
  * @param[out] y The y vector data
  * @param[in] incy The distance in memory between y vector elements
  */
void daxpy(int* N, double* alpha, double* x, int* incx, double*y, int* incy);

/** @brief Copy a vector: \f$y = x\f$
  * @param[in] N The vector length
  * @param[in] incx The distance in memory between vector elements in x
  * @param[in] x The x vector data
  * @param[in] incx The distance in memory between x vector elements
  * @param[out] y The y vector data
  * @param[in] incy The distance in memory between y vector elements
  */
void dcopy(int* N, double* x, int* incx, double*y, int* incy);

/** @brief Dot product of two vectors: \f$y^T x\f$
  * @param[in] N The vector length
  * @param[in] x The x vector data
  * @param[in] incx The distance in memory between vector elements in x
  * @param[in] y The y vector data
  * @param[in] incy The distance in memory between y vector elements
  * @return The value of the dot product
  */
double ddot(int* N, double* x, int* incx, double* y, int* incy);

/** @brief A matrix vector product: \f$y = \alpha Ax +\beta y\f$ or \f$y = \alpha A^Tx +\beta y\f$
  * @param[in] trans 'N' for no transpose or 'T' for transpose
  * @param[in] M Number of rows of A
  * @param[in] N Number of columns of A
  * @param[in] alpha The first scale factor
  * @param[in] A The A matrix data
  * @param[in] LDA Leading dimension of A
  * @param[in] x The x vector data
  * @param[in] incx The distance in memory between vector elements in x
  * @param[in] beta The second scale factor
  * @param y The y vector data
  * @param[in] incy The distance in memory between y vector elements
  */
void dgemv(char* trans, int* M, int* N, double* alpha, double* A,
           int* LDA, double* x, int* incx,
           double* beta, double* y, int* incy);

/** @brief A matrix vector product: \f$C = \alpha AB +\beta C\f$, \f$C = \alpha A^TB +\beta C\f$ or \f$C = \alpha A^TB^T +\beta C\f$
  * @param[in] transA 'N' for no transpose of A or 'T' for transpose of A
  * @param[in] transB 'N' for no transpose of B or 'T' for transpose of B
  * @param[in] M Number of rows of op(A) (and C)
  * @param[in] N Number of columns of op(B) (and C)
  * @param[in] K Number of columns of op(A) (and rows of op(B))
  * @param[in] alpha The first scale factor
  * @param[in] A The A matrix data
  * @param[in] LDA Leading dimension of A
  * @param[in] B The B matrix data
  * @param[in] LDB Leading dimension of B
  * @param[in] beta The second scale factor
  * @param[out] C The C matrix data
  * @param[in] LDC Leading dimension of C
  */
void dgemm(char* transA, char* transB, int* M, int* N, int* K, double* alpha,
           double* A, int* LDA, double* B, int* LDB,
           double* beta, double* C, int* LDC);

// lapack prototypes

/** @brief A LU solve for a general matrix \f$x = A^{-1}b\f$
  * @param[in] n Matrix dimension
  * @param[in] nrhs Number of right hand sides
  * @param[in] A The A matrix data
  * @param[in] lda Leading dimension of A
  * @param[in] ipiv The pivot numbers
  * @param B The right hand side matrix data on entry, the solution on return
  * @param[in] ldb Leading dimension of B
  * @param[out] info The return code
  * @details ipiv must be preallocated with n entries
  */
void dgesv(const int* n, const int* nrhs,
           double* A, const int* lda, int* ipiv,
           double* B, const int* ldb, int* info);

/** @brief Cholesky factorize a SPD matrix \f$A=LL^T\f$
  * @param[in] uplo If 'U' the upper triangular  part is stored, if 'L' the lower triangular part
  * @param[in] n Matrix dimension
  * @param[in] A The A matrix data
  * @param[in] lda Leading dimension of A
  * @param[out] info The return code
  */
int dpotrf_(char *uplo, int *n, double *A, int* lda, int *info);

/** @brief Solve using a Cholesky factorized matrix \f$z = L^{-T}b, x = L^{-1}z\f$
  * @param[in] uplo If 'U' the upper triangular  part is stored, if 'L' the lower triangular part
  * @param[in] n Matrix dimension
  * @param[in] nrhs Number of right hand sides
  * @param[in] A The A matrix data
  * @param[in] lda Leading dimension of A
  * @param B The right hand side matrix data on entry, the solution on return
  * @param[in] ldb Leading dimension of B
  * @param[out] info The return code
  */
void dpotrs_(char *uplo, int *n, int *nrhs, double* A, int* lda,
             double* B, int* ldb, int* info);

/** @brief Solve using a backward/forward substitution \f$x = A^{-1}b\f$ or \f$x = A^{-T}b\f$
  * @param[in] uplo If 'U' A is upper triangular, if 'L' A is lower triangular
  * @param[in] trans 'N' for no transpose or 'T' for transpose
  * @param[in] diag 'N' for non-unit triangular or 'U' for unit triangular
  * @param[in] n Matrix dimension
  * @param[in] nrhs Number of right hand sides
  * @param[in] A The A matrix data
  * @param[in] lda Leading dimension of A
  * @param B The right hand side matrix data on entry, the solution on return
  * @param[in] ldb Leading dimension of B
  * @param[out] info The return code
  */
int dtrtrs_(char *uplo, char* trans, char* diag, int *n, int *nrhs,
            double* A, int* lda, double* B, int* ldb, int* info);
