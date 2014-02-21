#ifndef SOLVERS_H_
#define SOLVERS_H_

#include "common.h"

//! \brief Approximately solve a linear system using Gauss-Jacobi iterations
//! \param A The matrix to solve for
//! \param u The right hand side on entry, the solution on exit
//! \param tol The tolerance used in the iterations (infinity norm of increment)
//! \param maxit Maximum number of iterations to perform
int GaussJacobi(Matrix A, Vector u, double tol, int maxit);

//! \brief Approximately solve a linear system using Gauss-Jacobi iterations utilizing BLAS
//! \param A The matrix to solve for
//! \param u The right hand side on entry, the solution on exit
//! \param tol The tolerance used in the iterations (infinity norm of increment)
//! \param maxit Maximum number of iterations to perform
int GaussJacobiBlas(Matrix A, Vector u, double tol, int maxit);

//! \brief Approximately solve a linear system using Gauss-Seidel iterations
//! \param A The matrix to solve for
//! \param u The right hand side on entry, the solution on exit
//! \param tol The tolerance used in the iterations (infinity norm of increment)
//! \param maxit Maximum number of iterations to perform
int GaussSeidel(Matrix A, Vector u, double tol, int maxit);

//! \brief Solve a linear system using the conjugate gradient method
//! \param A The matrix
//! \param b Right hand side vector on entry, solution on exit
//! \param tolerance Residual tolerance
int cg(Matrix A, Vector b, double tolerance);

//! \brief Function pointer for performing u = A*v
//! \param u The resulting vector
//! \param v The vector to apply the function to
typedef void (*MatVecFunc)(Vector u, Vector v);

//! \brief Function pointer for performing u = A*v with u and v stored as matrices
//! \param u The resulting vector
//! \param v The vector to apply the function to
typedef void (*MatMatFunc)(Matrix u, Matrix v);

//! \brief Solve a linear system using the matrix-free conjugate gradient method
//! \param A The function evaluating the matrix-vector product
//! \param b Right hand side vector on entry, solution on exit
//! \param tolerance Residual tolerance
int cgMatrixFree(MatVecFunc A, Vector b, double tolerance);

//! \brief Solve a linear system using the matrix-free conjugate gradient method
//! \param A The function evaluating the matrix-vector product
//! \param b Right hand side vector on entry, solution on exit
//! \param tolerance Residual tolerance
int cgMatrixFreeMat(MatMatFunc A, Matrix b, double tolerance);

//! \brief Solve a linear system using the matrix-free preconditioned conjugate gradient method
//! \param A The function evaluating the matrix-vector product
//! \param pre The function evaluating the preconditioner
//! \param b Right hand side vector on entry, solution on exit
//! \param tolerance Residual tolerance
int pcgMatrixFree(MatVecFunc A, MatVecFunc pre, Vector b, double tolerance);

//! \brief Solve a linear system using the matrix-free preconditioned conjugate gradient method
//! \param A The function evaluating the matrix-vector product
//! \param pre The function evaluating the preconditioner
//! \param b Right hand side vector on entry, solution on exit
//! \param tolerance Residual tolerance
int pcgMatrixFreeMat(MatMatFunc A, MatMatFunc pre, Matrix b, double tolerance);
#endif
