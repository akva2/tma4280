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
#endif
