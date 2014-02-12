#ifndef POISSON_COMMON_H_
#define POISSON_COMMON_H_

#include "common.h"

//! \brief Generate the eigenvalues for a 1D Poisson operator
//! \param m The number of internal grid points
//! \details Only valid for a Poisson problem with dirichlet boundary conditions.
Vector generateEigenValuesP1D(int m);

//! \brief Generate the eigenvectors for a 1D Poisson operator
//!\ param m The number of internal grid points
//! \details Only valid for a Poisson problem with dirichlet boundary conditions.
Matrix generateEigenMatrixP1D(int m);

#ifndef HAVE_MKL
#define fst fst_
#define fstinv fstinv_
#endif

/* function prototypes */
void fst(double *v, int *n, double *w, int *nn);
void fstinv(double *v, int *n, double *w, int *nn);

#endif
