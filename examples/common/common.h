#ifndef COMMON_H_
#define COMMON_H_

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm WorldComm;
extern MPI_Comm SelfComm;
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

//! \brief value of pi
#define M_PI            3.14159265358979323846

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

//! \brief A structure describing a vector
typedef struct {
  double* data;   //!< The vector data
  int len;        //!< The local length of the vector
  int glob_len;   //!< The global length of the vector
  int stride;     //!< The distance in memory between vector elements
#ifdef HAVE_MPI
  MPI_Comm* comm; //!< The MPI communicator the vector is split across
#endif
  int comm_size;  //!< The size of the MPI communicator the vector is split across
  int comm_rank;  //!< The rank of this process within the communicator the vector is split across
  int* displ;     //!< Displacements for parallel vectors
  int* sizes;     //!< The size of each process for parallel vectors
} vector_t;

//! \brief Convenience typedef
typedef vector_t* Vector;

//! \brief A structure describing a matrix
typedef struct {
  double** data; //!< The matrix data
  Vector as_vec; //!< The matrix viewed as one long vector
  Vector* col;   //!< The columns of the matrix as vectors
  Vector* row;   //!< The rows of the matrix as vectors
  int rows;      //!< Local number of rows in matrix
  int cols;      //!< Local number of columns in matrix
  int glob_rows; //!< The global number of rows in matrix
  int glob_cols; //!< The global number of columns in matrix
} matrix_t;

//! \brief Convenience typedef
typedef matrix_t* Matrix;

//! \brief A function taking one double argument, returning a double argument
typedef double (*function1D)(double x);

//! \brief A function taking two double arguments, returning a double argument
typedef double (*function2D)(double x, double y);

//! \brief Initialize the application, possibly initializing MPI
//! \param argc Number of command line parameters
//! \param argv The command line parameters
//! \param[out] rank The process' rank within the MPI_COMM_WORLD is returned here
//! \param[out] size The size of the MPI_COMM_WORLD is returned here
void init_app(int argc, char** argv, int* rank, int* size);

//! \brief Close down the application, possibly deinitializing MPI
void close_app();

//! \brief Create a vector
//! \param len The length of the vector
//! \return The new vector
Vector createVector(int len);

#ifdef HAVE_MPI
//! \brief Create a parallel vector
//! \param globLen The global vector length
//! \param comm The communicator to split the vector across
//! \param allocdata If 0, no data is allocated to vector
//! \param pad Whether or not to pad vector with space at ends
//! \return The new vector
Vector createVectorMPI(int globLen, MPI_Comm* comm, int allocdata, int pad);
#endif

//! \brief Free up memory allocated to a vector
//! \param vec The vector to free
void freeVector(Vector vec);

//! \brief Calculate a parallel splitting of a vector
//! \param globLen The global length of the vector
//! \param size The number of processes to divide vector across
//! \param[out] len The length on each of the processes
//! \param[out] The displacement for each of the processes
void splitVector(int globLen, int size, int** len, int** displ);

//! \brief Create a matrix (Fortran format)
//! \param n1 The number of rows
//! \param n2 The number of columns
//! \return The new matrix
Matrix createMatrix(int n1, int n2);

#ifdef HAVE_MPI
//! \brief Create a parallel matrix (Fortran format)
//! \param n1 The local number of rows. -1 to split
//! \param n2 The local number of columns. -1 to split
//! \param N1 The global number of rows
//! \param N2 The global number of columns
//! \param comm The communicator to split the matrix across
//! \return The new matrix
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm);

//! \brief Create a parallel matrix (Fortran format) with a carthesian topology
//! \param N1 The global number of rows.
//! \param N2 The global number of columns.
//! \param comm The communicator to split the matrix across
//! \return The new matrix
Matrix createMatrixMPICart(int N1, int N2, MPI_Comm* comm, int pad);
#endif

//! \brief Free up memory allocated to a matrix
//! \param A The matrix to free
void freeMatrix(Matrix A);

//! \brief Transpose a matrix \f$A = B^T\f$
//! \param A The transposed matrix
//! \param B The matrix to transpose
void transposeMatrix(Matrix A, const Matrix B);

//! \brief Get the maximum number of available threads
//! \return Number of available threads
int getMaxThreads();

//! \brief Get current thread ID
//! \return Current thread ID
int getCurrentThread();

//! \brief Get current wall-clock time
//! \return The current wall time in seconds
double WallTime();

//! \brief Perform a dot product
//! \param u First vector
//! \param v Second vector
//! \return Value of dot product
double dotproduct(Vector u, Vector v);

//! \brief Perform a matrix-vector product \f$u = \alpha op(A) v + \beta u$\f
//! \param u Resulting vector
//! \param A The matrix
//! \param v The vector to operate on
//! \param alpha Alpha value
//! \param beta Beta value
//! \param trans 'T' for transpose, 'N' for no transpose
void MxV(Vector u, Matrix A, Vector v, double alpha, double beta, char trans);

//! \brief Perform a matrix-matrix product \f$U = \alpha op(A)op(v) + \beta u$\f
//! \param U Resulting matrix
//! \param A The matrix to operate with
//! \param V The matrix to operate on
//! \param alpha Alpha value
//! \param beta Beta value
//! \param transA 'T' for transpose of A, 'N' for no transpose
//! \param transV 'T' for transpose of V, 'N' for no transpose
void MxM(Matrix U, Matrix A, Matrix V, double alpha, double beta,
         char transA, char transV);

//! \brief Fill a diagonal of a matrix with a constant value
//! \param A The matrix to fill the diagonal for
//! \param diag The offset from the main diagonal
//! \param value The value to fill with
void diag(Matrix A, int diag, double value);

//! \brief Create an equidistant mesh
//! \param x0 The starting point
//! \param x1 The ending point
//! \param N The number of grid points - 1
Vector equidistantMesh(double x0, double x1, int N);

//! \brief Evaluate a function on the internal points of a mesh
//! \param u The resulting values
//! \param grid The grid
//! \param func The function to evaluate
void evalMeshInternal(Vector u, Vector grid, function1D func);

//! \brief Evaluate a function on the internal points of a 2D mesh
//! \param u The resulting values
//! \param grid The grid
//! \param func The function to evaluate
//! \param border If 1, space is reserved for boundary in the matrix
void evalMeshInternal2(Matrix u, Vector grid, function2D func, int boundary);

//! \brief Evaluate a function with a displacement
//! \param u The resulting values
//! \param grid The grid
//! \param func The function to evaluate
//! \param displ Displacement in grid
void evalMeshDispl(Vector u, Vector grid, function1D func);

//! \brief Scale a vector
//! \param alpha The scaling factor
void scaleVector(Vector u, double alpha);

//! \brief y = alpha*x + y
//! \param y y vector
//! \param x x vector
//! \param alpha alpha value
void axpy(Vector y, const Vector x, double alpha);

//! \brief Solve a linear system using gaussian elimination (LU)
//! \param A The matrix to solve for
//! \param x Right hand side on entry, solution on return
//! \param ipiv An array pointing to pivot numbers. If non-null on entry the matrix is
//!             assumed prefactored.
void lusolve(Matrix A, Vector x, int** ipiv);

//! \brief Solve a linear system using cholesky factorization (LL^T)
//! \param A The matrix to solve for
//! \param x Right hand side on entry, solution on return
//! \param prefactored If 1, A is prefactored
void llsolve(Matrix A, Vector x, int prefactored);

//! \brief Apply backward/forward substitution to a system of equations \f$x = A^{-1}b\f$ or \f$x = A^{-T}b\f$
//! \param[in] A The A matrix data
//! \param x The right hand side matrix data on entry, the solution on return
//! \param[in] uplo If 'U' A is upper triangular, if 'L' A is lower triangular
void lutsolve(const Matrix A, Vector x, char uplo);

//! \brief Find the inf norm of a vector: \f$\|u\|_\infty\f$
//! \param u The u vector
//! \return The inf-norm of the vector
double maxNorm(const Vector u);

//! \brief Copy a vector: \f$y = x\f$
//! \param y The y vector
//! \param[in] x The x vector
void copyVector(Vector y, const Vector x);

//! \brief Fill a vector with a constant: \f$u(i) = \alpha\,\forall\,i\f$
//! \param u The u vector
//! \param[in] alpha The fill constant
void fillVector(Vector u, double alpha);

//! \brief Extract a subblock from a matrix
//! \param[in] A The full matrix
//! \param[in] r_ofs The starting row of the subblock
//! \param[in] r Number of rows in subblock
//! \param[in] c_ofs The starting column of the subblock
//! \param[in] c Number of columns in subblock
//! \return The subblock
Matrix subMatrix(const Matrix A, int r_ofs, int r, int c_ofs, int c);

//! \brief Save a matrix to an .asc file
//! \param A The matrix to save
//! \param file Filename to save to
void saveMatrix(const Matrix A, char* file);

//! \brief Collect a parallel vector for operator evaluation
//! \param u The vector to collect
void collectVector(Vector u);

//! \brief Print a vector to the terminal for inspection
//! \param u The vector to print
void printVector(const Vector u);

//! \brief Clone a vector
//! \param u Vector to clone
//! \returns New clone of vector
Vector cloneVector(const Vector u);

//! \brief Clone a matrix
//! \param u Matrix to clone
//! \returns New clone of matrix 
Matrix cloneMatrix(const Matrix u);

#endif
