#pragma once

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm WorldComm; //!< our duplicate of MPI_COMM_WORLD
extern MPI_Comm SelfComm;  //!< our duplicate of MPI_COMM_SELF
#endif

#define M_PI (4.0*atan(1))

/** @brief A (parallel) vector
  * @details This structure holds information about a vector. The vector may 
  *          be split across a MPI communicator.
  */
typedef struct {
  double* data; //!< The vector data
  int len;      //!< Local vector length
  int stride;   //!< Distance in memory between vector elements
  int globLen;  //!< Global vector length
#ifdef HAVE_MPI
  MPI_Comm* comm; //!< Communicator vector is split across
#endif
  int comm_size;  //!< The number of processes in communicator
  int comm_rank;  //!< Rank in communicator
  int* displ;     //!< Vector displacements
  int* sizes;     //!< Vector sizes
} vector_t;

typedef vector_t* Vector; //!< Convenience typedef

/** @brief A (parallel) matrix
  * @details This structure holds information about a matrix. The matrix may 
  *         be split across a MPI communicator (currently either in rows
  *         or columns).
  */
typedef struct {
  double** data;  //!< The local matrix data
  Vector as_vec;  //!< The matrix as a vector
  Vector* col;    //!< The columns of the matrix
  Vector* row;    //!< The rows of the matrix
  int rows;       //!< Number of local rows
  int cols;       //!< Number of local columns
  int glob_rows;  //!< Number of global rows
  int glob_cols;  //!< Number of global columns
} matrix_t;

typedef matrix_t* Matrix; //!< Convenience typedef

/** @brief Initialize the application, setting up MPI if available
  * @param[in] argc The number of command line parameters
  * @param[in] argv The command line parameters
  * @param[out] rank Rank in WORLD communicator
  * @param[out] size Size of WORLD communicator
  */
void init_app(int argc, char** argv, int* rank, int* size);

/** @brief Close the application, shutting down MPI if available
  */
void close_app();

/** @brief Get the current thread id
  * @return Current thread id
  * @details Will always return 0 if called outside a parallel section
  */
int get_thread();

/** @brief Get the number of threads currently running
  * @return Number of threads currently running
  * @details Will always return 1 if called outside a parallel section
  */
int num_threads();

/** @brief Get the number of available threads
  * @return Number of available threads
  */
int max_threads();

/** @brief Split a vector in chunks
  * @param[in] globLen Length of vector
  * @param[in] size Number of chunks
  * @param[out] len The size of each chunk
  * @param[out] displ The displacement offset for each chunk
  */
void splitVector(int globLen, int size, int** len, int** displ);

/** @brief Allocate a serial vector
  * @param[in] len Length of vector
  * @return The new vector
  */
Vector createVector(int len);

#ifdef HAVE_MPI
/** @brief Allocate a vector split across a communicator
  * @param[in] globLen Length of global vector
  * @param[in] allocdata If 0, do not allocate data.
  * @param[in] comm Communicator to split vector across
  * @return The new vector
  */
Vector createVectorMPI(int globLen, int allocdata, MPI_Comm* comm);
#endif

/** @brief Extract a subblock from a matrix
  * @param[in] A The full matrix
  * @param[in] r_ofs The starting row of the subblock
  * @param[in] r Number of rows in subblock
  * @param[in] c_ofs The starting column of the subblock
  * @param[in] c Number of columns in subblock
  * @return The subblock
  */
Matrix subMatrix(const Matrix A, int r_ofs, int r, int c_ofs, int c);

/** @brief Free a vector
  * @param vec The vector to free
  */
void freeVector(Vector vec);

/** @brief Perform an axpy operation: \f$y = \alpha x + y\f$
  * @param y The y vector
  * @param[in] x The x vector
  * @param[in] alpha The scale factor
  */
void axpy(Vector y, const Vector x, double alpha);

/** @brief Copy a vector: \f$y = x\f$
  * @param y The y vector
  * @param[in] x The x vector
  */
void copyVector(Vector y, const Vector x);

/** @brief Copy parts of a vector: \f$y = x(part)\f$
  * @param[out] y The y vector
  * @param[in] x The x vector
  * @param[in] ysize The number of elements to copy
  * @param[in] xdispl The displacement in x
  * @details The displacement only applies to x
  */
void copyVectorDispl(Vector y, const Vector x, int ysize, int xdispl);

/** @brief Scale a vector: \f$u = \alpha u\f$
  * @param u The u vector
  * @param[in] alpha The scale factor
  */
void scaleVector(Vector u, double alpha);

/** @brief Fill a vector with a constant: \f$u(i) = \alpha\,\forall\,i\f$
  * @param u The u vector
  * @param[in] alpha The fill constant
  */
void fillVector(Vector u, double alpha);

/** @brief Find the inf norm of a vector: \f$\|u\|_\infty\f$
  * @param u The u vector
  * @return The inf-norm of the vector
  */
double maxNorm(const Vector u);

/** @brief Allocate a serial matrix in column-major format
  * @param[in] n1 Number of rows in matrix
  * @param[in] n2 Number of columns in matrix
  * @return The new matrix
  */
Matrix createMatrix(int n1, int n2);

/** @brief Clone a matrix
  * @param[in] A Matrix to clone
  * @return The new matrix
  * @details Note that only a subset of the parallel information is copied
  */
Matrix cloneMatrix(const Matrix A);

#ifdef HAVE_MPI
/** @brief Allocate a matrix split across a communicator
  * @param[in] n1 Number of local rows, -1 if they are to be split
  * @param[in] n2 Number of local columns, -1 if they are to be split
  * @param[in] N1 Number of global rows
  * @param[in] N2 Number of global columns
  * @param[in] comm Communicator to split vector across
  * @return The new matrix
  */
Matrix createMatrixMPI(int n1, int n2, int N1, int N2, MPI_Comm* comm);
#endif

/** @brief Free a matrix
  * @param A The matrix to free
  */
void freeMatrix(Matrix A);

/** @brief A matrix vector product: \f$y = \alpha Ax +\beta y\f$
  * @param y The y vector
  * @param[in] A The A matrix
  * @param[in] x The x vector
  * @param[in] alpha The first scale factor
  * @param[in] beta The second scale factor
  */
void MxV(Vector y, const Matrix A, const Vector x, double alpha, double beta);

/** @brief A matrix vector product with vector displacement: \f$y = \alpha Ax +\beta y\f$
  * @param y The y vector
  * @param[in] A The A matrix
  * @param[in] x The x vector
  * @param[in] alpha The first scale factor
  * @param[in] beta The second scale factor
  * @param[in] ydispl The displacement offset for the y vector
  */
void MxVdispl(Vector y, const Matrix A, const Vector x,
              double alpha, double beta, int ydispl);

/** @brief A matrix vector product: \f$C = \alpha AB +\beta C\f$
  * @param C The C matrix
  * @param[in] A The A matrix
  * @param[in] B The B matrix
  * @param[in] alpha The first scale factor
  * @param[in] beta The second scale factor
  * @param[in] transA Transpose value for A
  * @param[in] transB Transpose value for B
  */
void MxM(Matrix C, const Matrix A, const Matrix B, double alpha, double beta,
         char transA, char transB);

/** @brief A matrix vector product with displacements: \f$C = \alpha AB +\beta C\f$
  * @param C The C matrix
  * @param[in] A The A matrix
  * @param[in] B The B matrix
  * @param[in] b_ofs The column displacement of B matrix
  * @param[in] b_col Number of columns in B matrix
  * @param[in] c_ofs The column displacement of C matrix
  * @param[in] alpha The first scale factor
  * @param[in] beta The second scale factor
  */
void MxM2(Matrix C, const Matrix A, const Matrix B, int b_ofs, int b_col,
          int c_ofs, double alpha, double beta);

/** @brief Transpose a matrix: \f$A = B^T\f$
  * @param[out] A The A matrix
  * @param[in] B The B matrix
  */
void transposeMatrix(Matrix A, const Matrix B);

/** @brief Dot product of two vectors: \f$y^T x\f$
  * @param[in] x The x vector
  * @param[in] y The y vector
  * @return The value of the dot product
  */
double innerproduct(const Vector x, const Vector y);

/** @brief Dot product of two vectors with displacement: \f$y^T x\f$
  * @param[in] x The x vector
  * @param[in] y The y vector
  * @param[in] xdispl The x vector displacement
  * @param[in] len The length of the dot product
  * @return The value of the dot product
  */
double innerproduct2(const Vector x, const Vector y, int xdispl, int len);

/** @brief A LU solve for a general matrix \f$x = A^{-1}b\f$
  * @param A The A matrix on entry, the factored matrix on return
  * @param x The right hand side matrix data on entry, the solution on return
  * @param ipiv Array with the pivot numbers
  * @details ipiv is expected to point to a NULL pointer on first call.
  *          Caller manages the cleanup.
  */
void lusolve(Matrix A, Vector x, int** ipiv);

/** @brief A cholesky solve for a SPD matrix \f$x = A^{-1}b\f$
  * @param A The A matrix on entry, the factored matrix on return
  * @param x The right hand side matrix data on entry, the solution on return
  */
void llsolve(Matrix A, Vector x);

/** @brief Solve using a backward/forward substitution \f$x = A^{-1}b\f$ or \f$x = A^{-T}b\f$
  * @param[in] A The A matrix data
  * @param x The right hand side matrix data on entry, the solution on return
  * @param[in] uplo If 'U' A is upper triangular, if 'L' A is lower triangular
  */
void lutsolve(const Matrix A, Vector x, char uplo);

/** @brief Get current wall clock time in seconds
  * @return Current wall clock time
  */
double WallTime();

/** @brief Save a vector to an asc file
  * @param[in] name The file name
  * @param[in] x Vector to save
  */
void saveVectorSerial(char* name, const Vector x);

/** @brief Save a matrix to an asc file
  * @param[in] name The file name
  * @param[in] x Matrix to save
  */
void saveMatrixSerial(char* name, const Matrix x);

/** @brief Save a parallel vector to an asc file
  * @param[in] name The file name
  * @param[in] x Vector to save
  */
void saveVectorMPI(char* name, const Vector x);

/** @brief A function pointer to a function taking two doubles as parameters
  */
typedef double(*function2)(double x, double y);

/** @brief Evaluate a function over a mesh \f$u_{jN+i} = f(x_i, y_j)\f$
  * @param[out] u The vector to store values in
  * @param[in] x The x grid
  * @param[in] y The y grid
  * @param[in] f The function to evaluate
  */
void evalMesh(Vector u, const Vector x, const Vector y, function2 f);

/** @brief Evaluate a function over a mesh with displacement\f$u_{ij} = f(x_{i+\text{displ}}, y_{j+\text{displ}})\f$
  * @param[out] u The matrix to store values in
  * @param[in] x The x grid
  * @param[in] y The y grid
  * @param[in] f The function to evaluate
  * @param[in] xdispl The x displacement
  * @param[in] ydispl The y displacement
  */
void evalMeshDispl(Matrix u, const Vector x, const Vector y, function2 f,
                   int xdispl, int ydispl);

/** @brief Evaluate a function over a mesh, scale and add \f$u_{jN+i} = u_{jN+i}+\alpha f(x_i, y_j)\f$
  * @param[out] u The vector to store values in
  * @param[in] x The x grid
  * @param[in] y The y grid
  * @param[in] f The function to evaluate
  * @param[in] alpha The scale factor
  */
void evalMesh2(Vector u, const Vector x, const Vector y, 
               function2 f, double alpha);

/** @brief Evaluate a function over a mesh, scale and add \f$u_{ij} = u_{ij}+\alpha f(x_{i+\text{displ}}, y_{j+\text{displ}})\f$
  * @param[out] u The matrix to store values in
  * @param[in] x The x grid
  * @param[in] y The y grid
  * @param[in] f The function to evaluate
  * @param[in] alpha The scale factor
  * @param[in] xdispl The x displacement
  * @param[in] ydispl The y displacement
  */
void evalMesh2Displ(Matrix u, const Vector x, const Vector y, 
                    function2 f, double alpha, int xdispl, int ydispl);

/** @brief Poisson source function for \f$f(x,y) = x(x-1)\sin(2\pi y)\f$
  * @param[in] x The x coordinate
  * @param[in] y The y coordinate
  * @return The function value
  */
double poisson_source(double x, double y);

/** @brief Evaluates\f$f(x,y) = x(x-1)\sin(2\pi y)\f$
  * @param[in] x The x coordinate
  * @param[in] y The y coordinate
  * @return The function value
  */
double exact_solution(double x, double y);

/** @brief Create a 2D Helmholtz matrix \f$A + \muI$
  * @param[in] M Number of DOFs in each spatial direction
  * @param[in] mu The mass term scale factor
  * @return The matrix operator
  */
Matrix createPoisson2D(int M, double mu);

void collectMatrix(Matrix u);
