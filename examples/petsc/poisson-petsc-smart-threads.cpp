#include <stdio.h>

#include "petscksp.h"

#include <omp.h>

int main(int argc, char** argv)
{
  /* the total number of grid points in each spatial direction is (n+1) */
  /* the total number of degrees-of-freedom in each spatial direction is (n-1) */
  /* this version requires n to be a power of 2 */
  if( argc < 2 ) {
    printf("need a problem size\n");
    return 1;
  }

  int n  = atoi(argv[1]);
  int m  = n-1;

  // Initialize Petsc
  PetscInitialize(&argc,&argv,0,PETSC_NULL);

  double start = omp_get_wtime();

  // Create our vector
  Vec b;
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,m*m);
  VecSetFromOptions(b);

  // Create our matrix
  Mat A;
  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetType(A,MATAIJ);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*m,m*m);

  // and setup the sparsity pattern
  PetscInt* d_Nz = (PetscInt*)malloc(m*m*sizeof(PetscInt));
  int total=0;
  for (int i=0;i<m*m;++i) {
    int cols=1;
    if (i-m >= 0)
      cols++;
    if (i-1 >= 0)
      cols++;
    if (i+1 < m*m)
      cols++;
    if (i+m < m*m)
      cols++;
    d_Nz[i] = cols;
    total += cols;
  }
  MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,d_Nz);
  PetscInt* col = (PetscInt*)malloc(total*sizeof(PetscInt));
  int c=0;
  for (int i=0;i<m*m;++i) {
    if (i-m >= 0)
      col[c++] = i-m;
    if (i-1 >= 0)
      col[c++] = i-1;
    col[c++] = i;
    if (i+1 < m*m)
      col[c++] = i+1;
    if (i+m < m*m)
      col[c++] = i+m;
  }
  MatSeqAIJSetColumnIndices(A,col);

  // set sparsity pattern in stone
  MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
  MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  MatSetUp(A);

  // Create linear solver object
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  // setup rhs
  PetscInt* inds = (PetscInt*)malloc(m*m*sizeof(PetscInt));
  double* vals = (double*)malloc(m*m*sizeof(double));
  double h    = 1./(double)n;
  for (int i=0;i<m*m;++i) {
    inds[i] = i;
    vals[i] = h*h;
  }
  VecSetValues(b,m*m,inds,vals,INSERT_VALUES);
  free(inds);
  free(vals);

  // setup matrix
#pragma omp parallel for schedule(static)
  for (int i=0;i<m*m;++i) {
    MatSetValue(A,i,i,4.f,INSERT_VALUES);  // main diagonal
    if (i%m != m-1)
      MatSetValue(A,i,i+1,-1.f,INSERT_VALUES);
    if (i%m && i-1 >= 0)
      MatSetValue(A,i,i-1,-1.f,INSERT_VALUES);
    if (i >= m)
      MatSetValue(A,i,i-m,-1.f,INSERT_VALUES);
    if (i < m*(m-1))
      MatSetValue(A,i,i+m,-1.f,INSERT_VALUES);
  }

  // sync processes
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

//  PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE);
//  MatView(A,PETSC_VIEWER_STDOUT_WORLD);
//  VecView(b,PETSC_VIEWER_STDOUT_WORLD);

  printf("assembly: %f\n",omp_get_wtime()-start);

  // solve
  KSPSetType(ksp,"cg");
  KSPSetFromOptions(ksp);
  KSPSetTolerances(ksp,1.e-8,1.e-10,1.e6,10000);
  KSPSetOperators(ksp,A,A);
  PC pc;
  KSPGetPC(ksp,&pc);
//  PCASMSetType(pc,PC_ASM_BASIC);
  PCSetFromOptions(pc);
  PCSetUp(pc);

  // setup solver
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  Vec x;
  VecDuplicate(b,&x);
  KSPSolve(ksp,b,x);

  double val;
  VecNorm(x,NORM_INFINITY,&val);

  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  if (rank == 0) {
    PetscInt its;
    KSPGetIterationNumber(ksp,&its);
    printf (" umax = %e, its %i \n",val,(int)its);
  }

  PetscFinalize();
  return 0;

}
