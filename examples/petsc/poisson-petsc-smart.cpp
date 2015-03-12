#include <stdio.h>
#include <sys/time.h>

#include "petscksp.h"

double WallTime (int size)
{
  if (size == 1) {
    struct timeval tmpTime;
    gettimeofday(&tmpTime,NULL);
    return tmpTime.tv_sec + tmpTime.tv_usec/1.0e6;
  } else
    return MPI_Wtime();
}

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
  int rank,size;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  double start = WallTime(size);

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
  MatSetUp(A);

  // and setup the sparsity pattern
  PetscInt first, last;
  MatGetOwnershipRange(A,&first,&last);
  PetscInt* d_Nz = (PetscInt*)malloc((last-first)*sizeof(PetscInt));
  PetscInt* o_Nz = (PetscInt*)malloc((last-first)*sizeof(PetscInt));
  for (int i=first;i<last;++i) {
    d_Nz[i-first] = 5;
    o_Nz[i-first] = 5;
  }
  MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,d_Nz,PETSC_DEFAULT,o_Nz);

  // Create linear solver object
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD,&ksp);

  // setup rhs
  PetscInt low, high;
  VecGetOwnershipRange(b,&low,&high);
  PetscInt* inds = (PetscInt*)malloc((high-low)*sizeof(PetscInt));
  double* vals = (double*)malloc((high-low)*sizeof(double));
  double h    = 1./(double)n;
  for (int i=low;i<high;++i) {
    inds[i-low] = i;
    vals[i-low] = h*h;
  }
  VecSetValues(b,high-low,inds,vals,INSERT_VALUES);
  free(inds);
  free(vals);

  // setup matrix
  for (int i=low;i<high;++i) // main diagonal
    MatSetValue(A,i,i,4.f,INSERT_VALUES);
  for (int i=low;i<high;++i) // right coupling
    if (i%m != m-1)
      MatSetValue(A,i,i+1,-1.f,INSERT_VALUES);
  for (int i=(low==0?1:low);i<high;++i) // left coupling
    if (i%m)
      MatSetValue(A,i,i-1,-1.f,INSERT_VALUES);
  for (int i=(low==0?m:low);i<high;++i) // down coupling
    MatSetValue(A,i,i-m,-1.f,INSERT_VALUES);
  for (int i=low;i<(high==m*m?high-m:high);++i) // down coupling
    MatSetValue(A,i,i+m,-1.f,INSERT_VALUES);

  // sync processes
  VecAssemblyBegin(b);
  VecAssemblyEnd(b);
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

//  PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_DENSE);
//  MatView(A,PETSC_VIEWER_STDOUT_WORLD);
//  VecView(b,PETSC_VIEWER_STDOUT_WORLD);
  if (rank == 0)
    printf("assembly: %f\n",WallTime(size)-start);

  // solve
  KSPSetType(ksp,"cg");
  KSPSetTolerances(ksp,1.e-8,1.e-10,1.e6,10000);
  KSPSetOperators(ksp,A,A);
  PC pc;
  KSPGetPC(ksp,&pc);
  PCSetType(pc,"asm");
  PCASMSetType(pc,PC_ASM_BASIC);
  PCSetFromOptions(pc);
  start = WallTime(size);

  PCSetUp(pc);

  // setup solver
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  Vec x;
  VecDuplicate(b,&x);
  KSPSolve(ksp,b,x);

  double val;
  VecNorm(x,NORM_INFINITY,&val);

  if (rank == 0) {
    PetscInt its;
    KSPGetIterationNumber(ksp,&its);
    printf (" umax = %e, its %i \n",val,(int)its);
    printf ("solve %f\n",WallTime(size)-start);
  }

  PetscFinalize();
  return 0;

}
