// and setup the sparsity pattern
PetscInt first, last;
MatGetOwnershipRange(A,&first,&last);
PetscInt* d_Nz = 
  (PetscInt*)malloc((last-first)*sizeof(PetscInt));
PetscInt* o_Nz = 
  (PetscInt*)malloc((last-first)*sizeof(PetscInt));
for (int i=first;i<last;++i) {
  d_Nz[i-first] = 5;
  o_Nz[i-first] = 5;
}
MatMPIAIJSetPreallocation(A,PETSC_DEFAULT,
                          d_Nz,PETSC_DEFAULT,o_Nz);
