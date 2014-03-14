PetscInt* d_Nz = (PetscInt*)malloc(m*m*sizeof(PetscInt));
int total=0;
for (int i=0;i<m*m;++i) {
  (count actual values per row)
}
MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,d_Nz);
PetscInt* col = (PetscInt*)malloc(total*sizeof(PetscInt));
int c=0;
for (int i=0;i<m*m;++i) {
  (declare the actual column indices)
}
MatSeqAIJSetColumnIndices(A,col);
