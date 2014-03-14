PetscInt low, high;
VecGetOwnershipRange(b,&low,&high);
PetscInt* inds = (PetscInt*)malloc((high-low)*sizeof(PetscInt));
double* vals = (double*)malloc((high-low)*sizeof(double));
double h    = 1./(double)n;
for (int i=0;i<high-1;++i) {
  inds[i] = low+i;
  vals[i] = h*h;
}
VecSetValues(b,high-low,inds,vals,INSERT_VALUES);
free(inds);
free(vals);
