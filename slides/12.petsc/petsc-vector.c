Vec b;
VecCreate(PETSC_COMM_WORLD,&b);
VecSetSizes(b,m*m,PETSC_DECIDE);
VecSetFromOptions(b);

// setup rhs
double h    = 1./(double)n;
for (int j=0; j < m*m; j++)
  VecSetValue(b,j,h*h,INSERT_VALUES);
