Mat A;
MatCreate(PETSC_COMM_WORLD,&A);
MatSetType(A,MATSEQAIJ);
MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*m,m*m);

// setup matrix
for (int i=0;i<m*m;++i) // main diagonal
  MatSetValue(A,i,i,4.f,INSERT_VALUES);
for (int i=0;i<m*m-1;++i) // right coupling
  if (i%m != m-1)
    MatSetValue(A,i,i+1,-1.f,INSERT_VALUES);
for (int i=1;i<m*m;++i) // left coupling
  if (i%m)
    MatSetValue(A,i-1,i,-1.f,INSERT_VALUES);
for (int i=m;i<m*m;++i) // down coupling
  MatSetValue(A,i,i-m,-1.f,INSERT_VALUES);
for (int i=0;i<m*m-m;++i) // down coupling
  MatSetValue(A,i,i+m,-1.f,INSERT_VALUES);
