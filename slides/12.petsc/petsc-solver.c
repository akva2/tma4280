KSP ksp;
KSPCreate(PETSC_COMM_WORLD,&ksp);
KSPSetType(ksp,"cg");
KSPSetTolerances(ksp,1.e-10,1.e-10,1.e6,10000);
KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);
PC pc;
KSPGetPC(ksp,&pc);
PCSetType(pc,"ilu");
PCSetFromOptions(pc);
PCSetUp(pc);
// setup solver
KSPSetFromOptions(ksp);
KSPSetUp(ksp);
