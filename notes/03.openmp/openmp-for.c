#pragma omp parallel for schedule(static)
for( int i=0;i<100;++i )
  DoSomething(i);
