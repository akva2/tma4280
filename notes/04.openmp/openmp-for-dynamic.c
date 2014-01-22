#pragma omp parallel for schedule(dynamic)
for( int i=0;i<100;++i )
  DoSomething(i);
