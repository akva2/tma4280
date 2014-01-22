#pragma omp parallel for schedule(dynamic,5)
for( int i=0;i<100;++i )
  DoSomething(i);
