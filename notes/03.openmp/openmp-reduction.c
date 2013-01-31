Real sum=0;
#pragma omp parallel for schedule(static) reduction(+:sum)
for( int i=0;i<N;++i ) {
  sum += v[i];
}
