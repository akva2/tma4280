Real prod=1;
#pragma omp parallel for schedule(static) reduction(*:prod)
for( int i=0;i<N;++i ) {
  prod *= v[i];
}
