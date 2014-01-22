double dosum(double** A, double** v, int K, int N)
{
    double alpha=0;
    double** temp = createMatrix(K,N);
#pragma omp parallel for schedule(static) \
	reduction(+:alpha)
    for( int i=0;i<K;++i ) {
        MxV(temp[i],A,v[i],N);
        alpha += innerproduct(temp[i],v[i],N);
    }

    return alpha;
}
