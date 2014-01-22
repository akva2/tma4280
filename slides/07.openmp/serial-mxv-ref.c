double dosum(double** A, double** v, int K, int N)
{
    double alpha=0;
    /* CHANGED */
    double** temp = createMatrix(K,N);
    for( int i=0;i<K;++i ) {
        /* CHANGED */
        MxV(temp[i],A,v[i],N);
        alpha += innerproduct(temp[i],v[i],N);
    }

    return alpha;
}
