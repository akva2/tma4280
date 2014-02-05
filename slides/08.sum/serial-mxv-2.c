double dosum(double** A, double** v, int K, int N)
{
    double alpha=0;
    double temp[N];
    for( int i=0;i<K;++i ) {
        MxV(temp,A,v[i],N);
        alpha += innerproduct(temp,v[i],N);
    }

    return alpha;
}
