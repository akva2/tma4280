void MxV(double* u, double** A, double* v, int N)
{
#pragma omp parallel for schedule(static)
    for( int i=0;i<N;++i) {
        u[i] = 0;
        for( int j=0;j<N;++j )
            u[i] += A[i][j]*v[j];
    }
}

double innerproduct(double* u, double* v, int N)
{
    double result=0;
#pragma omp parallel for schedule(static) \
        reduction(+:result)
    for( int i=0;i<N;++i )
        result += u[i]*v[i];
}

