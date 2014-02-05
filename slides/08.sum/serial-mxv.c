void MxV(double* u, double** A, double* v, int N)
{
    for( int i=0;i<N;++i) {
        u[i] = 0;
        for( int j=0;j<N;++j )
            u[i] += A[i][j]*v[j];
    }
}

double innerproduct(double* u, double* v, int N)
{
    double result=0;
    for( int i=0;i<N;++i )
        result += u[i]*v[i];
}

