#pragma omp parallel sections
{
#pragma omp parallel section
  {
    DoJob1();
  }
#pragma omp parallel section
  {
    DoJob2();
  }
#pragma omp parallel section
  {
    DoJob3();
  }
}
