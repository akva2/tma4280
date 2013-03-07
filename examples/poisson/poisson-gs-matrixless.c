#include <stdlib.h>
#include <stdio.h>

#include <math.h>

#include "common.h"

/*void GS(Vector u, double tolerance, int maxit)*/
/*{*/
/*  int it=0;*/
/*  Vector b = createVector(u->len);*/
/*  Vector e = createVector(u->len);*/
/*  Vector v = createVector(u->len);*/
/*  int M = sqrt(u->len);*/
/*  int* sizes, *displ;*/
/*  splitVector(M, 2*max_threads(), &sizes, &displ);*/
/*  copyVector(b, u);*/
/*  fillVector(u, 0.0);*/
/*  double max = tolerance+1;*/
/*  while (max > tolerance && ++it < maxit) {*/
/*    copyVector(e, u);*/
/*    copyVector(u, b);*/
/*    for (int color=0;color<2;++color) {*/
/*      for (int i=0;i<M;++i) {*/
/*#pragma omp parallel*/
/*        {*/
/*          int cnt=i*M+displ[get_thread()*2+color];*/
/*          for (int j=0;j<sizes[get_thread()*2+color];++j, ++cnt) {*/
/*            if (j+displ[get_thread()*2+color] > 0)*/
/*              u->data[cnt] += v->data[cnt-1];*/
/*            if (j+displ[get_thread()*2+color] < M-1)*/
/*              u->data[cnt] += v->data[cnt+1];*/
/*            if (i > 0)*/
/*              u->data[cnt] += v->data[cnt-M];*/
/*            if (i < M-1)*/
/*              u->data[cnt] += v->data[cnt+M];*/
/*            u->data[cnt] /= 4.0;*/
/*            v->data[cnt] = u->data[cnt];*/
/*          }*/
/*        }*/
/*      }*/
/*    }*/
/*    axpy(e, u, -1.0);*/
/*    max = sqrt(innerproduct(e, e));*/
/*  }*/
/*  printf("number of iterations %i %f\n", it, max);*/
/*  freeVector(b);*/
/*  freeVector(e);*/
/*  freeVector(v);*/
/*  free(sizes);*/
/*  free(displ);*/
/*}*/

void GS(Matrix u, double tolerance, int maxit)
{
  int it=0;
  Matrix b = cloneMatrix(u);
  Matrix e = cloneMatrix(u);
  Matrix v = cloneMatrix(u);
  int* sizes, *displ;
  splitVector(u->rows-2, 2*max_threads(), &sizes, &displ);
  copyVector(b->as_vec, u->as_vec);
  fillVector(u->as_vec, 0.0);
  double max = tolerance+1;
  while (max > tolerance && ++it < maxit) {
    copyVector(e->as_vec, u->as_vec);
    copyVector(u->as_vec, b->as_vec);
    for (int color=0;color<2;++color) {
      for (int i=1;i<u->cols-1;++i) {
#pragma omp parallel
        {
          int cnt=displ[get_thread()*2+color]+1;
          for (int j=0;j<sizes[get_thread()*2+color];++j, ++cnt) {
            u->data[i][cnt] += v->data[i][cnt-1];
            u->data[i][cnt] += v->data[i][cnt+1];
            u->data[i][cnt] += v->data[i-1][cnt];
            u->data[i][cnt] += v->data[i+1][cnt];
            u->data[i][cnt] /= 4.0;
            v->data[i][cnt] = u->data[i][cnt];
          }
        }
      }
    }
    axpy(e->as_vec, u->as_vec, -1.0);
    max = sqrt(innerproduct(e->as_vec, e->as_vec));
  }
  printf("number of iterations %i %f\n", it, max);
  freeMatrix(b);
  freeMatrix(e);
  freeMatrix(v);
  free(sizes);
  free(displ);
}

int main(int argc, char** argv)
{
  int rank, size;
  init_app(argc, argv, &rank, &size);

  if (argc < 2) {
    printf("usage: %s <N> [L]\n",argv[0]);
    close_app();
    return 1;
  }

  /* the total number of grid points in each spatial direction is (N+1) */
  /* the total number of degrees-of-freedom in each spatial direction is (N-1) */
  int N  = atoi(argv[1]);
  int M  = N-1;
  double L=1.0;
  if (argc > 2)
    L = atof(argv[2]);

  double h = L/N;

  Vector grid = createVector(M);
  for (int i=0;i<M;++i)
    grid->data[i] = (i+1)*h;

  int coords[2] = {0};
  int sizes[2] = {1};
#ifdef HAVE_MPI
  sizes[0] = sizes[1] = 0;
  MPI_Dims_create(size,2,sizes);
  int periodic[2];
  periodic[0] = periodic[1] = 0;
  MPI_Comm comm;
  MPI_Cart_create(MPI_COMM_WORLD,2,sizes,periodic,0,&comm);
  MPI_Cart_coords(comm,rank,2,coords);
#endif

  int* len[2];
  int* displ[2];
  splitVector(M, sizes[0], &len[0], &displ[0]);
  splitVector(M, sizes[1], &len[1], &displ[1]);

#ifdef HAVE_MPI
  Matrix u = createMatrixMPI(len[0][coords[0]]+2, len[1][coords[1]]+2, M, M, &comm);
#else
  Matrix u = createMatrix(M+2, M+2);
#endif
  evalMeshDispl(u, grid, grid, poisson_source,
                displ[0][coords[0]], displ[1][coords[1]]);
  scaleVector(u->as_vec, h*h);

  double time = WallTime();
  GS(u, 1e-6, 5000);

  evalMesh2Displ(u, grid, grid, exact_solution, -1.0,
                 displ[0][coords[0]], displ[1][coords[1]]);
  double max = maxNorm(u->as_vec);

  if (rank == 0) {
    printf("elapsed: %f\n", WallTime()-time);
    printf("max: %f\n", max);
  }

  freeMatrix(u);
  freeVector(grid);
  for (int i=0;i<2;++i) {
    free(len[i]);
    free(displ[i]);
  }

  close_app();
  return 0;
}
