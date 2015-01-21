// Simple program demonstrating use of the matrix class

#include "matrix.hh"
#include <chrono>
#include <random>

static const int rows = 3; // number of rows in test matrices
static const int cols = 6; // number of columns in test matrices

int main(int argc, char** argv)
{
  Matrix<int,ColMajor> Ai(rows,cols); // a column-major integer matrix
  Matrix<float>        Af(rows,cols); // a row-major single precision floating point matrix
  Matrix<double>       Ad(rows,cols); // a row-major double precision floating point matrix

  // grab seed from the current timestamp
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // instance a random number generator
  std::default_random_engine generator (seed);

  // instance a uniform distribution
  std::uniform_real_distribution<double> random(0.0, 1.0);

  // note that the loop order is not optimal for the integer matrix
  size_t k=1;
  for (size_t i=0;i<rows;++i) {
    for (size_t j=0;j<cols;++j, k++) {
      Ai(i,j) = k;
      Af(i,j) = k+random(generator); // add some random stuff for fun
      Ad(i,j) = k+random(generator); // add some random stuff for fun
    }
  }

  // print matrices to screen
  std::cout << "== Integer matrix ==" << std::endl;
  Ai.print();
  std::cout << "== Float matrix ==" << std::endl;
  Af.print(7);
  std::cout << "== Double matrix ==" << std::endl;
  Ad.print(16);

  return 0;
}
