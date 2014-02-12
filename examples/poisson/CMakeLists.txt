project(poisson-example)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(../common common)

enable_language(Fortran)

add_definitions(${COMMON_DEFINITIONS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_C_FLAGS}")
include_directories(${COMMON_INCLUDE_PATHS})

add_library(poisson_common solvers.c poissoncommon.c fst.f)

add_executable(poisson1D poisson1D.c)
target_link_libraries(poisson1D poisson_common ${COMMON_LIBRARIES})

add_executable(poisson2D poisson2D.c)
target_link_libraries(poisson2D poisson_common ${COMMON_LIBRARIES})

enable_testing()
add_test(poisson1D-lu poisson1D 100 1)
add_test(poisson1D-cholesky poisson1D 100 2)
add_test(poisson1D-gj poisson1D 100 3)
add_test(poisson1D-gj-blas poisson1D 100 4)
add_test(poisson1D-gs poisson1D 100 5)
add_test(poisson1D-gs-blas poisson1D 100 6)
add_test(poisson1D-gj-matrixless poisson1D 100 7)
add_test(poisson1D-gs-matrixless poisson1D 100 8)
add_test(poisson1D-gs-matrixless-rb poisson1D 99 9)
add_test(poisson1D-diagonalization poisson1D 100 10)
add_test(poisson1D-diagonalization-fst poisson1D 128 11)

add_test(poisson2D-lu poisson2D 30 1)
add_test(poisson2D-cholesky poisson2D 30 2)
add_test(poisson2D-gj poisson2D 30 3)
add_test(poisson2D-gj-blas poisson2D 30 4)
add_test(poisson2D-gs poisson2D 30 5)
add_test(poisson2D-gs-blas poisson2D 30 6)
add_test(poisson2D-gj-matrixless poisson2D 30 7)
add_test(poisson2D-gs-matrixless-local poisson2D 30 8)
add_test(poisson2D-diagonalization poisson2D 30 10)
add_test(poisson2D-diagonalization-fst poisson2D 32 11)
