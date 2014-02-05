project(summing-example)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(../common common)

add_definitions(${COMMON_DEFINITIONS})
list(APPEND CMAKE_C_FLAGS ${COMMON_C_FLAGS})
include_directories(${COMMON_INCLUDE_PATHS})

add_executable(serial serial.c)
target_link_libraries(serial ${COMMON_LIBRARIES})

if(OPENMP_FOUND)
  add_executable(omp-micro omp-micro.c)
  target_link_libraries(omp-micro ${COMMON_LIBRARIES})

  # This translates to OMP_NUM_THREADS=2 <dir>/omp-micro 32 1024
  add_test(omp-micro ${PROJECT_BINARY_DIR}/omp-micro 32 1024)
  set_tests_properties(omp-micro PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

  add_executable(omp-macro omp-macro.c)
  target_link_libraries(omp-macro ${COMMON_LIBRARIES})

  # This translates to OMP_NUM_THREADS=2 <dir>/omp-macro 32 1024 
  add_test(omp-macro ${PROJECT_BINARY_DIR}/omp-macro 32 1024)
  set_tests_properties(omp-macro PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

  add_executable(omp-macro-conservative omp-macro-conservative.c)
  target_link_libraries(omp-macro-conservative ${COMMON_LIBRARIES})

  # This translates to OMP_NUM_THREADS=2 <dir>/omp-macro 32 1024 
  add_test(omp-macro-conservative ${PROJECT_BINARY_DIR}/omp-macro-conservative 32 1024)
  set_tests_properties(omp-macro-conservative PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)

  add_executable(omp-macro-blas omp-macro-blas.c)
  target_link_libraries(omp-macro-blas ${COMMON_LIBRARIES})

  # This translates to OMP_NUM_THREADS=2 <dir>/omp-macro-blas 32 1024 
  add_test(omp-macro-blas ${PROJECT_BINARY_DIR}/omp-macro-blas 32 1024)
  set_tests_properties(omp-macro-blas PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
endif()

if(MPI_FOUND)
  add_executable(mpi-splitv mpi-splitv.c)
  target_link_libraries(mpi-splitv ${COMMON_LIBRARIES})
  add_test(mpi-splitv ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/mpi-splitv 32 16384)

  add_executable(mpi-splitv-blas mpi-splitv-blas.c)
  target_link_libraries(mpi-splitv-blas ${COMMON_LIBRARIES})
  add_test(mpi-splitv-blas ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/mpi-splitv-blas 32 16384)

  add_executable(mpi-splitav mpi-splitav.c)
  target_link_libraries(mpi-splitav ${COMMON_LIBRARIES})
  add_test(mpi-splitav ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/mpi-splitav 32 16384)
endif()
