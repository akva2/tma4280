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

  add_executable(omp-macro omp-macro.c)
  target_link_libraries(omp-macro ${COMMON_LIBRARIES})

  add_executable(omp-macro-conservative omp-macro-conservative.c)
  target_link_libraries(omp-macro-conservative ${COMMON_LIBRARIES})
endif()

if(MPI_FOUND)
  add_executable(mpi-splitv mpi-splitv.c)
  target_link_libraries(mpi-splitv ${COMMON_LIBRARIES})

  add_executable(mpi-splitav mpi-splitav.c)
  target_link_libraries(mpi-splitav ${COMMON_LIBRARIES})
endif()
