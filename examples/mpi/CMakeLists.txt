project(hellompi)
cmake_minimum_required(VERSION 2.6)

enable_language(C)
enable_language(Fortran)

add_subdirectory(../common common)

add_definitions(${COMMON_DEFINITIONS})
list(APPEND CMAKE_C_FLAGS ${COMMON_C_FLAGS})

include_directories(${COMMON_INCLUDE_PATHS})

add_executable(hellompi hellompi.c)
target_link_libraries(hellompi ${COMMON_LIBRARIES})

add_executable(integrate integrate.c)
target_link_libraries(integrate ${COMMON_LIBRARIES})

add_executable(hellompi-f hellompi.f90)
target_link_libraries(hellompi-f ${COMMON_LIBRARIES})

enable_testing()
add_test(hellompi ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/hellompi)
add_test(hellompi-f ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/hellompi-f)
