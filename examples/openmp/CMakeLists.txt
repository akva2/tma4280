project(hello-openmp)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(../common common)

include_directories(${COMMON_INCLUDE_PATHS})
add_definitions(${COMMON_DEFINITIONS})
list(APPEND CMAKE_C_FLAGS ${COMMON_C_FLAGS})

add_executable(integrate-openmp integrate.c)
target_link_libraries(integrate-openmp ${COMMON_LIBRARIES})

enable_testing()

# This translates to OMP_NUM_THREADS=2 <dir>/integrate-openmp 10000
add_test(integrate-openmp ${PROJECT_BINARY_DIR}/integrate-openmp 10000)
set_tests_properties(integrate-openmp PROPERTIES ENVIRONMENT OMP_NUM_THREADS=2)
