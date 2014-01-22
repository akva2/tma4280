project(hello-openmp)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(../common common)

include_directories(${COMMON_INCLUDE_PATHS})
add_definitions(${COMMON_DEFINITIONS})
list(APPEND CMAKE_C_FLAGS ${COMMON_C_FLAGS})

add_executable(integrate-openmp integrate.c)
target_link_libraries(integrate-openmp ${COMMON_LIBRARIES})
