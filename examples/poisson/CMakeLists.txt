project(poisson-example)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(../common common)

add_definitions(${COMMON_DEFINITIONS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_C_FLAGS}")
include_directories(${COMMON_INCLUDE_PATHS})

add_executable(poisson1D poisson1D.c)
target_link_libraries(poisson1D ${COMMON_LIBRARIES})
