project(poisson)
cmake_minimum_required(VERSION 2.6)

enable_language(CXX)

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

find_package(OpenMP REQUIRED)

# Add local modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package (Petsc REQUIRED)
include_directories(${PETSC_INCLUDES})
add_executable(poisson-petsc poisson-petsc.cpp)
target_link_libraries(poisson-petsc ${PETSC_LIBRARIES})

add_executable(poisson-petsc-vecsmart poisson-petsc-vecsmart.cpp)
target_link_libraries(poisson-petsc-vecsmart ${PETSC_LIBRARIES})

add_executable(poisson-petsc-smart poisson-petsc-smart.cpp)
target_link_libraries(poisson-petsc-smart ${PETSC_LIBRARIES})

IF(NOT USE_MPI)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
  add_executable(poisson-petsc-smart-threads poisson-petsc-smart-threads.cpp)
  target_link_libraries(poisson-petsc-smart-threads ${PETSC_LIBRARIES})
ENDIF()
