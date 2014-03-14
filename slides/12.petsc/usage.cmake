# Add local modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(Petsc REQUIRED)

include_directories(${PETSC_INCLUDES})

add_executable(petsc-test petsc-test.c)
target_link_libraries(petsc-test ${PETSC_LIBRARIES})
