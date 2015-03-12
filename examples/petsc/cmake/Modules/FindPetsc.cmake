find_path(
  PETSC_INCLUDE_DIR
  NAMES petsc.h
  PATHS /usr/include/petsc/
  $ENV{PETSC_DIR}/include
  )
find_path(
  PETSCCONF_INCLUDE_DIR
  NAMES petscconf.h
  PATHS /usr/include/petsc/ 
  $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include
  )

include($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/conf/PETScConfig.cmake)
find_library(PETSC_LIB_PETSC     petsc  HINTS $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
set(PETSC_LIBRARIES ${PETSC_LIB_PETSC} ${PETSC_PACKAGE_LIBS})
set(PETSC_INCLUDES ${PETSC_INCLUDE_DIR} ${PETSCCONF_INCLUDE_DIR}
                   ${PETSC_PACKAGE_INCLUDES})
mark_as_advanced(PETSC_LIBRARIES PETSC_INCLUDES)
