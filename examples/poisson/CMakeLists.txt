# Name of our project
PROJECT(Poisson)

# Tell cmake which version we require - 2.6 is enough for us
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# We use the C language
ENABLE_LANGUAGE(C)
ENABLE_LANGUAGE(Fortran)

# We want C99
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG("-std=c99" HAVE_C99)

IF (HAVE_C99)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
ELSE(HAVE_C99)
  MESSAGE(WARNING "C99 support not available, program may fail to compile!")
ENDIF(HAVE_C99)

OPTION(USE_OPENMP "Use OpenMP?" ON)
OPTION(USE_MPI "Use MPI?" ON)

# We want OpenMP
IF(USE_OPENMP)
  FIND_PACKAGE(OpenMP)
ENDIF(USE_OPENMP)
IF(USE_MPI)
  FIND_PACKAGE(MPI)
ENDIF(USE_MPI)
IF(CMAKE_COMPILER_IS_GNUCC)
  FIND_PACKAGE(LAPACK REQUIRED)
  SET(DEPLIBS ${DEPLIBS} ${LAPACK_LIBRARIES})
ELSE(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mkl=sequential -DHAVE_MKL")
  SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mkl=sequential")
ENDIF(CMAKE_COMPILER_IS_GNUCC)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP=1")
ENDIF(OPENMP_FOUND)
IF(MPI_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_MPI=1")
  SET(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})
  SET(DEPLIBS ${DEPLIBS} ${MPI_C_LIBRARIES})
ENDIF(MPI_FOUND)

# Send the include directories to the compiler
INCLUDE_DIRECTORIES(${INCLUDES})

# Add a library with common sources
ADD_LIBRARY(common common.c fst.f)

# Macro for recording targets
MACRO(ADD_APP APP)
  ADD_EXECUTABLE(${APP} ${APP}.c)
  TARGET_LINK_LIBRARIES(${APP} common ${DEPLIBS})
  SET_TARGET_PROPERTIES(${APP} PROPERTIES LINKER_LANGUAGE C)
  SET(all-bin "${all-bin} ${APP}")
ENDMACRO(ADD_APP)

# Add a program consisting of these sources
IF(NOT MPI_FOUND)
  ADD_APP(poisson-lu)
  ADD_APP(poisson-cholesky)
  ADD_APP(poisson-banded-cholesky)
  ADD_APP(poisson-diag)
  ADD_APP(poisson-diag-fst)
  ADD_APP(poisson-gj)
  ADD_APP(poisson-gs)
  ADD_APP(poisson-cg)
  ADD_APP(poisson-cg-local)
  ADD_APP(poisson-cg-tensor)
ENDIF()
ADD_APP(poisson-gj-matrixless)
IF(NOT MPI_FOUND)
ADD_APP(poisson-gs-matrixless)
ADD_APP(poisson-cg-matrixless)
ADD_APP(poisson-pcg)
ENDIF()

# For generating the doxy
ADD_CUSTOM_TARGET(doc doxygen 
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  COMMENT "Generating API documentation" VERBATIM)

# Generate convenience script
CONFIGURE_FILE(runall.sh.in runall.sh @ONLY)
