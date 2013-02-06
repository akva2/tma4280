# Name of our project
PROJECT(MPISum)

# Tell cmake which version we require - 2.6 is enough for us
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# We use the C language
ENABLE_LANGUAGE(C)

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
  FIND_PACKAGE(BLAS REQUIRED)
  SET(DEPLIBS ${DEPLIBS} ${BLAS_LIBRARIES})
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
ADD_LIBRARY(common common.c)

# Macro for recording targets
MACRO(ADD_APP APP SOURCES)
  ADD_EXECUTABLE(${APP} ${SOURCES})
  TARGET_LINK_LIBRARIES(${APP} common ${DEPLIBS})
  SET(all-bin "${all-bin} ${APP}")
ENDMACRO(ADD_APP)

# Add a program consisting of these sources
ADD_APP(mpi-splitv mpi-splitv.c)
ADD_APP(mpi-splitv-blas mpi-splitv-blas.c)
ADD_APP(mpi-splitv-blas-l3 mpi-splitv-blas-l3.c)

IF(OPENMP_FOUND)
  ADD_APP(hybrid-splitv hybrid-splitv.c)
  ADD_APP(hybrid-splitv-blas hybrid-splitv-blas.c)
  ADD_APP(hybrid-splitv-blas-l3 hybrid-splitv-blas-l3.c)
ENDIF(OPENMP_FOUND)

# Generate convenience script
CONFIGURE_FILE(runall.sh.in runall.sh @ONLY)
