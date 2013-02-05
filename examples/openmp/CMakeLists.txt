# Name of our project
PROJECT(OpenMP)

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

# We want OpenMP
IF(USE_OPENMP)
  FIND_PACKAGE(OpenMP)
ENDIF(USE_OPENMP)
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
ADD_APP(serial serial.c)
ADD_APP(serial-blas serial-blas.c)
ADD_APP(serial-blas-l3 serial-blas-l3.c)
ADD_APP(omp-micro omp-micro.c)
ADD_APP(omp-macro omp-macro.c)

IF(USE_OPENMP)
  ADD_APP(omp-macro-conservative omp-macro-conservative.c)
  ADD_APP(omp-blas omp-blas.c)
  ADD_APP(omp-blas-l3 omp-blas-l3.c)
ENDIF(USE_OPENMP)

# Generate convenience script
CONFIGURE_FILE(runall.sh.in runall.sh @ONLY)
