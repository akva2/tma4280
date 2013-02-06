# Name of our project
PROJECT(DotProduct)

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

# Add a program consisting of these sources
ADD_EXECUTABLE(serial serial.c)
TARGET_LINK_LIBRARIES(serial common ${DEPLIBS})

# Add a program consisting of these sources
ADD_EXECUTABLE(serial-blas serial-blas.c)
TARGET_LINK_LIBRARIES(serial-blas common ${DEPLIBS})

# Add a program consisting of these sources
ADD_EXECUTABLE(serial-blas-l3 serial-blas-l3.c)
TARGET_LINK_LIBRARIES(serial-blas-l3 common ${DEPLIBS})

# Add a program consisting of these sources
ADD_EXECUTABLE(omp-micro omp-micro.c)
TARGET_LINK_LIBRARIES(omp-micro common ${DEPLIBS})

# Add a program consisting of these sources
ADD_EXECUTABLE(omp-macro omp-macro.c)
TARGET_LINK_LIBRARIES(omp-macro common ${DEPLIBS})

# Add a program consisting of these sources
IF(USE_OPENMP)
  ADD_EXECUTABLE(omp-macro-conservative omp-macro-conservative.c)
  TARGET_LINK_LIBRARIES(omp-macro-conservative common ${DEPLIBS})

  # Add a program consisting of these sources
  ADD_EXECUTABLE(omp-blas omp-blas.c)
  TARGET_LINK_LIBRARIES(omp-blas common ${DEPLIBS})

  # Add a program consisting of these sources
  ADD_EXECUTABLE(omp-blas-l3 omp-blas-l3.c)
  TARGET_LINK_LIBRARIES(omp-blas-l3 common ${DEPLIBS})
ENDIF(USE_OPENMP)
