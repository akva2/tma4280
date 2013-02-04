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
FIND_PACKAGE(BLAS REQUIRED)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_OPENMP=1")
ENDIF(OPENMP_FOUND)

# Send the include directories to the compiler
INCLUDE_DIRECTORIES(${INCLUDES})

# Add a library with common sources
ADD_LIBRARY(common common.c)

SET(DEPLIBS ${BLAS_LIBRARIES})

# Add a program consisting of these sources
ADD_EXECUTABLE(serial serial.c)
TARGET_LINK_LIBRARIES(serial common ${DEPLIBS})

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
  ADD_EXECUTABLE(omp-macro-conservative-blas omp-macro-conservative-blas.c)
  TARGET_LINK_LIBRARIES(omp-macro-conservative-blas common ${DEPLIBS})

  ADD_EXECUTABLE(omp-macro-conservative-blas-timing omp-macro-conservative-blas-timing.c)
  TARGET_LINK_LIBRARIES(omp-macro-conservative-blas-timing common ${DEPLIBS})
ENDIF(USE_OPENMP)
