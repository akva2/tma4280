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

# We want OpenMP
FIND_PACKAGE(OpenMP)
FIND_PACKAGE(BLAS)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# Send the include directories to the compiler
INCLUDE_DIRECTORIES(${INCLUDES})

# Add a program consisting of these sources
ADD_EXECUTABLE(serial serial.c)

# Add a program consisting of these sources
ADD_EXECUTABLE(omp-micro omp-micro.c)

# Add a program consisting of these sources
ADD_EXECUTABLE(omp-macro omp-macro.c)

# Add a program consisting of these sources
ADD_EXECUTABLE(omp-macro-conservative omp-macro-conservative.c)

IF(BLAS_LIBRARIES)
  ADD_EXECUTABLE(omp-macro-conservative-blas omp-macro-conservative-blas.c)
  TARGET_LINK_LIBRARIES(omp-macro-conservative-blas ${BLAS_LIBRARIES})
ENDIF(BLAS_LIBRARIES)
