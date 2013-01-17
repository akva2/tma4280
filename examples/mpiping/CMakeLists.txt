# Name of our project
PROJECT(PingMPI)

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

# We require MPI
FIND_PACKAGE(MPI REQUIRED)

# Add the MPI include path so that our program will find mpi.h
SET(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})

# The list of our dependency libraries - here it is just the MPI libraries
SET(DEPLIBS ${DEPLIBS} ${MPI_LIBRARIES})

# Send the include directories to the compiler
INCLUDE_DIRECTORIES(${INCLUDES})

# List of source files our programs consists of
SET(SRCS mpiping.c)

# Add a program consisting of these sources
ADD_EXECUTABLE(mpiping ${SRCS})

# Tell linker to link against the MPI libraries
TARGET_LINK_LIBRARIES(mpiping ${DEPLIBS})
