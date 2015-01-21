project(cxxmatrix)
cmake_minimum_required(VERSION 2.6)

enable_language(CXX)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)

add_executable(cxxmatrix testmatrix.cpp)
