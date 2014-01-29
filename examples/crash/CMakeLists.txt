project(hello)
enable_language(C)
cmake_minimum_required(VERSION 2.6)

add_executable(hello hello.c)
add_executable(hello2 hello2.c utils.c)
add_executable(timing timing.c)

add_library(utils utils.c utils2.c)
add_executable(hello3 hello3.c)
target_link_libraries(hello3 utils)
add_executable(hello4 hello4.c)
target_link_libraries(hello4 utils)

enable_testing()
add_test(hello  ${PROJECT_BINARY_DIR}/hello)
add_test(hello2 ${PROJECT_BINARY_DIR}/hello2 10)
add_test(hello3 ${PROJECT_BINARY_DIR}/hello3 10)
add_test(hello4 ${PROJECT_BINARY_DIR}/hello4 10)
