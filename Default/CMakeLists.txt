
cmake_minimum_required(VERSION 3.13)
project(Example LANGUAGES CXX)

find_package(KokkosRemote REQUIRED)


file(GLOB SOURCES *.cpp)

foreach(SRC_FILE ${SOURCES})
  message(STATUS "Adding source file:" ${SRC_FILE})
  get_filename_component(SRC_FILE_NAME ${SRC_FILE} NAME)
  string(REGEX REPLACE "\\.[^.]*$" "" SRC_FILE_NAME ${SRC_FILE_NAME})
  add_executable(${SRC_FILE_NAME} ${SRC_FILE})
  target_link_libraries(${SRC_FILE_NAME} Kokkos::kokkosremote)
endforeach()
