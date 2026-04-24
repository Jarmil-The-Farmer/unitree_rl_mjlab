cmake_minimum_required(VERSION 3.12)
project(sinusoid_response_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(
  /usr/local/include
  /usr/local/include/ddscxx
)

link_directories(/usr/local/lib)

add_executable(sinusoid_response sinusoid_response.cpp)

target_link_libraries(sinusoid_response
  unitree_sdk2
  /usr/local/lib/libddsc.so
  /usr/local/lib/libddscxx.so
  pthread
  rt
)
