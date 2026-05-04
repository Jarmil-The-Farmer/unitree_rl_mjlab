cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  message("Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -fPIC")
else()
  message("Release mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -DNDEBUG -fPIC")
endif()

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")
find_package(unitree_sdk2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)


# MuJoCo C library — downloaded by simulate/download_mujoco.sh.
# Override with -DMUJOCO_DIR=/path/to/mujoco if installed elsewhere.
if(NOT DEFINED MUJOCO_DIR)
  set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mujoco")
endif()
if(NOT EXISTS "${MUJOCO_DIR}/lib/libmujoco.so")
  message(FATAL_ERROR
    "MuJoCo not found at ${MUJOCO_DIR}/lib/libmujoco.so\n"
    "Run: ./simulate/download_mujoco.sh")
endif()

include_directories(
  ${MUJOCO_DIR}/include
  ${MUJOCO_DIR}/simulate
  src/lodepng
  ${CMAKE_CURRENT_SOURCE_DIR}/../deploy/robots/g1/include
)
link_directories(${MUJOCO_DIR}/lib)

file(GLOB SIM_SRC
  src/joystick/joystick.cc

  # exclude mujoco/simulate/main.cc
  ${MUJOCO_DIR}/simulate/glfw_*.cc
  ${MUJOCO_DIR}/simulate/platform_*.cc
  ${MUJOCO_DIR}/simulate/simulate.cc
  src/lodepng/lodepng.cpp
)

link_libraries(
  pthread
  mujoco
  glfw
  yaml-cpp
  unitree_sdk2
  boost_program_options
  fmt
  zmq
  turbojpeg
)

add_executable(unitree_mujoco
  ${SIM_SRC}
  src/main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/../deploy/robots/g1/src/inspire_hand_ctrl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../deploy/robots/g1/src/inspire_hand_state.cpp
)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)
