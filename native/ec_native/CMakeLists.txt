cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(mujoco CONFIG REQUIRED)

set(EC_UNITREE_SDK_ROOT "" CACHE PATH "Optional Unitree SDK2 source root")
if(NOT EC_UNITREE_SDK_ROOT AND DEFINED ENV{EC_UNITREE_SDK_ROOT})
  set(EC_UNITREE_SDK_ROOT "$ENV{EC_UNITREE_SDK_ROOT}")
endif()
if(NOT EC_UNITREE_SDK_ROOT AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/unitree_sdk2/CMakeLists.txt")
  set(EC_UNITREE_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/unitree_sdk2")
endif()
set(EC_WITH_UNITREE OFF)
if(EC_UNITREE_SDK_ROOT)
  if(NOT EXISTS "${EC_UNITREE_SDK_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "EC_UNITREE_SDK_ROOT is not a Unitree SDK2 source tree")
  endif()
  set(BUILD_EXAMPLES OFF CACHE BOOL "Build Unitree SDK examples" FORCE)
  add_subdirectory(
    "${EC_UNITREE_SDK_ROOT}"
    "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2"
    EXCLUDE_FROM_ALL)
  set(EC_WITH_UNITREE ON)
endif()

include(FetchContent)
set(EC_ONNXRUNTIME_VERSION "1.28.0" CACHE STRING "Pinned ONNX Runtime version")
set(EC_ONNXRUNTIME_ROOT "" CACHE PATH "Existing ONNX Runtime release root")
if(NOT EC_ONNXRUNTIME_ROOT)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(FATAL_ERROR
      "Set EC_ONNXRUNTIME_ROOT for this platform; the automatic release is linux-x64")
  endif()
  FetchContent_Declare(
    onnxruntime_release
    URL "https://github.com/microsoft/onnxruntime/releases/download/v${EC_ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${EC_ONNXRUNTIME_VERSION}.tgz"
    URL_HASH "SHA256=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(onnxruntime_release)
  set(EC_ONNXRUNTIME_ROOT "${onnxruntime_release_SOURCE_DIR}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(
  onnxruntime PROPERTIES
  IMPORTED_LOCATION "${EC_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.28.0"
  INTERFACE_INCLUDE_DIRECTORIES "${EC_ONNXRUNTIME_ROOT}/include")

pybind11_add_module(
  _ec_native MODULE
  src/bindings.cpp
  src/native_backend.cpp
  src/native_fake_runtime.cpp
  src/native_tracker_core.cpp
  src/onnx_engine.cpp)
target_compile_definitions(_ec_native PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_options(_ec_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
target_link_libraries(_ec_native PRIVATE onnxruntime mujoco::mujoco)
set_target_properties(_ec_native PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
if(EC_WITH_UNITREE)
  target_sources(_ec_native PRIVATE src/unitree_backend.cpp src/mujoco_dds_plant.cpp)
  target_compile_definitions(_ec_native PRIVATE EC_WITH_UNITREE=1)
  target_link_libraries(_ec_native PRIVATE unitree_sdk2)
endif()
set_target_properties(_ec_native PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
if(UNIX AND NOT APPLE)
  target_link_libraries(_ec_native PRIVATE rt)
endif()

install(TARGETS _ec_native LIBRARY DESTINATION ec_native)
install(
  FILES
    "${EC_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1"
    "${EC_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.28.0"
    "${EC_ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so"
  DESTINATION ec_native/lib)
install(
  FILES
    "${EC_ONNXRUNTIME_ROOT}/LICENSE"
    "${EC_ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt"
  DESTINATION ec_native/licenses/onnxruntime)
if(EC_WITH_UNITREE)
  install(
    FILES
      "${EC_UNITREE_SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so"
      "${EC_UNITREE_SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so"
    DESTINATION ec_native/lib)
  install(
    FILES "${EC_UNITREE_SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddsc.so"
    DESTINATION ec_native/lib
    RENAME libddsc.so.0)
  install(
    FILES "${EC_UNITREE_SDK_ROOT}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}/libddscxx.so"
    DESTINATION ec_native/lib
    RENAME libddscxx.so.0)
  install(
    FILES "${EC_UNITREE_SDK_ROOT}/LICENSE"
    DESTINATION ec_native/licenses/unitree_sdk2)
endif()
