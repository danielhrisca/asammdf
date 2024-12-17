cmake_minimum_required(VERSION 3.26...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C VERSION ${SKBUILD_PROJECT_VERSION})

find_package(
  Python
  REQUIRED
  COMPONENTS
  Interpreter
  Development.Module
  ${SKBUILD_SABI_COMPONENT}
  NumPy
  )

python_add_library(cutils MODULE WITH_SOABI USE_SABI 3.11 src/asammdf/blocks/cutils.c)

if(DEFINED ENV{LIBDEFLATE_PREFIX})
    message(STATUS "Finding libdeflate in $ENV{LIBDEFLATE_PREFIX}")
    target_include_directories(_deflate PUBLIC "$ENV{LIBDEFLATE_PREFIX}/include")
    target_link_directories(_deflate PUBLIC "$ENV{LIBDEFLATE_PREFIX}/lib")
    target_link_libraries(_deflate PRIVATE deflate)
else()
    message(STATUS "Building and linking bundled libdeflate222222")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(LIBDEFLATE_BUILD_SHARED_LIB OFF)
    add_subdirectory(libdeflate EXCLUDE_FROM_ALL)
    target_link_libraries(cutils PRIVATE libdeflate_static)
endif()

target_link_libraries(cutils PRIVATE Python::NumPy)

install(TARGETS cutils LIBRARY DESTINATION "asammdf/blocks")
