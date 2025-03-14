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
  
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LIBDEFLATE_COMPRESSION_SUPPORT OFF)
set(LIBDEFLATE_GZIP_SUPPORT OFF)
set(LIBDEFLATE_BUILD_GZIP OFF)
  
  # Add submodule libdeflate
add_subdirectory(ext/libdeflate EXCLUDE_FROM_ALL)

python_add_library(cutils MODULE WITH_SOABI USE_SABI 3.10 src/asammdf/blocks/cutils.c)

target_link_libraries(cutils PRIVATE Python::NumPy libdeflate::libdeflate_static)

install(TARGETS cutils DESTINATION "asammdf/blocks")
