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

python_add_library(cutils MODULE WITH_SOABI USE_SABI 3.9 src/asammdf/blocks/cutils.c)

target_link_libraries(cutils PRIVATE Python::NumPy)

install(TARGETS cutils DESTINATION "asammdf/blocks")
