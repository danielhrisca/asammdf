cmake_minimum_required(VERSION 3.15...3.26)

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES C
  VERSION ${SKBUILD_PROJECT_VERSION})

find_package(
  Python
  COMPONENTS
  Interpreter
  Development.Module
  ${SKBUILD_SABI_COMPONENT}
  NumPy
  REQUIRED)

python_add_library(cutils
                   MODULE
                   src/asammdf/blocks/cutils.c
                   WITH_SOABI USE_SABI 3.9)

target_link_libraries(cutils PRIVATE Python::NumPy)

install(TARGETS cutils DESTINATION "asammdf/blocks")
