


#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)


SET(CURRENT_PACKAGE ShyLU)
CONFIGURE_FILE("${Trilinos_SOURCE_DIR}/packages/Trilinos_DLLExportMacro.h.in"
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_DLLExportMacro.h)


#
# B) Define the header and source files (and include directories)
#

SET(HEADERS "")
SET(SOURCES "")


# Set the base Trilinos directory in order to find Trilinos_version.h

ASSERT_DEFINED(PROJECT_SOURCE_DIR)
ASSERT_DEFINED(PROJECT_BINARY_DIR)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

#
# Include packages/src directory
#
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/interfaces)


APPEND_SET(HEADERS
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  src/shylu_internal_gmres.h
  src/shylu_internal_gmres_tools.h
  src/IQRSolver.h
  src/shylu_debug_manager.hpp
  src/Ifpack_ShyLU.h
  src/shylu_amesos_schur_operator.h
  src/shylu_util_decl.hpp
  src/shylu_symbolic.hpp
  src/shylu.hpp
  src/shylu_factor.hpp
  src/shylu_solve.hpp
  src/shylu_probing_operator.h
  src/shylu_local_schur_operator.h
  src/shylu_internal.hpp
  src/shylu_config.hpp
  src/shylu_partition_interface.hpp
  src/shylu_partition_interface_def.hpp
  src/shylu_partition_interface_decl.hpp
  src/shylu_directsolver_interface.hpp
  src/shylu_directsolver_interface_def.hpp
  src/shylu_directsolver_interface_decl.hpp
  )

# Set sources

APPEND_SET(SOURCES
  src/IQRSolver.cpp
  src/Ifpack_ShyLU.cpp
  src/shylu_amesos_schur_operator.cpp
  src/shylu_probing_operator.cpp
  src/shylu_local_schur_operator.cpp
  src/shylu_partition_interface.cpp
  src/shylu_directsolver_interface.cpp
  )

TRIBITS_ADD_LIBRARY(shylu
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )
