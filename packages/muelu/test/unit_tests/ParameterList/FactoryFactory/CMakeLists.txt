
# Note about the use of wildcard in CMakeLists.txt: CMake dont't know
# when new files is added. You need to re-run CMake manually to
# regenerate a build system incorporating the new file.
# YOU MUST ALSO TOUCH A CMAKE CONFIGURATION FILE WHEN YOU PUSH THE NEW
# FILE TO FORCE THE RECONFIGURE ON OTHER PEOPLE'S BUILDS.
FILE(GLOB xmlFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.xml)
list(REMOVE_ITEM xmlFiles DirectSolver.xml)

IF ( NOT ${PACKAGE_NAME}_ENABLE_Experimental)
  list(REMOVE_ITEM xmlFiles BlockedPFactory.xml)
ENDIF()

TRIBITS_COPY_FILES_TO_BINARY_DIR(ParameterList_FactoryFactory_cp
  SOURCE_FILES ${xmlFiles}
  )
