# Install / export rules: makes MatrixFlash-Pro consumable through
#   find_package(matrix_pro REQUIRED)
#   target_link_libraries(app PRIVATE matrix_pro::matrix_pro)
#
# The exported target keeps the CUDA::cublas / CUDA::cusolver requirements as
# INTERFACE dependencies so downstream builds inherit the include paths and
# link libraries automatically.

include_guard(GLOBAL)

if(NOT MATRIX_PRO_INSTALL)
  return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS matrix_pro
        EXPORT matrix_proTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.cuh")

install(EXPORT matrix_proTargets
        FILE matrix_proTargets.cmake
        NAMESPACE matrix_pro::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix_pro)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/matrix_pro-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/matrix_pro-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix_pro)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/matrix_pro-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/matrix_pro-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/matrix_pro-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/matrix_pro)