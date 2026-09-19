# ============================================================================
#  MatrixFlash-Pro test suite
#  ---------------------------------------------------------------------------
#  unit/         op-level behavior: every assertion is a mathematical fact
#  integration/  cross-module flows (autograd, training ops, CUDA infra)
#
#  Registration is table-driven: add the file to the list, nothing else.
#  Every executable is registered as a CTest test and carries the `unit` or
#  `integration` label so subsets can be run with:
#     ctest -C Release -L unit
# ============================================================================

find_package(Threads REQUIRED)

set(MATRIX_PRO_UNIT_TESTS
	unit/test_matrix.cpp
	unit/test_operations.cpp
	unit/test_linalg.cpp
	unit/test_errors.cpp
	unit/test_transforms.cpp
	unit/test_indexing.cpp
	unit/test_sparse.cpp
	unit/test_view.cpp
	unit/test_dtype_precision.cpp
	unit/test_conv.cpp
	unit/test_math.cpp
	unit/test_tensor_advanced.cpp
	unit/test_linalg_extended.cpp
	unit/test_sparse_formats.cpp
	unit/test_rng_extended.cpp
)

set(MATRIX_PRO_INTEGRATION_TESTS
	integration/test_autograd.cpp
	integration/test_ml.cpp
	integration/test_batch.cpp
	integration/test_infra.cpp
	integration/test_new_features.cpp
	integration/test_repro2.cpp
	integration/test_execution.cpp
	integration/test_serialization.cpp
	integration/test_autograd_extended.cpp
)

# --- unit tests --------------------------------------------------------------
foreach(source IN LISTS MATRIX_PRO_UNIT_TESTS)
	get_filename_component(name ${source} NAME_WE)
	add_executable(matrix_pro_${name} ${source})
	target_link_libraries(matrix_pro_${name} PRIVATE matrix_pro)
	target_include_directories(matrix_pro_${name} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/support)
	matrix_pro_set_warnings(matrix_pro_${name})
	add_test(NAME matrix_pro_${name} COMMAND matrix_pro_${name})
	set_tests_properties(matrix_pro_${name} PROPERTIES
		LABELS unit
		TIMEOUT 300)
endforeach()

# --- integration tests -------------------------------------------------------
foreach(source IN LISTS MATRIX_PRO_INTEGRATION_TESTS)
	get_filename_component(name ${source} NAME_WE)
	add_executable(matrix_pro_${name} ${source})
	target_link_libraries(matrix_pro_${name} PRIVATE matrix_pro Threads::Threads)
	target_include_directories(matrix_pro_${name} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/support)
	matrix_pro_set_warnings(matrix_pro_${name})
	add_test(NAME matrix_pro_${name} COMMAND matrix_pro_${name})
	set_tests_properties(matrix_pro_${name} PROPERTIES
		LABELS integration
		TIMEOUT 300)
endforeach()

# --- convenience umbrella target: `cmake --build build --target matrix_pro_tests`
add_custom_target(matrix_pro_tests
	COMMENT "MatrixFlash-Pro: all test executables")
foreach(source IN LISTS MATRIX_PRO_UNIT_TESTS MATRIX_PRO_INTEGRATION_TESTS)
	get_filename_component(name ${source} NAME_WE)
	add_dependencies(matrix_pro_tests matrix_pro_${name})
endforeach()
