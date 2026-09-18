# ============================================================================
#  MatrixFlash-Pro benchmarks
#  ---------------------------------------------------------------------------
#  All benchmark programs share one measurement harness
#  (support/benchmark_support.hpp) and are linked against the
#  matrix_pro_bench_support library, so every executable only compiles its own
#  thin main() and links the shared bench code.
#
#     matrix_pro_bench_<name>   one workload family, full CLI + CSV/JSON export
#     matrix_pro_bench_all      runs every family in sequence
# ============================================================================

find_package(Threads REQUIRED)

# --- shared harness + workload bodies ---------------------------------------
add_library(matrix_pro_bench_support STATIC
	support/benchmark_support.cpp
	bench_matmul.cpp
	bench_elementwise.cpp
	bench_reductions.cpp
	bench_activations.cpp
	bench_linalg.cpp
	bench_conv.cpp
	bench_precision.cpp
	bench_sparse.cpp
	bench_training.cpp
)
target_link_libraries(matrix_pro_bench_support PUBLIC matrix_pro Threads::Threads)
target_include_directories(matrix_pro_bench_support PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/support)
matrix_pro_set_warnings(matrix_pro_bench_support)

# --- one executable per workload family --------------------------------------
set(MATRIX_PRO_BENCHES
	matmul
	elementwise
	reductions
	activations
	linalg
	conv
	precision
	sparse
	training
)

foreach(bench IN LISTS MATRIX_PRO_BENCHES)
	add_executable(matrix_pro_bench_${bench} support/bench_main.cpp)
	target_compile_definitions(matrix_pro_bench_${bench} PRIVATE
		MATRIX_PRO_SINGLE_BENCH="${bench}")
	target_link_libraries(matrix_pro_bench_${bench} PRIVATE matrix_pro_bench_support)
	matrix_pro_set_warnings(matrix_pro_bench_${bench})
endforeach()

# --- aggregate runner: every family, one report -------------------------------
add_executable(matrix_pro_bench_all support/bench_main.cpp)
target_compile_definitions(matrix_pro_bench_all PRIVATE MATRIX_PRO_SINGLE_BENCH="")
target_link_libraries(matrix_pro_bench_all PRIVATE matrix_pro_bench_support)
matrix_pro_set_warnings(matrix_pro_bench_all)

# Convenience target: `cmake --build build --target matrix_pro_benchmarks`
add_custom_target(matrix_pro_benchmarks COMMENT "MatrixFlash-Pro: all benchmark executables")
foreach(bench IN LISTS MATRIX_PRO_BENCHES)
	add_dependencies(matrix_pro_benchmarks matrix_pro_bench_${bench})
endforeach()
add_dependencies(matrix_pro_benchmarks matrix_pro_bench_all)

# Keep the benchmark outputs out of the source tree by default; CI can point
# --csv/--json at benchmarks/results explicitly.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)
