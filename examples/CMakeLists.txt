# ============================================================================
#  MatrixFlash-Pro examples
#  ---------------------------------------------------------------------------
#  Every example is a standalone, runnable program. They double as compile-time
#  smoke tests for the public umbrella header <matrix_pro/matrix_pro.hpp>.
# ============================================================================

# Canonical smoke example: the smallest possible end-to-end use of the library.
add_executable(matrix_pro_basic basic_gpu_usage.cpp)
target_link_libraries(matrix_pro_basic PRIVATE matrix_pro)
matrix_pro_set_warnings(matrix_pro_basic)

# Showcase: device-resident matrices, cuBLAS GEMM throughput, fused / in-place
# kernels, zero-copy views and the async stream pool.
add_executable(matrix_pro_demo_high_perf high_performance_demo.cpp)
target_link_libraries(matrix_pro_demo_high_perf PRIVATE matrix_pro)
matrix_pro_set_warnings(matrix_pro_demo_high_perf)

# Showcase: reverse-mode autograd, a tiny MLP trained with the fused losses.
add_executable(matrix_pro_demo_mlp autograd_mlp_demo.cpp)
target_link_libraries(matrix_pro_demo_mlp PRIVATE matrix_pro)
matrix_pro_set_warnings(matrix_pro_demo_mlp)
