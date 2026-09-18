// ============================================================================
//  Benchmark entry point
//  ---------------------------------------------------------------------------
//  One source file, several executables. MATRIX_PRO_SINGLE_BENCH is injected by
//  benchmarks/CMakeLists.txt:
//     "matmul"  -> matrix_pro_bench_matmul runs only that workload family
//     ""        -> matrix_pro_bench_all runs every family in sequence
// ============================================================================

#include "benchmark_support.hpp"

#include <iostream>

#ifndef MATRIX_PRO_SINGLE_BENCH
#define MATRIX_PRO_SINGLE_BENCH ""
#endif

int main(int argc, char** argv) {
    try {
        return matrix_pro::bench::run_main(argc, argv, MATRIX_PRO_SINGLE_BENCH, argv[0]);
    } catch (const std::exception& error) {
        std::cerr << "fatal benchmark error: " << error.what() << "\n";
        return 1;
    }
}