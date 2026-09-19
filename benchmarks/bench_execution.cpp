#include <cstddef>
#include <iostream>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/streams/execution.hpp"

namespace matrix_pro {
namespace bench {
namespace {

void run(const Options& options, std::vector<Result>& out) {
    std::vector<std::size_t> sizes = options.sizes.empty() ?
        std::vector<std::size_t>{256, 512, 1024} : options.sizes;

    for (std::size_t n : sizes) {
        Matrix a = Matrix::ones(n, n);
        Matrix b = Matrix::ones(n, n);
        Matrix c(n, n);

        // Standard repeated launch
        const Stats ms_standard = sample_ms([&] {
            c = a + b;
        }, options);
        add_result(out, "execution", std::to_string(n) + "x" + std::to_string(n) + " direct",
                   n, ms_standard, 0.0, "ms", "standard kernel launch");

        // CUDA Graph captured launch
        CudaGraph graph;
        c = a + b;
        c.synchronize();

        graph.begin_capture();
        c = a + b;
        graph.end_capture();

        const Stats ms_graph = sample_ms([&] {
            graph.replay();
        }, options);
        double speedup = ms_standard.median / std::max(ms_graph.median, 1e-6);
        add_result(out, "execution", std::to_string(n) + "x" + std::to_string(n) + " graph",
                   n, ms_graph, speedup, "x speedup", "CUDA Graph replay");
    }
}

} // namespace

const BenchmarkInfo& execution_benchmark() {
    static const BenchmarkInfo info{"execution",
                                    "CUDA Graph vs direct launch latency and overhead",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro

