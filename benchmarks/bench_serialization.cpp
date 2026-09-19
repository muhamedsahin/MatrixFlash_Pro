#include <cstdio>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_support.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/core/tensor.hpp"
#include "matrix_pro/core/serialization.hpp"

namespace matrix_pro {
namespace bench {
namespace {

void run(const Options& options, std::vector<Result>& out) {
    std::vector<std::size_t> sizes = options.sizes.empty() ?
        std::vector<std::size_t>{1024, 2048, 4096} : options.sizes;

    const std::string tmp_file = "bench_tmp_ckpt.mfc";

    for (std::size_t n : sizes) {
        Matrix w = Matrix::ones(n, n);
        std::size_t bytes = n * n * sizeof(float);

        Checkpoint ckpt;
        ckpt.add("weights", w);

        const Stats ms_save = sample_ms([&] {
            ckpt.save(tmp_file);
        }, options);
        add_result(out, "serialization", std::to_string(n) + "x" + std::to_string(n) + " save",
                   n, ms_save, to_gbps(bytes, ms_save.median), "GB/s", "checkpoint save");

        const Stats ms_load = sample_ms([&] {
            Checkpoint loaded = Checkpoint::load(tmp_file);
        }, options);
        add_result(out, "serialization", std::to_string(n) + "x" + std::to_string(n) + " load",
                   n, ms_load, to_gbps(bytes, ms_load.median), "GB/s", "checkpoint load");

        std::remove(tmp_file.c_str());
    }
}

} // namespace

const BenchmarkInfo& serialization_benchmark() {
    static const BenchmarkInfo info{"serialization",
                                    "Checkpoint serialization I/O throughput in GB/s",
                                    run};
    return info;
}

} // namespace bench
} // namespace matrix_pro

