// Unit: dtype abstraction, fp64 reference reductions and mixed-precision GEMM.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "matrix_pro/core/dtype.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/precision.hpp"

#include "test_support.hpp"

using matrix_pro::Matrix;
using matrix_pro::test::check;
using matrix_pro::test::check_near;

int main() {
    try {
        matrix_pro::test::section("dtype_metadata");
        {
            check(matrix_pro::dtype_size(matrix_pro::DType::f32) == 4 &&
                  matrix_pro::dtype_size(matrix_pro::DType::f64) == 8 &&
                  matrix_pro::dtype_size(matrix_pro::DType::f16) == 2 &&
                  matrix_pro::dtype_size(matrix_pro::DType::bf16) == 2,
                  "dtype_size returns the byte widths");
            check(std::string(matrix_pro::dtype_name(matrix_pro::DType::f64)) == "f64",
                  "dtype_name returns the readable name");
        }

        matrix_pro::test::section("typed_buffer");
        {
            matrix_pro::TypedBuffer<double> buffer(16);
            check(buffer.size() == 16 && buffer.get() != nullptr, "TypedBuffer allocates the requested count");
            buffer = matrix_pro::TypedBuffer<double>(8);   // move assign frees the old block
            check(buffer.size() == 8, "move assignment transfers ownership");
            matrix_pro::TypedBuffer<double> empty;
            check(empty.empty(), "a default buffer is empty");
        }

        matrix_pro::test::section("fp64_reductions");
        {
            // 1e8 + 1e-4 loses the small term in fp32 accumulation, survives in fp64.
            Matrix m(2, 1, std::vector<float>{1.0e8f, 1.0e-4f});
            const double sum = matrix_pro::sum_f64(m);
            const double mean = matrix_pro::mean_f64(m);
            check_near(sum, 1.0e8 + 1.0e-4, 1e-3, "sum_f64 keeps the small term");
            check_near(mean, (1.0e8 + 1.0e-4) / 2.0, 1e-3, "mean_f64 divides the exact sum");

            Matrix u{{3.0f, 4.0f}};   // l2 norm 5
            check_near(matrix_pro::l2_norm_f64(u), 5.0, 1e-5, "l2_norm_f64 computes the euclidean norm");
        }

        matrix_pro::test::section("matmul_half");
        {
            Matrix a{{1.0f, 2.0f},
                     {3.0f, 4.0f}};
            Matrix b{{5.0f, 6.0f},
                     {7.0f, 8.0f}};
            Matrix c = matrix_pro::matmul_half(a, b);
            c.download();
            check(c.rows() == 2 && c.cols() == 2, "matmul_half keeps the GEMM shape");
            // fp16 rounds inputs to ~3 decimal digits; allow a loose tolerance.
            check(std::abs(c.at(0, 0) - 19.0f) < 1e-1f && std::abs(c.at(1, 1) - 50.0f) < 1e-1f,
                  "fp16 GEMM matches the reference within half precision");
        }

        matrix_pro::test::section("matmul_double_accumulate");
        {
            Matrix a{{1.0f, 2.0f},
                     {3.0f, 4.0f}};
            Matrix b{{5.0f, 6.0f},
                     {7.0f, 8.0f}};
            Matrix c = matrix_pro::matmul_double_accumulate(a, b);
            c.download();
            check(std::abs(c.at(0, 0) - 19.0f) < 1e-3f && std::abs(c.at(0, 1) - 22.0f) < 1e-3f &&
                  std::abs(c.at(1, 0) - 43.0f) < 1e-3f && std::abs(c.at(1, 1) - 50.0f) < 1e-3f,
                  "fp64 accumulation matches the exact reference");
        }

        matrix_pro::test::section("fp16_pack_roundtrip");
        {
            Matrix m(1, 4, std::vector<float>{1.0f, -0.5f, 65504.0f, 0.25f});
            matrix_pro::TypedBuffer<std::uint16_t> packed(4);
            matrix_pro::pack_f16(m.device_data(), packed.get(), 4);

            Matrix out(1, 4);
            matrix_pro::unpack_f16(packed.get(), out.device_data(), 4);
            out.download();
            check(std::abs(out.at(0, 0) - 1.0f) < 1e-3f && std::abs(out.at(0, 1) + 0.5f) < 1e-3f,
                  "pack/unpack round-trips representable values");
            check(std::abs(out.at(0, 2) - 65504.0f) < 1.0f, "the fp16 maximum survives the round-trip");
            check(std::abs(out.at(0, 3) - 0.25f) < 1e-3f, "exact powers of two survive exactly");
        }
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("DTYPE_PRECISION");
}
