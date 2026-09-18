#include <cmath>
#include <iostream>

#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/core/dtype.hpp"
#include "matrix_pro/nn/fused.hpp"
#include "matrix_pro/indexing/indexing.hpp"
#include "matrix_pro/nn/inplace.hpp"
#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/rng/rng.hpp"
#include "matrix_pro/sparse/sparse.hpp"
#include "matrix_pro/streams/stream_pool.hpp"
#include "matrix_pro/view/view.hpp"

#include "test_support.hpp"
using matrix_pro::test::check;

using matrix_pro::Matrix;

int main() {
    try {
        {
            Matrix a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
            matrix_pro::MatrixView tv = matrix_pro::transpose_view(a);
            check(tv.rows() == 3 && tv.cols() == 2, "transpose_view shape");
            Matrix t = matrix_pro::materialize(tv);
            t.download();
            check(t.at(0, 1) == 4.0f && t.at(2, 0) == 3.0f, "transpose_view values");
            matrix_pro::MatrixView sv = matrix_pro::slice_view(a, 0, 1, 1, 3);
            Matrix s = matrix_pro::materialize(sv);
            s.download();
            check(s.at(0, 0) == 2.0f && s.at(0, 1) == 3.0f, "slice_view values");
        }
        {
            Matrix a{{1.0f, -2.0f}, {3.0f, -4.0f}};
            matrix_pro::relu_(a);
            a.download();
            check(a.at(0, 1) == 0.0f && a.at(1, 0) == 3.0f, "relu_ in place");
            matrix_pro::add_scalar_(a, 1.0f);
            a.download();
            check(std::abs(a.at(0, 0) - 2.0f) < 1e-5f, "add_scalar_ in place");
            Matrix bias{{10.0f, 20.0f}};
            matrix_pro::broadcast_add_(a, bias);
            a.download();
            check(std::abs(a.at(0, 0) - 12.0f) < 1e-5f, "broadcast_add_ in place");
        }
        {
            Matrix w{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
            Matrix idx(2, 1, std::vector<float>{2.0f, 0.0f});
            Matrix e = matrix_pro::embedding(w, idx);
            e.download();
            check(e.at(0, 0) == 5.0f && e.at(1, 1) == 2.0f, "embedding lookup");
            Matrix g{{1.0f, 1.0f}, {1.0f, 1.0f}};
            Matrix eb = matrix_pro::embedding_backward(g, idx, 3);
            eb.download();
            check(eb.at(0, 0) == 1.0f && eb.at(1, 0) == 0.0f && eb.at(2, 1) == 1.0f,
                  "embedding backward scatter");
        }
        {
            Matrix d{{1.0f, 0.0f}, {0.0f, 2.0f}};
            matrix_pro::SparseCSR csr = matrix_pro::SparseCSR::from_dense(d);
            check(csr.nnz() == 2, "csr nnz");
            Matrix x(2, 1, std::vector<float>{3.0f, 4.0f});
            Matrix y = matrix_pro::spmv(csr, x);
            y.download();
            check(std::abs(y.at(0, 0) - 3.0f) < 1e-4f && std::abs(y.at(1, 0) - 8.0f) < 1e-4f,
                  "spmv result");
        }
        {
            Matrix x{{0.0f, 1.0f}};
            Matrix y{{2.0f, 3.0f}};
            Matrix f = matrix_pro::fused_sigmoid_mul(x, y);
            Matrix ref = matrix_pro::elementwise_multiply(x.sigmoid(), y);
            f.download(); ref.download();
            check(std::abs(f.at(0, 0) - ref.at(0, 0)) < 1e-5f &&
                  std::abs(f.at(0, 1) - ref.at(0, 1)) < 1e-5f, "fused_sigmoid_mul");
            Matrix r = matrix_pro::fused_relu_add(x, y);
            r.download();
            check(r.at(0, 0) == 2.0f && r.at(0, 1) == 4.0f, "fused_relu_add");
        }
        {
            matrix_pro::rng_seed(42);
            Matrix n = matrix_pro::randn_gpu(4, 64);
            n.download();
            double mean = 0.0;
            for (float v : n.data()) mean += v;
            mean /= n.size();
            check(std::abs(mean) < 0.35, "randn_gpu mean near zero");
            Matrix u = matrix_pro::uniform_gpu(2, 64, -1.0f, 1.0f);
            u.download();
            bool in_range = true;
            for (float v : u.data()) in_range &= (v >= -1.0f && v <= 1.0f);
            check(in_range, "uniform_gpu range");
            Matrix dd = matrix_pro::dropout_gpu(Matrix::ones(2, 64), 0.5f);
            dd.download();
            check(!dd.all() || !dd.any(), "dropout_gpu drops some elements");
        }
        {
            Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
            double s = matrix_pro::sum_f64(a);
            check(std::abs(s - 10.0) < 1e-9, "sum_f64 exact");
            check(std::string(matrix_pro::dtype_name(matrix_pro::DType::f64)) == "f64",
                  "dtype_name");
        }
        {
            cudaStream_t s0 = matrix_pro::pool_stream(0);
            cudaStream_t s1 = matrix_pro::pool_stream(1);
            check(s0 != nullptr && s1 != nullptr && s0 != s1, "pool streams distinct");
            Matrix m{{1.0f, 5.0f, 2.0f}};
            auto* slot = static_cast<std::size_t*>(matrix_pro::pin_host(sizeof(std::size_t)));
            *slot = 999;
            matrix_pro::argmax_async(m, slot);
            matrix_pro::synchronize();
            check(*slot == 1, "argmax_async value");
            matrix_pro::unpin_host(slot);
            matrix_pro::synchronize_pool();
        }

        return matrix_pro::test::summary("NEW-FEATURES");
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        return 99;
    }
}

