#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <vector>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("SparseCOO from triplets & to_dense");
        {
            std::vector<int> rows = {0, 1, 2};
            std::vector<int> cols = {0, 2, 1};
            std::vector<float> vals = {5.0f, 3.0f, 7.0f};

            SparseCOO coo = SparseCOO::from_triplets(3, 3, rows, cols, vals);
            ctx.check(coo.rows() == 3, "coo rows = 3");
            ctx.check(coo.cols() == 3, "coo cols = 3");
            ctx.check(coo.nnz() == 3, "coo nnz = 3");

            Matrix dense = coo.to_dense();
            dense.download();
            ctx.check_near(dense.at(0, 0), 5.0f, 1e-5, "dense(0,0)=5");
            ctx.check_near(dense.at(1, 2), 3.0f, 1e-5, "dense(1,2)=3");
            ctx.check_near(dense.at(2, 1), 7.0f, 1e-5, "dense(2,1)=7");
            ctx.check_near(dense.at(0, 1), 0.0f, 1e-5, "dense(0,1)=0");
        }

        ctx.section("SparseCOO to SparseCSR roundtrip");
        {
            std::vector<int> rows = {0, 1, 2};
            std::vector<int> cols = {0, 2, 1};
            std::vector<float> vals = {5.0f, 3.0f, 7.0f};

            SparseCOO coo = SparseCOO::from_triplets(3, 3, rows, cols, vals);
            SparseCSR csr = coo.to_csr();
            ctx.check(csr.rows() == 3 && csr.cols() == 3, "csr dims match");
            ctx.check(csr.nnz() == 3, "csr nnz matches");

            Matrix d = csr.to_dense();
            d.download();
            ctx.check_near(d.at(0, 0), 5.0f, 1e-5, "csr roundtrip (0,0)");
            ctx.check_near(d.at(1, 2), 3.0f, 1e-5, "csr roundtrip (1,2)");
        }

        ctx.section("SparseCSC from_dense & to_dense");
        {
            Matrix m(2, 3, {
                1.0f, 0.0f, 2.0f,
                0.0f, 3.0f, 0.0f
            });
            SparseCSC csc = SparseCSC::from_dense(m);
            ctx.check(csc.rows() == 2 && csc.cols() == 3, "csc dims match");
            ctx.check(csc.nnz() == 3, "csc nnz = 3");

            Matrix reconstructed = csc.to_dense();
            reconstructed.download();
            m.download();
            for (std::size_t i = 0; i < 6; ++i) {
                ctx.check_near(reconstructed.data()[i], m.data()[i], 1e-5, "csc matches original dense");
            }
        }

        ctx.section("SparseCSC SpMV: y = A * x");
        {
            Matrix A(2, 2, {
                2.0f, 0.0f,
                1.0f, 3.0f
            });
            Matrix x(2, 1, {
                4.0f,
                5.0f
            });
            // y = [2*4 = 8, 1*4 + 3*5 = 19]
            SparseCSC csc = SparseCSC::from_dense(A);
            Matrix y = csc.spmv(x);
            y.download();
            ctx.check_near(y.at(0, 0), 8.0f, 1e-4, "csc spmv [0,0]");
            ctx.check_near(y.at(1, 0), 19.0f, 1e-4, "csc spmv [1,0]");
        }

        ctx.section("Sparse conversions: csr_to_coo and csr_to_csc");
        {
            Matrix m(3, 3, {
                1.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f,
                0.0f, 0.0f, 3.0f
            });
            SparseCSR csr = SparseCSR::from_dense(m);
            SparseCOO coo = csr_to_coo(csr);
            SparseCSC csc = csr_to_csc(csr);

            ctx.check(coo.nnz() == 3, "coo nnz = 3");
            ctx.check(csc.nnz() == 3, "csc nnz = 3");

            Matrix m_coo = coo.to_dense();
            Matrix m_csc = csc.to_dense();
            m_coo.download();
            m_csc.download();
            ctx.check_near(m_coo.at(0, 0), 1.0f, 1e-5, "coo from csr");
            ctx.check_near(m_csc.at(2, 2), 3.0f, 1e-5, "csc from csr");
        }

        return ctx.summary("SPARSE_FORMATS");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

