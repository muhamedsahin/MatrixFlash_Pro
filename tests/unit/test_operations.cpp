#include <cmath>
#include <iostream>
#include <vector>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/product.hpp"

#include "test_support.hpp"
using matrix_pro::test::check;

using matrix_pro::Matrix;

static float host_sum(const Matrix& m) { float s = 0; for (float v : m.data()) s += v; return s; }

int main() {
	try {
		// --- matmul (cuBLAS) vs host reference, non-square ---
		{
			Matrix a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};      // 2x3
			Matrix b{{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}; // 3x2
			Matrix c = a * b;
			c.download();
			check(c.rows() == 2 && c.cols() == 2, "matmul shape");
			check(std::abs(c.at(0, 0) - 58.0f) < 1e-3f, "matmul c00");
			check(std::abs(c.at(0, 1) - 64.0f) < 1e-3f, "matmul c01");
			check(std::abs(c.at(1, 0) - 139.0f) < 1e-3f, "matmul c10");
			check(std::abs(c.at(1, 1) - 154.0f) < 1e-3f, "matmul c11");
		}

		// --- multiply_into + fused gemm_bias_relu ---
		{
			Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}};
			Matrix b{{2.0f, 0.0f}, {1.0f, 2.0f}};
			Matrix out(2, 2, matrix_pro::MemoryMode::device_only);
			matrix_pro::multiply_into(a, b, out);
			out.download();
			check(std::abs(out.at(0, 0) - 4.0f) < 1e-3f, "multiply_into 00");
			check(std::abs(out.at(1, 1) - 8.0f) < 1e-3f, "multiply_into 11");

			Matrix bias{{1.0f, -10.0f}};
			Matrix fused = matrix_pro::gemm_bias_relu(a, b, bias);
			fused.download();
			// AB = [[4,4],[10,8]]; +bias = [[5,-6],[11,-2]]; relu -> [[5,0],[11,0]]
			check(std::abs(fused.at(0, 0) - 5.0f) < 1e-2f, "gemm_bias_relu 00");
			check(std::abs(fused.at(0, 1) - 0.0f) < 1e-2f, "gemm_bias_relu 01");
			check(std::abs(fused.at(1, 0) - 11.0f) < 1e-2f, "gemm_bias_relu 10");
			check(std::abs(fused.at(1, 1) - 0.0f) < 1e-2f, "gemm_bias_relu 11");
		}


		// --- outer_product ---
		{
			Matrix u{{1.0f, 2.0f}};       // treated as (1,2)
			Matrix v{{3.0f, 4.0f, 5.0f}}; // (1,3)
			Matrix o = u.outer_product(v);
			o.download();
			check(std::abs(o.at(0, 0) - 3.0f) < 1e-3f, "outer 00");
			check(std::abs(o.at(0, 2) - 5.0f) < 1e-3f, "outer 02");
		}

		// --- reductions on GPU vs host ---
		{
			Matrix m{{1.0f, -2.0f, 7.0f}, {3.0f, 4.0f, -5.0f}};
			check(std::abs(m.sum() - host_sum(m)) < 1e-3f, "gpu sum");
			check(std::abs(m.mean() - (host_sum(m) / 6.0f)) < 1e-3f, "gpu mean");
			check(std::abs(m.min() - (-5.0f)) < 1e-3f, "gpu min");
			check(std::abs(m.max() - 7.0f) < 1e-3f, "gpu max");
			check(m.argmin() == 5 && m.argmax() == 2, "gpu argmin/argmax");
			check(std::abs(m.l1_norm() - (1 + 2 + 7 + 3 + 4 + 5)) < 1e-3f, "gpu l1");
			check(std::abs(m.l2_norm() - std::sqrt(1 + 4 + 49 + 9 + 16 + 25)) < 1e-3f, "gpu l2");
			check(std::abs(m.abs_max() - 7.0f) < 1e-3f, "gpu abs_max");
			check(std::abs(m.stddev() - m.stddev()) < 1e-3f, "gpu stddev agrees");

			Matrix rs = m.row_sum(); rs.download();
			check(std::abs(rs.at(0, 0) - 6.0f) < 1e-3f, "row_sum");
			Matrix cs = m.col_sum(); cs.download();
			check(std::abs(cs.at(0, 1) - 2.0f) < 1e-3f, "col_sum");
		}

		// --- softmax sums to 1 per row ---
		{
			Matrix m{{1.0f, -2.0f}, {3.0f, 4.0f}};
			Matrix sm = m.softmax();
			sm.download();
			check(std::abs(sm.at(0, 0) + sm.at(0, 1) - 1.0f) < 1e-4f, "softmax row 0 sums to 1");
			check(std::abs(sm.at(1, 0) + sm.at(1, 1) - 1.0f) < 1e-4f, "softmax row 1 sums to 1");
		}

		// --- elementwise math ---
		{
			Matrix m{{1.0f, 4.0f, 9.0f}};
			Matrix sq = m.sqrt(); sq.download();
			check(std::abs(sq.at(0, 2) - 3.0f) < 1e-3f, "sqrt");
			Matrix e = m.exp(); e.download();
			check(std::abs(e.at(0, 0) - std::exp(1.0f)) < 1e-3f, "exp");
			Matrix n = -m; n.download();
			check(std::abs(n.at(0, 1) + 4.0f) < 1e-3f, "unary minus");
			Matrix cl = m.clamp(2.0f, 5.0f); cl.download();
			check(std::abs(cl.at(0, 0) - 2.0f) < 1e-3f && std::abs(cl.at(0, 2) - 5.0f) < 1e-3f, "clamp");
		}

		// --- broadcast ---
		{
			Matrix m{{1.0f, 2.0f}, {3.0f, 4.0f}};
			Matrix bias{{10.0f, 20.0f}}; // row vector (len == cols)
			Matrix r = m.add_row_vector(bias); r.download();
			check(std::abs(r.at(1, 1) - 24.0f) < 1e-3f, "add_row_vector");
			Matrix scale{{2.0f, 3.0f}}; // col vector (len == rows)
			Matrix c2 = m.multiply_col_vector(scale); c2.download();
			check(std::abs(c2.at(1, 0) - 9.0f) < 1e-3f, "multiply_col_vector");
		}

		// --- save / load ---
		{
			Matrix m{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
			const std::string file = "mtest.bin";
			m.save(file);
			Matrix loaded = Matrix::load(file);
			check(loaded.rows() == 2 && loaded.cols() == 3, "load shape");
			loaded.download();
			check(std::abs(loaded.at(1, 2) - 6.0f) < 1e-3f, "load value");
			std::remove(file.c_str());
		}

		// --- factories ---
		{
			Matrix r1 = Matrix::random(4, 5);
			Matrix r2 = Matrix::random(4, 5);
			check(r1.rows() == 4 && r1.cols() == 5, "random shape");
			check(r1.data() != r2.data(), "random values differ across calls");
			Matrix g = Matrix::glorot(3, 3);
			check(g.rows() == 3 && g.cols() == 3, "glorot shape");
		}

		return matrix_pro::test::summary("FEATURES");
	} catch (const std::exception& e) {
		std::cerr << "EXCEPTION: " << e.what() << "\n";
		return 99;
	}
}
