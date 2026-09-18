#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include "matrix_pro/core/matrix.hpp"
#include "matrix_pro/ops/operations.hpp"

#include "test_support.hpp"
using matrix_pro::test::check;

using matrix_pro::Matrix;

static bool approx(const std::vector<float>& a, const std::vector<float>& b, float tol) {
	if (a.size() != b.size()) return false;
	for (std::size_t i = 0; i < a.size(); ++i) {
		if (std::abs(a[i] - b[i]) > tol) return false;
	}
	return true;
}

int main() {
	try {
		// --- 1.1 solve(A, b) -> A*x = b ---
		{
			Matrix a{{3.0f, 1.0f}, {1.0f, 2.0f}};
			Matrix b{{9.0f}, {8.0f}};
			Matrix x = a.solve(b);
			x.download();
			check(std::abs(x.at(0, 0) - 2.0f) < 1e-4f, "solve x0");
			check(std::abs(x.at(1, 0) - 3.0f) < 1e-4f, "solve x1");

			Matrix id = Matrix::identity(2);
			Matrix xm = a.solve(id);
			xm.download();
			Matrix inv = a.inverse(); inv.download();
			check(approx(xm.data(), inv.data(), 1e-3f), "solve multi-RHS equals inverse");
		}

		// --- 1.2 qr() -> Q * R = A, Q^T Q = I ---
		{
			Matrix a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 10.0f}};
			auto [q, r] = a.qr();
			Matrix re = q * r; re.download();
			Matrix da = a; da.download();
			check(approx(re.data(), da.data(), 1e-2f), "qr reconstruct A");

			Matrix qt = q.transpose();
			Matrix identity = qt * q; identity.download();
			Matrix expected = Matrix::identity(3);
			check(approx(identity.data(), expected.data(), 1e-3f), "qr Q^T Q = I");
		}

		// --- 1.3 svd() -> U*S*V^T = A (rectangular) ---
		{
			Matrix a{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};   // 3x2
			auto res = a.svd();
			check(res.u.rows() == 3 && res.u.cols() == 3, "svd U shape");
			check(res.v.rows() == 2 && res.v.cols() == 2, "svd V shape");
			res.s.download();
			check(res.s.rows() == 2 && res.s.cols() == 2, "svd S shape");

			const std::size_t m = a.rows(), n = a.cols();
			std::vector<float> d(m * n, 0.0f);
			for (std::size_t i = 0; i < std::min(m, n); ++i) d[i * n + i] = res.s.data()[i * 2 + i];
			Matrix sigma(m, n, d);

			Matrix re = res.u * sigma;
			Matrix re2 = re * res.v.transpose();
			re2.download();
			Matrix da = a; da.download();
			check(approx(re2.data(), da.data(), 1e-2f), "svd reconstruct A");
		}

		// --- 1.4 cholesky() -> L * L^T = A ---
		{
			Matrix a{{4.0f, 1.0f}, {1.0f, 3.0f}};
			Matrix l = a.cholesky(); l.download();
			check(std::abs(l.at(0, 0) - 2.0f) < 1e-4f, "cholesky L00");
			Matrix re = l * l.transpose(); re.download();
			Matrix da = a; da.download();
			check(approx(re.data(), da.data(), 1e-4f), "cholesky reconstruct L*L^T");
		}

		// --- 1.5 eigen() -> A = V * D * V^T (symmetric) ---
		{
			Matrix a{{2.0f, 1.0f}, {1.0f, 2.0f}};
			auto res = a.eigen();
			res.eigenvalues.download();
			check(std::abs(res.eigenvalues.data()[0] - 1.0f) < 1e-2f, "eigenvalue 1");
			check(std::abs(res.eigenvalues.data()[1 * 2 + 1] - 3.0f) < 1e-2f, "eigenvalue 3");
			Matrix re = res.eigenvectors * res.eigenvalues;
			Matrix re2 = re * res.eigenvectors.transpose();
			re2.download();
			Matrix da = a; da.download();
			check(approx(re2.data(), da.data(), 1e-2f), "eigen reconstruct A");
		}

		// --- 1.6 pinv() ---
		{
			Matrix a{{3.0f, 1.0f}, {1.0f, 2.0f}};
			Matrix p = a.pinv();
			Matrix re = a * p;
			Matrix re2 = re * a;
			re2.download();
			Matrix da = a; da.download();
			check(approx(re2.data(), da.data(), 1e-2f), "pinv A*pinv(A)*A = A");
		}

		// --- 1.7 rank() ---
		{
			check(Matrix::identity(5).rank() == 5, "rank identity = 5");
			check(Matrix::ones(4, 4).rank() == 1, "rank all-ones = 1");
			Matrix diag{{1.0f, 0.0f}, {0.0f, 0.0f}};
			check(diag.rank() == 1, "rank diag(1,0) = 1");
		}

		// --- 1.8 solve_least_squares() (over-determined) ---
		{
			Matrix a{{0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 1.0f}};
			Matrix b{{3.0f}, {5.0f}, {7.0f}};
			Matrix x = a.solve_least_squares(b);
			x.download();
			check(std::abs(x.at(0, 0) - 2.0f) < 1e-2f, "least_squares slope");
			check(std::abs(x.at(1, 0) - 3.0f) < 1e-2f, "least_squares intercept");
		}

		// --- 2.1 advanced numerical utilities ---
		{
			Matrix a{{3.0f, 4.0f}, {0.0f, 0.0f}};
			check(std::abs(a.frobenius_norm() - 5.0f) < 1e-4f, "frobenius norm");
			check(std::abs(Matrix::identity(2).condition_number() - 1.0f) < 1e-4f, "identity condition number");
		}

		{
			Matrix x{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
			Matrix cov = x.covariance();
			cov.download();
			check(cov.rows() == 2 && cov.cols() == 2, "covariance shape");
			check(std::abs(cov.at(0, 0) - 4.0f) < 1e-2f, "covariance diagonal 1");
			check(std::abs(cov.at(1, 1) - 4.0f) < 1e-2f, "covariance diagonal 2");
		}

		{
			Matrix x{{1.0f, 2.0f}, {2.0f, 5.0f}};
			Matrix corr = x.correlation();
			corr.download();
			check(std::abs(corr.at(0, 0) - 1.0f) < 1e-3f, "correlation diagonal");
			check(std::abs(corr.at(1, 1) - 1.0f) < 1e-3f, "correlation diagonal 2");
			check(std::abs(corr.at(0, 1) - 1.0f) < 1e-3f, "correlation off-diagonal");
		}

		return matrix_pro::test::summary("LINALG");
	} catch (const std::exception& e) {
		std::cerr << "EXCEPTION: " << e.what() << "\n";
		return 99;
	}
}