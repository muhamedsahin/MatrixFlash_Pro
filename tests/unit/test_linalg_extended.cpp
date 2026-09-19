#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <cmath>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("LU decomposition: PA = LU");
        {
            Matrix A(3, 3, {
                2.0f, -1.0f, -2.0f,
                -4.0f, 6.0f, 3.0f,
                -4.0f, -2.0f, 8.0f
            });

            LUResult res = lu(A);
            res.L.download();
            res.U.download();
            res.P.download();

            // Check L is lower triangular with 1 on diag
            for (std::size_t r = 0; r < 3; ++r) {
                ctx.check_near(res.L.at(r, r), 1.0f, 1e-4, "L diagonal is 1");
                for (std::size_t c = r + 1; c < 3; ++c) {
                    ctx.check_near(res.L.at(r, c), 0.0f, 1e-4, "L upper part is 0");
                }
            }

            // Check U is upper triangular
            for (std::size_t r = 0; r < 3; ++r) {
                for (std::size_t c = 0; c < r; ++c) {
                    ctx.check_near(res.U.at(r, c), 0.0f, 1e-4, "U lower part is 0");
                }
            }

            // Verify P * A == L * U
            Matrix PA = res.P * A;
            Matrix LU = res.L * res.U;
            PA.download();
            LU.download();
            for (std::size_t i = 0; i < 9; ++i) {
                ctx.check_near(PA.data()[i], LU.data()[i], 1e-3, "PA matches LU");
            }
        }

        ctx.section("trsm: triangular solve");
        {
            // Upper triangular A
            Matrix A(2, 2, {
                2.0f, 1.0f,
                0.0f, 3.0f
            });
            Matrix B(2, 1, {
                5.0f,
                6.0f
            });
            // Solution to A*X = B: 3*x2 = 6 => x2 = 2. 2*x1 + 2 = 5 => x1 = 1.5.
            Matrix X = trsm(A, B, true, true, false);
            X.download();
            ctx.check_near(X.at(0, 0), 1.5f, 1e-4, "trsm X[0,0] = 1.5");
            ctx.check_near(X.at(1, 0), 2.0f, 1e-4, "trsm X[1,0] = 2.0");
        }

        ctx.section("matrix_power: A^n");
        {
            Matrix A(2, 2, {
                1.0f, 1.0f,
                0.0f, 1.0f
            });
            Matrix A3 = matrix_power(A, 3);
            A3.download();
            // [[1, 1], [0, 1]]^3 = [[1, 3], [0, 1]]
            ctx.check_near(A3.at(0, 0), 1.0f, 1e-4, "A^3 [0,0]");
            ctx.check_near(A3.at(0, 1), 3.0f, 1e-4, "A^3 [0,1]");
            ctx.check_near(A3.at(1, 0), 0.0f, 1e-4, "A^3 [1,0]");
            ctx.check_near(A3.at(1, 1), 1.0f, 1e-4, "A^3 [1,1]");

            Matrix A0 = matrix_power(A, 0);
            A0.download();
            ctx.check_near(A0.at(0, 0), 1.0f, 1e-4, "A^0 identity [0,0]");
            ctx.check_near(A0.at(0, 1), 0.0f, 1e-4, "A^0 identity [0,1]");
        }

        ctx.section("matrix_exp: exp(0) = I");
        {
            Matrix Z = Matrix::zeros(2, 2);
            Matrix expZ = matrix_exp(Z);
            expZ.download();
            ctx.check_near(expZ.at(0, 0), 1.0f, 1e-4, "exp(0)[0,0] = 1");
            ctx.check_near(expZ.at(1, 1), 1.0f, 1e-4, "exp(0)[1,1] = 1");
            ctx.check_near(expZ.at(0, 1), 0.0f, 1e-4, "exp(0)[0,1] = 0");
        }

        ctx.section("log_determinant");
        {
            Matrix A(2, 2, {
                2.0f, 0.0f,
                0.0f, 3.0f
            });
            float ldet = log_determinant(A);
            ctx.check_near(ldet, std::log(6.0f), 1e-4, "log_det(diag(2,3)) = log(6)");
        }

        return ctx.summary("LINALG_EXTENDED");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

