#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/ops/operations.hpp"
#include "matrix_pro/ops/broadcast.hpp"

#include <cstdio>
#include <vector>

using namespace matrix_pro;

static void print_mat(const char* tag, Matrix m) {
    m.download();
    for (std::size_t i = 0; i < m.size(); ++i)
        std::printf("[%s %zu] %.6f\n", tag, i, m.data()[i]);
}

static Matrix vgrad(const Variable& v) {
    Matrix g = v.grad();
    g.download();
    return g;
}

int main() {
    const std::vector<float> lg{1, 2, 0.1f, 0.2f, 0, 3};
    const std::vector<float> tg{1, 0, 0, 0, 0, 1};
    const Matrix oh(2, 3, tg);

    {
        Variable x(Matrix(2, 3, lg), true);
        Variable target(oh, false);
        Variable loss = mse_loss(softmax_cross_entropy_loss(x, target),
                                 Variable(Matrix(1, 1, std::vector<float>{0}), false));
        loss.backward();
        print_mat("full-ce-mse", vgrad(x));
    }
    {
        Variable x(Matrix(2, 3, lg), true);
        softmax_cross_entropy_loss(x, Variable(oh, false)).backward();
        print_mat("ce-only", vgrad(x));
    }
    {
        Matrix b = softmax_cross_entropy_grad(Matrix(2, 3, lg), oh);
        print_mat("raw-ce-grad", b);
        Variable bv(b, true);
        mse_loss(bv, Variable(Matrix(2, 3, std::vector<float>{0, 0, 0, 0, 0, 0}), false)).backward();
        print_mat("mse-of-b", vgrad(bv));
    }
    {
        Variable raw(Matrix(2, 3, std::vector<float>{-0.378784f, 0.329501f, 0.049283f,
                                                     0.027377f, 0.022415f, -0.049792f}), true);
        Variable up(Matrix(1, 1, std::vector<float>{-1.0f}), false);
        Variable acc = raw.broadcast_multiply(up);
        acc.backward();
        print_mat("var-bmul", vgrad(raw));
    }
    return 0;
}
