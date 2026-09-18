#include "matrix_pro/autograd/autograd.hpp"
#include "matrix_pro/ops/operations.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <vector>

#include "test_support.hpp"

using namespace matrix_pro;

namespace {

using matrix_pro::test::check;

float loss_value(const Variable& loss) {
    Matrix v = loss.value();
    v.download();
    return v.data()[0];
}

Matrix variable_grad(const Variable& x) {
    Matrix g = x.grad();
    g.download();
    return g;
}

Matrix numeric_grad(const std::function<Variable(const Variable&)>& build,
                    const Matrix& x0, const Variable& target, float h = 1e-2f) {
    Matrix g(x0.rows(), x0.cols());
    for (std::size_t i = 0; i < x0.size(); ++i) {
        Matrix plus = x0, minus = x0;
        plus.data()[i] += h;
        minus.data()[i] -= h;
        plus.upload();
        minus.upload();
        const float fp = loss_value(mse_loss(build(Variable(plus, false)), target));
        const float fm = loss_value(mse_loss(build(Variable(minus, false)), target));
        g.data()[i] = (fp - fm) / (2.0f * h);
    }
    return g;
}

void check_grad(const std::function<Variable(const Variable&)>& build,
                const Matrix& x0, const Matrix& t, const char* name, float tol = 5e-2f) {
    // Ops such as flatten/reshape/slice/transpose change the output shape, so the
    // reference target must match that shape. It is captured once from the
    // initial output (shifted by 0.5 so the loss derivative stays non-trivial)
    // which keeps analytic and numeric gradients on the exact same scalar loss.
    Variable probe(x0, false);
    Matrix reference = build(probe).value();
    reference.download();
    const Matrix* target_matrix = &t;
    Matrix derived_target;
    if (reference.rows() != t.rows() || reference.cols() != t.cols()) {
        derived_target = reference.add_scalar(0.5f);
        target_matrix = &derived_target;
    }

    Variable x(x0, true);
    Variable target(*target_matrix, false);
    Variable loss = mse_loss(build(x), target);
    loss.backward();
    Matrix analytic = variable_grad(x);
    Matrix numeric = numeric_grad(build, x0, target);
    bool ok = analytic.rows() == numeric.rows() && analytic.cols() == numeric.cols();
    if (ok) {
        for (std::size_t i = 0; i < analytic.size() && ok; ++i)
            ok = std::fabs(analytic.data()[i] - numeric.data()[i]) < tol;
    }
    check(ok, name);
}

void variable_tests() {
    {
        Variable a(Matrix(2, 2, std::vector<float>{1, 2, 3, 4}), true);
        Variable b(Matrix(2, 2, std::vector<float>{5, 6, 7, 8}), true);
        a.subtract(b).backward();
        check(variable_grad(a).data()[0] == 1.0f && variable_grad(b).data()[0] == -1.0f, "subtract grad");
        a.zero_grad();
        b.zero_grad();
        a.elementwise_multiply(b).backward();
        Matrix ga = variable_grad(a);
        check(std::fabs(ga.data()[0] - 5) < 1e-5f && std::fabs(ga.data()[3] - 8) < 1e-5f, "elementwise mul grad");
        a.zero_grad();
        b.zero_grad();
        a.divide(b).backward();
        check(std::fabs(variable_grad(a).data()[1] - 1.0f / 6.0f) < 1e-5f, "divide grad a");
        check(std::fabs(variable_grad(b).data()[1] + 2.0f / 36.0f) < 1e-5f, "divide grad b");
    }
    {
        Variable a(Matrix(2, 1, std::vector<float>{2, -3}), true);
        a.pow(3.0f).backward();
        check(std::fabs(variable_grad(a).data()[0] - 12.0f) < 1e-3f &&
              std::fabs(variable_grad(a).data()[1] - 27.0f) < 1e-3f, "pow grad");
    }
    const Matrix x0(2, 2, std::vector<float>{0.3f, -0.7f, 1.2f, 0.5f});
    const Matrix t(2, 2, std::vector<float>{0.2f, 0.1f, -0.3f, 0.8f});
    check_grad([](const Variable& in) { return in.sigmoid(); }, x0, t, "sigmoid grad");
    check_grad([](const Variable& in) { return in.tanh(); }, x0, t, "tanh grad");
    check_grad([](const Variable& in) { return in.elu(); }, x0, t, "elu grad");
    check_grad([](const Variable& in) { return in.softplus(); }, x0, t, "softplus grad");
    check_grad([](const Variable& in) { return in.mish(); }, x0, t, "mish grad");
    check_grad([](const Variable& in) { return in.hardtanh(); }, x0, t, "hardtanh grad");
    check_grad([](const Variable& in) { return in.hardsigmoid(); }, x0, t, "hardsigmoid grad");
    check_grad([](const Variable& in) { return in.hardswish(); }, x0, t, "hardswish grad");
    check_grad([](const Variable& in) { return in.selu(); }, x0, t, "selu grad");
    check_grad([](const Variable& in) { return in.multiply(2.0f).add_scalar(1.0f).negate(); }, x0, t, "scalar chain grad");
    check_grad([](const Variable& in) { return in.transpose(); }, x0, t, "transpose grad");
    check_grad([](const Variable& in) { return in.flatten(); }, x0, t, "flatten grad");
    check_grad([](const Variable& in) { return in.reshape(4, 1); }, x0, t, "reshape grad");
    check_grad([](const Variable& in) { return in.slice(0, 1, 0, 2); }, x0, t, "slice grad");
    {
        Variable v(Matrix(1, 2, std::vector<float>{0.5f, -1.0f}), true);
        check_grad([&](const Variable& in) { return in.add_row_vector(v); },
                   Matrix(2, 2, std::vector<float>{0.1f, 0.9f, 1.1f, -2.0f}), t, "add_row_vector grad");
        check_grad([&](const Variable& in) { return in.multiply_row_vector(v); },
                   Matrix(2, 2, std::vector<float>{0.1f, 0.9f, 1.1f, -2.0f}), t, "multiply_row_vector grad");
    }
    {
        // General broadcasting: (2,2) with (1,2) / (2,1) operands.
        check_grad([](const Variable& in) {
            return in.broadcast_add(Variable(Matrix(1, 2, std::vector<float>{0.5f, -1.0f}), false));
        }, x0, t, "broadcast_add grad");
        check_grad([](const Variable& in) {
            return in.broadcast_multiply(Variable(Matrix(2, 1, std::vector<float>{1.5f, -0.5f}), false));
        }, x0, t, "broadcast_multiply grad");
    }
    {
        // concat: every operand receives the matching slice of the upstream grad.
        Variable a(Matrix(2, 1, std::vector<float>{1.0f, 2.0f}), true);
        Variable b(Matrix(2, 1, std::vector<float>{3.0f, 4.0f}), true);
        Variable::concat({a, b}, 0).backward();
        Matrix ga = variable_grad(a), gb = variable_grad(b);
        check(ga.rows() == 2 && ga.cols() == 1 && ga.data()[0] == 1.0f && ga.data()[1] == 1.0f, "concat axis0 grad a");
        check(gb.rows() == 2 && gb.cols() == 1 && gb.data()[0] == 1.0f && gb.data()[1] == 1.0f, "concat axis0 grad b");

        Variable c(Matrix(1, 2, std::vector<float>{1.0f, 2.0f}), true);
        Variable d(Matrix(1, 3, std::vector<float>{3.0f, 4.0f, 5.0f}), true);
        Variable::concat({c, d}, 1).backward();
        Matrix gc = variable_grad(c), gd = variable_grad(d);
        check(gc.rows() == 1 && gc.cols() == 2 && gc.data()[0] == 1.0f && gc.data()[1] == 1.0f, "concat axis1 grad c");
        check(gd.rows() == 1 && gd.cols() == 3 && gd.data()[2] == 1.0f, "concat axis1 grad d");
    }
    {
        const Matrix px(1, 4, std::vector<float>{-1.0f, 0.5f, -0.2f, 2.0f});
        const Matrix pt(1, 4, std::vector<float>{0.1f, -0.2f, 0.3f, 0.0f});
        Variable alpha(Matrix(1, 1, std::vector<float>{0.25f}), true);
        Variable x(px, true);
        mse_loss(x.prelu(alpha), Variable(pt, false)).backward();
        // y = x for x > 0 and y = alpha * x otherwise, so dL/dalpha is
        // sum over the non-positive side of x * 2 * (alpha * x - t) / N.
        float da = 0.0f;
        for (std::size_t i = 0; i < 4; ++i) {
            const float xi = px.data()[i];
            if (xi <= 0.0f) da += xi * 2.0f * (0.25f * xi - pt.data()[i]) / 4.0f;
        }
        check(std::fabs(variable_grad(alpha).data()[0] - da) < 5e-2f, "prelu alpha grad");
        check_grad([&](const Variable& in) { return in.prelu(Variable(Matrix(1, 1, std::vector<float>{0.25f}), false)); },
                   px, pt, "prelu input grad");
    }
    {
        const Matrix cx(2, 2, std::vector<float>{1.0f, -2.0f, 3.0f, 0.5f});
        check_grad(
            [](const Variable& in) {
                return in.custom_unary("square",
                    [](const Matrix& m) { return matrix_pro::elementwise_multiply(m, m); },
                    [](const Matrix& g, const Matrix& in_, const Matrix&) {
                        return matrix_pro::elementwise_multiply(g, in_) * 2.0f;
                    });
            },
            cx, t, "custom_unary grad");
    }
}

void loss_tests() {
    Matrix p(1, 3, std::vector<float>{1.0f, -2.0f, 0.5f});
    Matrix t(1, 3, std::vector<float>{0.5f, 0.5f, 0.5f});
    check_grad([&](const Variable& in) { return mse_loss(in, Variable(t, false)); }, p, t, "mse grad");
    check_grad([&](const Variable& in) { return mae_loss(in, Variable(t, false)); }, p, t, "mae grad");
    check_grad([&](const Variable& in) { return huber_loss(in, Variable(t, false), 0.7f); }, p, t, "huber grad");
    check_grad([&](const Variable& in) { return bce_with_logits_loss(in, Variable(t, false)); }, p, t, "bce grad");
    {
        Variable pv(p, true);
        bce_with_logits_loss(pv, Variable(t, false)).backward();
        float expected0 = (1.0f / (1.0f + std::exp(-1.0f)) - 0.5f) / 3.0f;
        check(std::fabs(variable_grad(pv).data()[0] - expected0) < 1e-4f, "bce analytic grad");
    }
    {
        // softmax CE is itself a loss: compare analytic grad of the reduced
        // scalar loss against finite differences of the same scalar loss.
        Matrix logits(2, 3, std::vector<float>{1.0f, 2.0f, 0.1f, 0.2f, 0.0f, 3.0f});
        Matrix oh(2, 3, std::vector<float>{1, 0, 0, 0, 0, 1});
        const float h = 1e-2f;
        Variable xl(logits, true);
        Variable loss = softmax_cross_entropy_loss(xl, Variable(oh, false));
        loss.backward();
        Matrix analytic = variable_grad(xl);
        Matrix numeric(logits.rows(), logits.cols());
        auto scalar_ce = [&](const Matrix& m) {
            Variable v(m, false);
            Matrix l = softmax_cross_entropy_loss(v, Variable(oh, false)).value();
            l.download();
            float total = 0.0f;
            for (std::size_t i = 0; i < l.size(); ++i) total += l.data()[i];
            return total / static_cast<float>(l.size());
        };
        for (std::size_t i = 0; i < logits.size(); ++i) {
            Matrix plus = logits, minus = logits;
            plus.data()[i] += h;
            minus.data()[i] -= h;
            plus.upload();
            minus.upload();
            numeric.data()[i] = (scalar_ce(plus) - scalar_ce(minus)) / (2.0f * h);
        }
        bool ok = true;
        for (std::size_t i = 0; i < analytic.size() && ok; ++i)
            ok = std::fabs(analytic.data()[i] - numeric.data()[i]) < 3e-2f;
        check(ok, "softmax CE grad");
        Matrix confident(1, 3, std::vector<float>{10.0f, 0.0f, 0.0f});
        Matrix oh1(1, 3, std::vector<float>{1, 0, 0});
        check(sum(softmax_cross_entropy_value(confident, oh1)) < 1e-3f, "softmax CE value");
    }
    {
        Matrix probs(1, 3, std::vector<float>{0.2f, 0.5f, 0.3f});
        Matrix tprobs(1, 3, std::vector<float>{0.3f, 0.4f, 0.3f});
        check_grad([&](const Variable& in) { return kl_divergence_loss(in, Variable(tprobs, false)); },
                   probs, Matrix(1, 3, std::vector<float>{0, 0, 0}), "kl divergence grad");
    }
}

void shape_tests() {
    Matrix a = Matrix(2, 2, std::vector<float>{1, 2, 3, 4});
    Matrix b = Matrix(2, 2, std::vector<float>{5, 6, 7, 8});
    Matrix c = concat({a, b}, 1);
    c.download();
    check(c.rows() == 2 && c.cols() == 4 && c.at(1, 3) == 8, "concat cols");
    Matrix s = stack({a, b});
    s.download();
    check(s.rows() == 2 && s.cols() == 4 && s.at(1, 2) == 7, "stack");
    Matrix r = a.reshape(1, 4);
    r.download();
    check(r.at(0, 3) == 4, "reshape");
    Matrix oh = one_hot(Matrix(3, 1, std::vector<float>{2, 0, 1}), 3);
    oh.download();
    check(oh.at(0, 2) == 1 && oh.at(1, 0) == 1 && oh.at(2, 1) == 1 && oh.at(0, 0) == 0, "one_hot");
    Matrix ba = broadcast_add(Matrix(2, 1, std::vector<float>{10, 20}), Matrix(1, 3, std::vector<float>{1, 2, 3}));
    ba.download();
    check(ba.rows() == 2 && ba.cols() == 3 && ba.at(1, 2) == 23, "broadcast add (M,1)+(1,N)");
    Matrix bm = broadcast_multiply(Matrix(2, 1, std::vector<float>{2, 3}), Matrix(1, 2, std::vector<float>{4, 5}));
    bm.download();
    check(bm.at(0, 1) == 10 && bm.at(1, 0) == 12, "broadcast multiply");
    Matrix div = Matrix(1, 2, std::vector<float>{8, 9}) / Matrix(1, 2, std::vector<float>{2, 3});
    div.download();
    check(div.at(0, 0) == 4 && div.at(0, 1) == 3, "elementwise divide");
}

void tensor_tests() {
    // batch_matmul backward: grad A = G x B^T, grad B = A^T x G with G = I.
    Tensor a({2, 2, 2}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor b({2, 2, 2}, std::vector<float>{0.5f, 0.1f, 0.2f, 0.3f, 0.4f, 0.2f, 0.1f, 0.6f});
    VarTensor av(a, true);
    VarTensor bv(b, true);
    av.batch_matmul(bv)
        .elementwise_multiply(VarTensor(Tensor({2, 2, 2}, std::vector<float>{1, 0, 0, 1, 1, 0, 0, 1}), false))
        .backward();
    Tensor ga = av.grad(); ga.download();
    Tensor gb = bv.grad(); gb.download();
    // grad A = G x B^T and grad B = A^T x G; with G = I this is A^T (not A).
    bool ok = true;
    for (int n = 0; n < 2; ++n)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                ok = ok && std::fabs(ga.data()[n * 4 + i * 2 + j] - b.data()[n * 4 + j * 2 + i]) < 1e-3f &&
                           std::fabs(gb.data()[n * 4 + i * 2 + j] - a.data()[n * 4 + j * 2 + i]) < 1e-3f;
    check(ok, "batch_matmul grad");

    // conv2d backward with all-ones upstream: sums check.
    Tensor input({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor weights({1, 1, 2, 2}, std::vector<float>{0.25f, -0.5f, 1.0f, 0.75f});
    Tensor bias({1}, std::vector<float>{0.1f});
    VarTensor iv(input, true);
    VarTensor wv(weights, true);
    VarTensor bv2(bias, true);
    VarTensor conv = iv.conv2d(wv, bv2);
    Tensor up_ones(conv.value().shape(), std::vector<float>(conv.value().size(), 1.0f));
    conv.elementwise_multiply(VarTensor(up_ones, false)).backward();
    Tensor gi = iv.grad(); gi.download();
    Tensor gw = wv.grad(); gw.download();
    Tensor gbias = bv2.grad(); gbias.download();
    // With an all-ones upstream gradient the convolution adjoints collapse to:
    //   sum(grad_input) = (#windows) * sum(weights) = 4 * 1.5 = 6
    //   sum(grad_weight) = sum of every window's input values = 12+16+24+28 = 80
    //   grad_bias[f] = number of windows = 4
    float total_in = 0, total_w = 0;
    for (std::size_t i = 0; i < gi.size(); ++i) total_in += gi.data()[i];
    for (std::size_t i = 0; i < gw.size(); ++i) total_w += gw.data()[i];
    check(std::fabs(total_in - 6.0f) < 1e-3f, "conv2d input grad sum");
    check(std::fabs(total_w - 80.0f) < 1e-3f, "conv2d weight grad sum");
    check(std::fabs(gbias.data()[0] - 4.0f) < 1e-3f, "conv2d bias grad");

    // pooling backward
    Tensor pool_in({1, 1, 2, 2}, std::vector<float>{1, 2, 3, 4});
    VarTensor pv(pool_in, true);
    pv.max_pool2d(2).elementwise_multiply(VarTensor(Tensor({1, 1, 1, 1}, std::vector<float>{1}), false)).backward();
    Tensor pg = pv.grad(); pg.download();
    check(std::fabs(pg.data()[0]) < 1e-6f && std::fabs(pg.data()[1]) < 1e-6f &&
          std::fabs(pg.data()[2]) < 1e-6f && std::fabs(pg.data()[3] - 1.0f) < 1e-6f, "max pool grad routes to argmax");
    VarTensor av2(pool_in, true);
    av2.avg_pool2d(2).elementwise_multiply(VarTensor(Tensor({1, 1, 1, 1}, std::vector<float>{1}), false)).backward();
    Tensor ag = av2.grad(); ag.download();
    bool avg_ok = true;
    for (std::size_t i = 0; i < 4; ++i) avg_ok = avg_ok && std::fabs(ag.data()[i] - 0.25f) < 1e-5f;
    check(avg_ok, "avg pool grad uniform");

    VarTensor cv(Tensor({1, 2}, std::vector<float>{1.0f, 4.0f}), true);
    cv.custom_unary("double",
        [](const Tensor& x) { return tensor_multiply(x, 2.0f); },
        [](const Tensor& g, const Tensor&, const Tensor&) { return tensor_multiply(g, 2.0f); })
        .elementwise_multiply(VarTensor(Tensor({1, 2}, std::vector<float>{1, 1}), false))
        .backward();
    Tensor cg = cv.grad(); cg.download();
    check(std::fabs(cg.data()[0] - 2.0f) < 1e-5f && std::fabs(cg.data()[1] - 2.0f) < 1e-5f, "vartensor custom_unary grad");
}

void persistence_tests() {
    Matrix m(2, 2, std::vector<float>{1.5f, -2.5f, 3.25f, 4.0f});
    m.save("autograd_test_matrix.bin");
    Matrix loaded = Matrix::load("autograd_test_matrix.bin");
    loaded.download();
    check(loaded.at(1, 1) == 4.0f && loaded.at(0, 1) == -2.5f, "save/load roundtrip");
    std::remove("autograd_test_matrix.bin");
}

}

int main() {
    try {
        variable_tests();
        loss_tests();
        shape_tests();
        tensor_tests();
        persistence_tests();
    } catch (const std::exception& error) {
        return matrix_pro::test::fatal(error);
    }
    return matrix_pro::test::summary("AUTOGRAD");
}


