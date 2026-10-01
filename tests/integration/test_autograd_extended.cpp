#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <vector>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("NoGradGuard scope");
        {
            ctx.check(is_grad_enabled(), "grad initially enabled");
            {
                NoGradGuard guard;
                ctx.check(!is_grad_enabled(), "grad disabled inside guard");
            }
            ctx.check(is_grad_enabled(), "grad restored after guard exit");
        }

        ctx.section("Variable detach & clone");
        {
            Matrix m(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
            Variable v(m, true);
            Variable d = v.detach();
            ctx.check(!d.requires_grad(), "detached variable requires_grad is false");
            ctx.check(d.is_leaf(), "detached variable is leaf");

            Variable c = v.clone();
            ctx.check(c.is_leaf(), "cloned variable is leaf");
            Matrix cloned = c.value(); cloned.download();
            ctx.check_near(cloned.at(0, 0), 1.0f, 1e-5, "cloned value matches");
        }

        ctx.section("Variable clip_grad_value_");
        {
            Matrix m(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
            Variable v(m, true);
            // loss = (v * 10).sum() => grad = 10
            Variable loss = v.multiply(10.0f).sum();
            loss.backward();

            std::vector<Variable> params = {v};
            Variable::clip_grad_value_(params, 5.0f);

            Matrix gradient = v.grad(); gradient.download();
            bool all_clipped = true;
            for (float g : gradient.data()) {
                if (g > 5.0f + 1e-4f || g < -5.0f - 1e-4f) all_clipped = false;
            }
            ctx.check(all_clipped, "gradients clipped to [-5, 5]");
        }

        ctx.section("Variable clip_grad_norm_");
        {
            Matrix m(1, 4, {1.0f, 1.0f, 1.0f, 1.0f});
            Variable v(m, true);
            // loss = (v * 10).sum() => grad = [10, 10, 10, 10], norm = 20
            Variable loss = v.multiply(10.0f).sum();
            loss.backward();

            std::vector<Variable> params = {v};
            Variable::clip_grad_norm_(params, 10.0f); // clip norm from 20 to 10

            Matrix gradient = v.grad(); gradient.download();
            float norm = v.grad().l2_norm();
            ctx.check_near(norm, 10.0f, 0.05, "gradient L2 norm clipped to 10");
        }

        return ctx.summary("AUTOGRAD_EXTENDED");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

