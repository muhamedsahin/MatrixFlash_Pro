#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <vector>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("Tensor construction & default strides");
        {
            Tensor t({2, 3, 4});
            ctx.check(t.rank() == 3, "rank is 3");
            ctx.check(t.size() == 24, "size is 24");
            ctx.check(t.is_contiguous(), "new tensor is contiguous");
            const auto& str = t.strides();
            ctx.check(str.size() == 3, "strides size is 3");
            ctx.check(str[0] == 12 && str[1] == 4 && str[2] == 1, "contiguous row-major strides");
        }

        ctx.section("Tensor fill & contiguous data");
        {
            Tensor t({2, 3});
            t.fill(42.0f);
            t.download();
            bool all_42 = true;
            for (float val : t.data()) {
                if (val != 42.0f) { all_42 = false; break; }
            }
            ctx.check(all_42, "fill set all elements to 42");
        }

        ctx.section("Tensor permute & contiguous copy");
        {
            // Create a 2x3 tensor with known values: [[1, 2, 3], [4, 5, 6]]
            std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            Tensor t({2, 3}, values);

            // Permute (transpose) dims (0, 1) -> (1, 0) => shape {3, 2}
            Tensor pt = t.permute({1, 0});
            ctx.check(pt.rank() == 2, "permuted rank is 2");
            ctx.check(pt.shape()[0] == 3 && pt.shape()[1] == 2, "permuted shape is {3, 2}");
            ctx.check(!pt.is_contiguous(), "transposed tensor is non-contiguous");

            // Make contiguous copy and verify values
            Tensor c = pt.contiguous();
            ctx.check(c.is_contiguous(), "c is contiguous");
            c.download();
            // Expected row-major {3, 2}: [[1, 4], [2, 5], [3, 6]]
            std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
            bool matches = true;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (c.data()[i] != expected[i]) { matches = false; break; }
            }
            ctx.check(matches, "permute + contiguous correctly reordered values");
        }

        ctx.section("Tensor view (reshape)");
        {
            std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            Tensor t({2, 3}, values);

            Tensor v = t.view({3, 2});
            ctx.check(v.shape()[0] == 3 && v.shape()[1] == 2, "view shape {3, 2}");
            ctx.check(v.size() == 6, "view size is 6");
        }

        ctx.section("Tensor squeeze / unsqueeze");
        {
            Tensor t({1, 4, 1, 5});
            Tensor s0 = t.squeeze(0);
            ctx.check(s0.shape() == std::vector<std::size_t>({4, 1, 5}), "squeeze dim 0");

            Tensor s2 = s0.squeeze(1);
            ctx.check(s2.shape() == std::vector<std::size_t>({4, 5}), "squeeze dim 1");

            Tensor u = s2.unsqueeze(1);
            ctx.check(u.shape() == std::vector<std::size_t>({4, 1, 5}), "unsqueeze dim 1");
        }

        ctx.section("Tensor narrow & select");
        {
            std::vector<float> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            Tensor t({3, 4}, vals);

            Tensor narrowed = t.narrow(0, 1, 2); // rows 1..2 (length 2)
            ctx.check(narrowed.shape() == std::vector<std::size_t>({2, 4}), "narrow shape {2, 4}");
            Tensor n_contig = narrowed.contiguous();
            n_contig.download();
            ctx.check_near(n_contig.data()[0], 5.0f, 1e-5, "narrow start matches row 1");

            Tensor selected = t.select(0, 2); // row 2
            ctx.check(selected.shape() == std::vector<std::size_t>({4}), "select rank reduced to 1");
            Tensor s_contig = selected.contiguous();
            s_contig.download();
            ctx.check_near(s_contig.data()[0], 9.0f, 1e-5, "select row 2 element 0 matches 9");
        }

        ctx.section("Tensor expand (broadcasting view)");
        {
            Tensor t({1, 3}, {10.0f, 20.0f, 30.0f});
            Tensor exp = t.expand({4, 3});
            ctx.check(exp.shape() == std::vector<std::size_t>({4, 3}), "expanded shape {4, 3}");
            ctx.check(exp.strides()[0] == 0, "broadcast dimension stride is 0");
            Tensor c = exp.contiguous();
            c.download();
            ctx.check(c.size() == 12, "expanded contiguous size 12");
            ctx.check_near(c.data()[0], 10.0f, 1e-5, "exp[0,0] = 10");
            ctx.check_near(c.data()[3], 10.0f, 1e-5, "exp[1,0] = 10");
            ctx.check_near(c.data()[4], 20.0f, 1e-5, "exp[1,1] = 20");
        }

        return ctx.summary("TENSOR_ADVANCED");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

