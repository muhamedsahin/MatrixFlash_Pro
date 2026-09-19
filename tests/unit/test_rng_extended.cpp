#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        ctx.section("bernoulli_gpu");
        {
            // Sample 10000 elements with prob 0.7
            Matrix b = bernoulli_gpu(100, 100, 0.7f);
            b.download();
            float sum = 0.0f;
            bool only_binary = true;
            for (float v : b.data()) {
                if (v != 0.0f && v != 1.0f) only_binary = false;
                sum += v;
            }
            ctx.check(only_binary, "bernoulli outputs only 0 or 1");
            float mean = sum / static_cast<float>(b.size());
            ctx.check_near(mean, 0.7f, 0.03, "bernoulli mean close to 0.7");
        }

        ctx.section("exponential_gpu");
        {
            float lambda = 2.0f;
            Matrix e = exponential_gpu(100, 100, lambda);
            e.download();
            float sum = 0.0f;
            bool non_negative = true;
            for (float v : e.data()) {
                if (v < 0.0f) non_negative = false;
                sum += v;
            }
            ctx.check(non_negative, "exponential outputs non-negative values");
            float mean = sum / static_cast<float>(e.size());
            // E[X] = 1/lambda = 0.5
            ctx.check_near(mean, 0.5f, 0.05, "exponential mean close to 1/lambda = 0.5");
        }

        ctx.section("randint_gpu");
        {
            Matrix r = randint_gpu(100, 100, 10, 20);
            r.download();
            bool in_range = true;
            for (float v : r.data()) {
                int iv = static_cast<int>(v);
                if (iv < 10 || iv >= 20) in_range = false;
            }
            ctx.check(in_range, "randint values in [10, 20)");
        }

        ctx.section("randperm_gpu");
        {
            std::size_t n = 50;
            Matrix p = randperm_gpu(n);
            p.download();
            ctx.check(p.size() == n, "randperm size = n");
            std::vector<int> vals;
            for (float v : p.data()) vals.push_back(static_cast<int>(v));
            std::sort(vals.begin(), vals.end());
            bool is_perm = true;
            for (std::size_t i = 0; i < n; ++i) {
                if (vals[i] != static_cast<int>(i)) { is_perm = false; break; }
            }
            ctx.check(is_perm, "randperm contains all integers 0..n-1 exactly once");
        }

        ctx.section("RNGState reproducibility");
        {
            RNGState state1(12345ULL);
            Matrix m1 = state1.uniform(10, 10, 0.0f, 1.0f);

            RNGState state2(12345ULL);
            Matrix m2 = state2.uniform(10, 10, 0.0f, 1.0f);

            m1.download();
            m2.download();
            bool exact_match = true;
            for (std::size_t i = 0; i < m1.size(); ++i) {
                if (m1.data()[i] != m2.data()[i]) { exact_match = false; break; }
            }
            ctx.check(exact_match, "same seed produces identical numbers in RNGState");
        }

        return ctx.summary("RNG_EXTENDED");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

