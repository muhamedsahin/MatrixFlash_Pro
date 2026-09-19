#include "matrix_pro/matrix_pro.hpp"
#include "../support/test_support.hpp"

#include <cmath>

using namespace matrix_pro;
using matrix_pro::test::Context;

int main() {
    try {
        Context ctx;

        // ---- trigonometric functions ----
        ctx.section("sin / cos identity: sin^2 + cos^2 = 1");
        {
            Matrix m(2, 3, {0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f});
            Matrix s = matrix_pro::sin(m);
            Matrix c = matrix_pro::cos(m);
            s.download(); c.download();
            Matrix s2 = s.elementwise_multiply(s);
            Matrix c2 = c.elementwise_multiply(c);
            Matrix sum = matrix_pro::add(s2, c2);
            sum.download();
            for (std::size_t i = 0; i < sum.size(); ++i)
                ctx.check_near(sum.data()[i], 1.0f, 1e-4, "sin^2+cos^2=1 [" + std::to_string(i) + "]");
        }

        ctx.section("tan = sin / cos");
        {
            Matrix m(1, 4, {0.1f, 0.3f, 0.7f, 1.2f});
            Matrix t = matrix_pro::tan(m);
            Matrix s = matrix_pro::sin(m);
            Matrix c = matrix_pro::cos(m);
            Matrix ratio = matrix_pro::divide(s, c);
            t.download(); ratio.download();
            for (std::size_t i = 0; i < t.size(); ++i)
                ctx.check_near(t.data()[i], ratio.data()[i], 1e-4, "tan=sin/cos [" + std::to_string(i) + "]");
        }

        ctx.section("asin(sin(x)) = x for small x");
        {
            Matrix m(1, 3, {0.1f, 0.3f, 0.5f});
            Matrix round_trip = matrix_pro::asin(matrix_pro::sin(m));
            round_trip.download(); m.download();
            for (std::size_t i = 0; i < m.size(); ++i)
                ctx.check_near(round_trip.data()[i], m.data()[i], 1e-4, "asin(sin(x))=x [" + std::to_string(i) + "]");
        }

        ctx.section("acos(cos(x)) = x for x in [0, pi]");
        {
            Matrix m(1, 3, {0.1f, 0.5f, 1.0f});
            Matrix round_trip = matrix_pro::acos(matrix_pro::cos(m));
            round_trip.download(); m.download();
            for (std::size_t i = 0; i < m.size(); ++i)
                ctx.check_near(round_trip.data()[i], m.data()[i], 1e-4, "acos(cos(x))=x [" + std::to_string(i) + "]");
        }

        ctx.section("atan known values");
        {
            Matrix m(1, 2, {1.0f, 0.0f});
            Matrix result = matrix_pro::atan(m);
            result.download();
            ctx.check_near(result.data()[0], std::atan(1.0f), 1e-5, "atan(1) = pi/4");
            ctx.check_near(result.data()[1], 0.0f, 1e-5, "atan(0) = 0");
        }

        ctx.section("atan2(y, x)");
        {
            Matrix y(1, 2, {1.0f, -1.0f});
            Matrix x(1, 2, {1.0f, -1.0f});
            Matrix result = matrix_pro::atan2(y, x);
            result.download();
            ctx.check_near(result.data()[0], std::atan2(1.0f, 1.0f), 1e-5, "atan2(1,1)");
            ctx.check_near(result.data()[1], std::atan2(-1.0f, -1.0f), 1e-4, "atan2(-1,-1)");
        }

        // ---- hyperbolic functions ----
        ctx.section("sinh / cosh");
        {
            Matrix m(1, 3, {0.0f, 0.5f, 1.0f});
            Matrix sh = matrix_pro::sinh(m);
            Matrix ch = matrix_pro::cosh(m);
            sh.download(); ch.download();
            for (std::size_t i = 0; i < 3; ++i) {
                ctx.check_near(sh.data()[i], std::sinh(m.data()[i]), 1e-4, "sinh [" + std::to_string(i) + "]");
                ctx.check_near(ch.data()[i], std::cosh(m.data()[i]), 1e-4, "cosh [" + std::to_string(i) + "]");
            }
        }

        // ---- rounding ----
        ctx.section("ceil / floor / round / trunc");
        {
            Matrix m(1, 4, {1.3f, 2.7f, -1.5f, -2.3f});
            Matrix ce = matrix_pro::ceil(m);  ce.download();
            Matrix fl = matrix_pro::floor(m); fl.download();
            Matrix ro = matrix_pro::round(m); ro.download();
            Matrix tr = matrix_pro::trunc(m); tr.download();
            ctx.check_near(ce.data()[0], 2.0f, 1e-5, "ceil(1.3)=2");
            ctx.check_near(fl.data()[0], 1.0f, 1e-5, "floor(1.3)=1");
            ctx.check_near(ro.data()[1], 3.0f, 1e-5, "round(2.7)=3");
            ctx.check_near(tr.data()[3], -2.0f, 1e-5, "trunc(-2.3)=-2");
        }

        // ---- special functions ----
        ctx.section("erf / erfc");
        {
            Matrix m(1, 3, {0.0f, 1.0f, 2.0f});
            Matrix e = matrix_pro::erf(m);    e.download();
            Matrix ec = matrix_pro::erfc(m);   ec.download();
            ctx.check_near(e.data()[0], 0.0f, 1e-5, "erf(0)=0");
            ctx.check_near(e.data()[1], std::erf(1.0f), 1e-4, "erf(1)");
            for (std::size_t i = 0; i < 3; ++i)
                ctx.check_near(e.data()[i] + ec.data()[i], 1.0f, 1e-4, "erf+erfc=1 [" + std::to_string(i) + "]");
        }

        ctx.section("erfinv roundtrip: erf(erfinv(x)) = x");
        {
            Matrix m(1, 3, {0.0f, 0.5f, 0.9f});
            Matrix inv = matrix_pro::erfinv(m);
            Matrix round_trip = matrix_pro::erf(inv);
            round_trip.download(); m.download();
            for (std::size_t i = 0; i < 3; ++i)
                ctx.check_near(round_trip.data()[i], m.data()[i], 1e-3, "erf(erfinv(x))=x [" + std::to_string(i) + "]");
        }

        // ---- fast elementwise ----
        ctx.section("reciprocal: reciprocal(x) * x = 1");
        {
            Matrix m(1, 4, {2.0f, 4.0f, 0.5f, 10.0f});
            Matrix r = matrix_pro::reciprocal(m);
            Matrix product = r.elementwise_multiply(m);
            product.download();
            for (std::size_t i = 0; i < 4; ++i)
                ctx.check_near(product.data()[i], 1.0f, 1e-5, "reciprocal*x=1 [" + std::to_string(i) + "]");
        }

        ctx.section("rsqrt: rsqrt(x)^2 * x = 1");
        {
            Matrix m(1, 4, {1.0f, 4.0f, 9.0f, 16.0f});
            Matrix r = matrix_pro::rsqrt(m);
            r.download();
            ctx.check_near(r.data()[0], 1.0f, 1e-5, "rsqrt(1)=1");
            ctx.check_near(r.data()[1], 0.5f, 1e-5, "rsqrt(4)=0.5");
            ctx.check_near(r.data()[2], 1.0f/3.0f, 1e-5, "rsqrt(9)=1/3");
        }

        ctx.section("sign");
        {
            Matrix m(1, 4, {3.0f, -2.0f, 0.0f, -0.0f});
            Matrix s = matrix_pro::sign(m);
            s.download();
            ctx.check_near(s.data()[0], 1.0f, 1e-5, "sign(3)=1");
            ctx.check_near(s.data()[1], -1.0f, 1e-5, "sign(-2)=-1");
            ctx.check_near(s.data()[2], 0.0f, 1e-5, "sign(0)=0");
        }

        // ---- binary ops ----
        ctx.section("lerp(a, b, 0.5) = (a+b)/2");
        {
            Matrix a(1, 3, {0.0f, 2.0f, 4.0f});
            Matrix b(1, 3, {10.0f, 8.0f, 6.0f});
            Matrix result = matrix_pro::lerp(a, b, 0.5f);
            result.download();
            ctx.check_near(result.data()[0], 5.0f, 1e-5, "lerp(0,10,0.5)=5");
            ctx.check_near(result.data()[1], 5.0f, 1e-5, "lerp(2,8,0.5)=5");
            ctx.check_near(result.data()[2], 5.0f, 1e-5, "lerp(4,6,0.5)=5");
        }

        ctx.section("minimum / maximum");
        {
            Matrix a(1, 3, {1.0f, 5.0f, 3.0f});
            Matrix b(1, 3, {2.0f, 3.0f, 4.0f});
            Matrix mn = matrix_pro::minimum(a, b);
            Matrix mx = matrix_pro::maximum(a, b);
            mn.download(); mx.download();
            ctx.check_near(mn.data()[0], 1.0f, 1e-5, "min(1,2)=1");
            ctx.check_near(mn.data()[1], 3.0f, 1e-5, "min(5,3)=3");
            ctx.check_near(mx.data()[0], 2.0f, 1e-5, "max(1,2)=2");
            ctx.check_near(mx.data()[2], 4.0f, 1e-5, "max(3,4)=4");
        }

        ctx.section("addcmul: self + alpha * t1 * t2");
        {
            Matrix self(1, 3, {1.0f, 2.0f, 3.0f});
            Matrix t1(1, 3, {2.0f, 3.0f, 4.0f});
            Matrix t2(1, 3, {3.0f, 4.0f, 5.0f});
            Matrix result = matrix_pro::addcmul(self, t1, t2, 2.0f);
            result.download();
            ctx.check_near(result.data()[0], 1.0f + 2.0f * 2.0f * 3.0f, 1e-4, "addcmul [0]");
            ctx.check_near(result.data()[1], 2.0f + 2.0f * 3.0f * 4.0f, 1e-4, "addcmul [1]");
        }

        return ctx.summary("MATH");
    } catch (const std::exception& e) {
        matrix_pro::test::Context{}.fatal(e);
        return 99;
    }
}

