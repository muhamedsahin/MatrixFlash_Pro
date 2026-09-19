#pragma once

namespace matrix_pro {

class Matrix;

// Trigonometric
Matrix sin(const Matrix& m);
Matrix cos(const Matrix& m);
Matrix tan(const Matrix& m);
Matrix asin(const Matrix& m);
Matrix acos(const Matrix& m);
Matrix atan(const Matrix& m);
Matrix atan2(const Matrix& y, const Matrix& x);

// Hyperbolic
Matrix sinh(const Matrix& m);
Matrix cosh(const Matrix& m);

// Rounding
Matrix ceil(const Matrix& m);
Matrix floor(const Matrix& m);
Matrix round(const Matrix& m);
Matrix trunc(const Matrix& m);

// Special functions
Matrix erf(const Matrix& m);
Matrix erfc(const Matrix& m);
Matrix erfinv(const Matrix& m);
Matrix lgamma(const Matrix& m);

// Fast elementwise
Matrix reciprocal(const Matrix& m);
Matrix rsqrt(const Matrix& m);
Matrix sign(const Matrix& m);
Matrix fmod(const Matrix& a, const Matrix& b);
Matrix remainder(const Matrix& a, const Matrix& b);
Matrix lerp(const Matrix& a, const Matrix& b, float t);
Matrix lerp(const Matrix& a, const Matrix& b, const Matrix& t);
Matrix addcmul(const Matrix& self, const Matrix& t1, const Matrix& t2, float alpha = 1.0f);
Matrix addcdiv(const Matrix& self, const Matrix& t1, const Matrix& t2, float alpha = 1.0f);

// Elementwise min/max
Matrix minimum(const Matrix& a, const Matrix& b);
Matrix maximum(const Matrix& a, const Matrix& b);

} // namespace matrix_pro
