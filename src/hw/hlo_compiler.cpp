/**
 * @file hlo_compiler.cpp
 * @brief Tek f32 dot iceren HLO metnini GEMM cagrisina indirger.
 *
 * Ornek:
 *   a = f32[8,4] parameter(0)
 *   b = f32[4,16] parameter(1)
 *   ROOT c = f32[8,16] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
 */

#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/core/heterogeneous_scheduler.hpp"
#include "matrix_pro/cpu/gemm.hpp"

#include <cctype>
#include <string>
#include <utility>

namespace matrix_pro {
namespace {

std::string compact(const char* text) {
    std::string out;
    if (text == nullptr) {
        return out;
    }
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (!std::isspace(static_cast<unsigned char>(*cursor))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*cursor))));
        }
    }
    return out;
}

int parse_int(const std::string& text, std::size_t& cursor) {
    if (cursor >= text.size() || !std::isdigit(static_cast<unsigned char>(text[cursor]))) {
        throw MatrixProError("HLO: expected an integer dimension");
    }
    int value = 0;
    while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor]))) {
        value = value * 10 + (text[cursor] - '0');
        ++cursor;
    }
    return value;
}

}  // namespace

void HloExecutable::run(const float* a, const float* b, float* c) const {
    gemm_f32(m_, n_, k_, a, k_, b, n_, c, n_, 1.0f, 0.0f);
}

HloExecutable compile_hlo(const char* text) {
    const std::string source = compact(text);
    const std::size_t p0 = source.find("parameter(0)");
    const std::size_t p1 = source.find("parameter(1)");
    const std::size_t dot = source.find("dot(");
    if (p0 == std::string::npos || p1 == std::string::npos || dot == std::string::npos) {
        throw MatrixProError("HLO: a single f32 dot with two parameters is required");
    }
    auto shape_before = [&](std::size_t marker) {
        const std::size_t key = source.rfind("f32[", marker);
        if (key == std::string::npos) {
            throw MatrixProError("HLO: missing f32 shape");
        }
        std::size_t cursor = key + 4;
        const int rows = parse_int(source, cursor);
        if (cursor >= source.size() || source[cursor] != ',') {
            throw MatrixProError("HLO: shape must be rank 2");
        }
        ++cursor;
        const int cols = parse_int(source, cursor);
        return std::pair<int, int>(rows, cols);
    };
    const std::pair<int, int> left = shape_before(p0);
    const std::pair<int, int> right = shape_before(p1);
    const std::pair<int, int> result = shape_before(dot);
    if (source.find("lhs_contracting_dims={1}") == std::string::npos ||
        source.find("rhs_contracting_dims={0}") == std::string::npos) {
        throw MatrixProError("HLO: only lhs_contracting_dims={1}, rhs_contracting_dims={0} is compiled");
    }
    if (left.second != right.first || result.first != left.first || result.second != right.second) {
        throw MatrixProError("HLO: dot shapes are inconsistent");
    }
    HloExecutable executable;
    executable.m_ = left.first;
    executable.k_ = left.second;
    executable.n_ = right.second;
    return executable;
}

}  // namespace matrix_pro
