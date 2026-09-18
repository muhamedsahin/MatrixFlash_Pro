#pragma once

// ============================================================================
//  MatrixFlash-Pro shared test harness
//  ---------------------------------------------------------------------------
//  Small, dependency-free assertion helper used by every test executable.
//  Usage:
//
//      #include "test_support.hpp"
//      using matrix_pro::test::check;
//
//      int main() {
//          try {
//              matrix_pro::test::section("matmul");
//              check(a + b == c, "add matches reference");
//          } catch (const std::exception& error) {
//              return matrix_pro::test::fatal(error);
//          }
//          return matrix_pro::test::summary("MATRIX");
//      }
//
//  Output is intentionally CTest-friendly: every failure is printed on its own
//  line prefixed with [  FAILED  ], and the process exit code is 0 only when
//  every check passed (99 for an unexpected exception).
// ============================================================================

#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

namespace matrix_pro {
namespace test {

// Shared counters for the current test executable (one process per test file,
// so a function-local static is the whole story).
class Context {
public:
    void section(const std::string& name) {
        section_ = name;
        std::cout << "[ RUN      ] " << name << '\n';
    }

    void check(bool ok, const std::string& what) {
        ++checks_;
        if (ok) return;
        ++failures_;
        std::cout << "[  FAILED  ] " << label() << what << '\n';
    }

    void check_near(double actual, double expected, double tolerance, const std::string& what) {
        const bool ok = std::fabs(actual - expected) <= tolerance;
        ++checks_;
        if (ok) return;
        ++failures_;
        std::cout << "[  FAILED  ] " << label() << what
                  << " (actual=" << actual << ", expected=" << expected
                  << ", tolerance=" << tolerance << ")\n";
    }

    // Verifies that `fn` throws an exception of type Exception.
    template <typename Exception, typename Fn>
    void check_throws(Fn&& fn, const std::string& what) {
        ++checks_;
        try {
            fn();
        } catch (const Exception&) {
            return;
        } catch (const std::exception& other) {
            ++failures_;
            std::cout << "[  FAILED  ] " << label() << what
                      << " (threw a different type: " << other.what() << ")\n";
            return;
        }
        ++failures_;
        std::cout << "[  FAILED  ] " << label() << what << " (nothing was thrown)\n";
    }

    int fatal(const std::exception& error) const {
        std::cout << "[  FAILED  ] uncaught exception in section '" << section_
                  << "': " << error.what() << '\n';
        return 99;
    }

    int summary(const std::string& suite) const {
        std::cout << "[==========] " << suite << ": " << checks_ << " checks, "
                  << failures_ << " failure" << (failures_ == 1 ? "" : "s") << '\n';
        if (failures_ == 0) {
            std::cout << "[  PASSED  ] " << suite << '\n';
            return 0;
        }
        std::cout << "[  FAILED  ] " << suite << '\n';
        return 1;
    }

    int failures() const noexcept { return failures_; }

private:
    std::string label() const {
        return section_.empty() ? std::string() : (section_ + ": ");
    }

    int checks_ = 0;
    int failures_ = 0;
    std::string section_;
};

inline Context& context() {
    static Context instance;
    return instance;
}

inline void section(const std::string& name) { context().section(name); }

inline void check(bool ok, const std::string& what) { context().check(ok, what); }

inline void check_near(double actual, double expected, double tolerance, const std::string& what) {
    context().check_near(actual, expected, tolerance, what);
}

template <typename Exception, typename Fn>
inline void check_throws(Fn&& fn, const std::string& what) {
    context().template check_throws<Exception>(std::forward<Fn>(fn), what);
}

inline int fatal(const std::exception& error) { return context().fatal(error); }

inline int summary(const std::string& suite) { return context().summary(suite); }

} // namespace test
} // namespace matrix_pro
