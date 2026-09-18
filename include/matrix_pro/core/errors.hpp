#pragma once

#include <stdexcept>
#include <string>

namespace matrix_pro {

// Root of the MatrixFlash-Pro exception hierarchy. Every exception thrown by
// the library derives from this type, so a single `catch (const
// matrix_pro::MatrixProError&)` handles all library failures while still
// allowing callers to react to a specific cause through the derived classes.
class MatrixProError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// An argument was rejected for a reason other than shape (bad probability,
// non-positive epsilon, empty operand, ...).
class InvalidArgumentError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

// Operand shapes or element counts are incompatible.
class ShapeMismatchError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

// An index, slice or range escapes the bounds of a container.
class OutOfRangeError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

// A CUDA runtime, cuBLAS or cuDNN call failed.
class CudaError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

// A cuSOLVER decomposition or linear-algebra routine failed.
class SolverError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

// A file / serialization operation failed.
class IoError : public MatrixProError {
public:
    using MatrixProError::MatrixProError;
};

} // namespace matrix_pro