// Dahili uygulama parçası: src/ops/elementwise.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Float4 launch yardımcılarinin aritmetik ve unary functorlari.

enum class UnaryOp { exp, log, sqrt, abs, clamp, sigmoid, tanh, pow_scalar, negate };

// Elementwise functors for the vectorized launch helpers. Keeping the work in
// functors lets a single templated kernel read/write 16 bytes per thread while
// still serving every operation.
struct AddOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};
struct SubtractOp {
    __device__ float operator()(float a, float b) const { return a - b; }
};
struct MultiplyOp {
    __device__ float operator()(float a, float b) const { return a * b; }
};
struct MultiplyScalarOp {
    float scalar;
    __device__ float operator()(float a) const { return a * scalar; }
};
struct AddScalarOp {
    float scalar;
    __device__ float operator()(float a) const { return a + scalar; }
};

struct UnaryDispatcher {
    UnaryOp op;
    float param_a;
    float param_b;
    __device__ float operator()(float value) const {
        switch (op) {
        case UnaryOp::exp:
            return expf(value);
        case UnaryOp::log:
            return logf(value);
        case UnaryOp::sqrt:
            return sqrtf(value);
        case UnaryOp::abs:
            return fabsf(value);
        case UnaryOp::clamp:
            return fminf(fmaxf(value, param_a), param_b);
        case UnaryOp::sigmoid:
            return 1.0f / (1.0f + expf(-value));
        case UnaryOp::tanh:
            return tanhf(value);
        case UnaryOp::pow_scalar:
            return powf(value, param_a);
        case UnaryOp::negate:
            return -value;
        }
        return value;
    }
};

__global__ void add_row_vector_kernel(const float* __restrict__ matrix,
                                      const float* __restrict__ vector, float* __restrict__ output,
                                      std::size_t rows, std::size_t cols) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols)
        return;
    output[index] = matrix[index] + vector[index % cols];
}
