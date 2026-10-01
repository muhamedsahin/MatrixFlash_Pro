// Dahili uygulama parçası: src/ops/masking.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Compare/logical, finite predicates, selection ve mask count çekirdekleri.

namespace {

enum class CompareOp { greater, less, equal, not_equal };
enum class LogicalOp { and_op, or_op, not_op };

__global__ void compare_kernel(const float* input, float* output, std::size_t count, float scalar,
                               CompareOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float value = input[index];
    switch (op) {
    case CompareOp::greater:
        output[index] = value > scalar ? 1.0f : 0.0f;
        break;
    case CompareOp::less:
        output[index] = value < scalar ? 1.0f : 0.0f;
        break;
    case CompareOp::equal:
        output[index] = value == scalar ? 1.0f : 0.0f;
        break;
    case CompareOp::not_equal:
        output[index] = value != scalar ? 1.0f : 0.0f;
        break;
    }
}

__global__ void compare_pair_kernel(const float* left, const float* right, float* output,
                                    std::size_t count, CompareOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float a = left[index];
    const float b = right[index];
    switch (op) {
    case CompareOp::greater:
        output[index] = a > b ? 1.0f : 0.0f;
        break;
    case CompareOp::less:
        output[index] = a < b ? 1.0f : 0.0f;
        break;
    case CompareOp::equal:
        output[index] = a == b ? 1.0f : 0.0f;
        break;
    case CompareOp::not_equal:
        output[index] = a != b ? 1.0f : 0.0f;
        break;
    }
}

__global__ void logical_kernel(const float* left, const float* right, float* output,
                               std::size_t count, LogicalOp op) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const bool a = left[index] != 0.0f;
    const bool b = right[index] != 0.0f;
    switch (op) {
    case LogicalOp::and_op:
        output[index] = (a && b) ? 1.0f : 0.0f;
        break;
    case LogicalOp::or_op:
        output[index] = (a || b) ? 1.0f : 0.0f;
        break;
    case LogicalOp::not_op:
        output[index] = (!a) ? 1.0f : 0.0f;
        break;
    }
}

__global__ void unary_predicate_kernel(const float* input, float* output, std::size_t count,
                                       int mode) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float value = input[index];
    if (mode == 0)
        output[index] = ::isnan(value) ? 1.0f : 0.0f;
    else if (mode == 1)
        output[index] = ::isinf(value) ? 1.0f : 0.0f;
    else
        output[index] = ::isfinite(value) ? 1.0f : 0.0f;
}

__global__ void where_kernel(const float* condition, const float* true_value,
                             const float* false_value, float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index] = condition[index] != 0.0f ? true_value[index] : false_value[index];
}

__global__ void where_scalar_kernel(const float* condition, float true_value, float false_value,
                                    float* output, std::size_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index] = condition[index] != 0.0f ? true_value : false_value;
}

__global__ void apply_mask_kernel(const float* input, const float* mask, float* output,
                                  std::size_t count, float value) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index] = mask[index] != 0.0f ? value : input[index];
}

__global__ void any_kernel(const float* input, std::size_t count, unsigned int* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index] != 0.0f)
        atomicOr(result, 1U);
}

__global__ void all_kernel(const float* input, std::size_t count, unsigned int* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && input[index] == 0.0f)
        atomicAnd(result, 0U);
}

__global__ void count_mask_kernel(const float* mask, std::size_t count,
                                  unsigned long long* result) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && mask[index] != 0.0f)
        atomicAdd(result, 1ULL);
}

__global__ void scatter_mask_kernel(const float* input, const float* mask, float* output,
                                    std::size_t count, unsigned long long* position) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && mask[index] != 0.0f) {
        const auto destination = atomicAdd(position, 1ULL);
        output[destination] = input[index];
    }
}

}
