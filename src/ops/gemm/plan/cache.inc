// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: 128 planlık thread-local önbellek ve eski raw dispatch adaptörü.

namespace {
using CacheKey = std::tuple<int, cudaStream_t, std::size_t, std::size_t, std::size_t, bool, bool,
                            int, int, std::size_t>;
GemmPlan& cached(std::size_t m, std::size_t n, std::size_t k, const GemmOptions& o, int device,
                 cudaStream_t stream) {
    static thread_local std::map<CacheKey, std::unique_ptr<GemmPlan>> plans;
    static thread_local GemmPlan* last = nullptr;
    static thread_local CacheKey last_key;
    CacheKey key{device,
                 stream,
                 m,
                 n,
                 k,
                 o.transpose_left,
                 o.transpose_right,
                 static_cast<int>(o.precision),
                 static_cast<int>(o.epilogue),
                 o.workspace_limit_bytes};
    if (last && key == last_key)
        return *last;
    auto it = plans.find(key);
    if (it == plans.end()) {
        // Only cold construction needs to change the library execution context.
        // Warm raw dispatch already supplies its stream and skips repeated queries.
        auto old = compute_stream();
        if (stream != old)
            detail::set_compute_stream(stream);
        std::unique_ptr<GemmPlan> plan;
        try {
            plan = std::make_unique<GemmPlan>(m, n, k, o);
        } catch (...) {
            if (stream != old)
                detail::set_compute_stream(old);
            throw;
        }
        if (stream != old)
            detail::set_compute_stream(old);
        if (plans.size() >= 128) {
            last = nullptr;
            plans.erase(plans.begin());
        }
        it = plans.emplace(key, std::move(plan)).first;
    }
    last = it->second.get();
    last_key = key;
    return *last;
}
}
namespace detail {
struct GemmPlanAccess {
    static void run(GemmPlan& plan, const float* left, const float* right, float* output,
                    const float* bias) {
        plan.impl_->run_ptr(left, right, output, 1, 0, bias);
    }
};
namespace gemm {
void gemm_cached_raw(const float* left, const float* right, float* output, const float* bias, int m,
                     int n, int k, Epilogue epi, cudaStream_t stream) {
    if (m < 0 || n < 0 || k < 0)
        throw InvalidArgumentError("Negative GEMM dimensions");
    GemmOptions options;
    options.precision = GemmPrecision::tf32;
    // The old default vector kernels used full FP32 input precision. Preserve
    // that contract when a long vector is redirected to the vendor backend.
    if (epi == Epilogue::none && (m == 1 || n == 1))
        options.precision = GemmPrecision::fp32;
    options.epilogue = static_cast<GemmEpilogue>(static_cast<int>(epi));
    GemmPlanAccess::run(cached(m, n, k, options, current_device(), stream), left, right, output,
                        bias);
}
}
}
