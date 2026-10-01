#include "matrix_pro/ops/gemm.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include "matrix_pro/detail/gemm/gemm_api.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace matrix_pro {
namespace {
void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw CudaError(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}
void require_not_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    checkCuda(cudaStreamIsCapturing(stream, &status), "GEMM capture query");
    if (status != cudaStreamCaptureStatusNone)
        throw InvalidArgumentError("Construct/tune GEMM plans before CUDA Graph capture");
}
int checked_dim(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw InvalidArgumentError("GEMM dimension exceeds the 32-bit backend limit");
    return static_cast<int>(value);
}
// Shared only within one thread/device/stream. Scratch memory is never used by
// independent streams, even when their submissions come from the same thread.
struct Environment {
    int device;
    cudaStream_t stream;
    cublasLtHandle_t lt = nullptr;
    cublasHandle_t blas = nullptr;
    void* workspace = nullptr;
    std::size_t bytes = 64u << 20;
    Environment(int d, cudaStream_t s) : device(d), stream(s) {
        try {
            check_blas(cublasCreate(&blas), "plan cuBLAS create");
            check_blas(cublasSetStream(blas, stream), "plan cuBLAS stream");
            check_blas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "plan scalar mode");
            if (cudaMalloc(&workspace, bytes) != cudaSuccess) {
                cudaGetLastError(); bytes = 0; workspace = nullptr;
            }
            if (workspace) check_blas(cublasSetWorkspace(blas, workspace, bytes), "plan workspace");
            if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) lt = nullptr;
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        int old = 0; cudaGetDevice(&old);
        if (old != device) cudaSetDevice(device);
        // cudaFree waits for users of scratch; ensure handles outlive submissions.
        if (workspace) cudaFree(workspace);
        if (lt) cublasLtDestroy(lt);
        if (blas) cublasDestroy(blas);
        workspace = nullptr; lt = nullptr; blas = nullptr;
        if (old != device) cudaSetDevice(old);
    }
    ~Environment() { release(); }
};
std::shared_ptr<Environment> environment(int device, cudaStream_t stream) {
    using Key = std::pair<int, cudaStream_t>;
    static thread_local std::map<Key, std::weak_ptr<Environment>> states;
    const Key key{device, stream};
    auto it = states.find(key);
    if (it != states.end()) if (auto existing = it->second.lock()) return existing;
    for (auto entry = states.begin(); entry != states.end();) {
        if (entry->second.expired()) entry = states.erase(entry); else ++entry;
    }
    auto created = std::make_shared<Environment>(device, stream);
    states[key] = created;
    return created;
}
cublasLtEpilogue_t lt_epilogue(GemmEpilogue e) {
    switch (e) {
        case GemmEpilogue::bias: return CUBLASLT_EPILOGUE_BIAS;
        case GemmEpilogue::bias_relu: return CUBLASLT_EPILOGUE_RELU_BIAS;
        case GemmEpilogue::bias_gelu: return CUBLASLT_EPILOGUE_GELU_BIAS;
        default: return CUBLASLT_EPILOGUE_DEFAULT;
    }
}
detail::gemm::Epilogue fallback_epilogue(GemmEpilogue e) {
    return static_cast<detail::gemm::Epilogue>(static_cast<int>(e));
}
__global__ void scale_output(float* output, std::size_t count, float beta) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x)*gridDim.x)
        output[i] = beta == 0 ? 0 : beta * output[i];
}
}

struct GemmPlan::Impl {
    int m, n, k, device;
    GemmOptions options;
    cudaStream_t stream;
    std::thread::id owner = std::this_thread::get_id();
    std::shared_ptr<Environment> env;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    std::array<cublasLtMatmulHeuristicResult_t, 16> candidates{};
    int count = 0, selected = -1;
    cublasComputeType_t compute;
    std::size_t workspace_bytes = 0;

    Impl(std::size_t rows, std::size_t cols, std::size_t inner, GemmOptions o)
        : m(checked_dim(rows)), n(checked_dim(cols)), k(checked_dim(inner)),
          device(current_device()), options(o), stream(compute_stream()),
          compute(o.precision == GemmPrecision::tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F_PEDANTIC) {
        if (o.precision != GemmPrecision::fp32 && o.precision != GemmPrecision::tf32)
            throw InvalidArgumentError("Invalid GEMM precision policy");
        if (static_cast<int>(o.epilogue) < 0 || static_cast<int>(o.epilogue) > 3)
            throw InvalidArgumentError("Invalid GEMM epilogue");
        require_not_capturing(stream);
        if (m == 0 || n == 0 || k == 0) return;
        env = environment(device, stream);
        workspace_bytes = std::min(o.workspace_limit_bytes, env->bytes);
        if (!env->lt) return;
        try {
            check_blas(cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F), "plan descriptor");
            const cublasOperation_t ta = o.transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N;
            const cublasOperation_t tb = o.transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
            check_blas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "plan transpose B");
            check_blas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "plan transpose A");
            const auto epilogue = lt_epilogue(o.epilogue);
            check_blas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "plan epilogue");
            const int ar = o.transpose_right ? k : n, ac = o.transpose_right ? n : k;
            const int br = o.transpose_left ? m : k, bc = o.transpose_left ? k : m;
            check_blas(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, ar, ac, ar), "plan B layout");
            check_blas(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, br, bc, br), "plan A layout");
            check_blas(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n), "plan C layout");
            cublasLtMatmulPreference_t pref = nullptr;
            check_blas(cublasLtMatmulPreferenceCreate(&pref), "plan preference");
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
            const auto status = cublasLtMatmulAlgoGetHeuristic(env->lt, op, a, b, c, c, pref,
                                                             static_cast<int>(candidates.size()), candidates.data(), &count);
            cublasLtMatmulPreferenceDestroy(pref);
            if (status != CUBLAS_STATUS_SUCCESS) count = 0;
            for (int i=0; i<count; ++i) if (candidates[i].state == CUBLAS_STATUS_SUCCESS) { selected=i; break; }
        } catch (...) { destroy_descriptors(); throw; }
    }
    void destroy_descriptors() noexcept {
        if(c) cublasLtMatrixLayoutDestroy(c); if(b) cublasLtMatrixLayoutDestroy(b);
        if(a) cublasLtMatrixLayoutDestroy(a); if(op) cublasLtMatmulDescDestroy(op);
        c=nullptr; b=nullptr; a=nullptr; op=nullptr;
    }
    ~Impl() { destroy_descriptors(); }
    void validate(const Matrix& left, const Matrix& right, const Matrix& output, const Matrix* bias) const {
        if (std::this_thread::get_id() != owner || current_device() != device || compute_stream() != stream)
            throw InvalidArgumentError("GEMM plan must execute on its creating thread, device and stream");
        if(left.rows()!=static_cast<std::size_t>(options.transpose_left?k:m) ||
           left.cols()!=static_cast<std::size_t>(options.transpose_left?m:k) ||
           right.rows()!=static_cast<std::size_t>(options.transpose_right?n:k) ||
           right.cols()!=static_cast<std::size_t>(options.transpose_right?k:n) ||
           output.rows()!=static_cast<std::size_t>(m) || output.cols()!=static_cast<std::size_t>(n))
            throw ShapeMismatchError("GEMM plan operand shape mismatch");
        if (!output.empty() && (output.device_data()==left.device_data() || output.device_data()==right.device_data()))
            throw InvalidArgumentError("GEMM output cannot alias an input");
        if(options.epilogue!=GemmEpilogue::none) {
            if(!bias || bias->size()!=static_cast<std::size_t>(n)) throw ShapeMismatchError("GEMM bias length must equal N");
            if(!output.empty() && output.device_data()==bias->device_data()) throw InvalidArgumentError("GEMM output cannot alias bias");
        }
    }
    cublasStatus_t run_lt_ptr(const float* left, const float* right, const float* input,
                         float* output, float alpha, float beta, const float* bias, int index) {
        if (bias && options.epilogue!=GemmEpilogue::none) {
            const auto status = cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
            if(status!=CUBLAS_STATUS_SUCCESS) return status;
        }
        return cublasLtMatmul(env->lt, op, &alpha, right, a, left, b,
                             &beta, input, c, output, c, &candidates[index].algo,
                             env->workspace, workspace_bytes, stream);
    }
    cublasStatus_t run_lt(const Matrix& left,const Matrix& right,const float* input,float* output,
                         float alpha,float beta,const Matrix* bias,int index) {
        return run_lt_ptr(left.device_data(),right.device_data(),input,output,alpha,beta,bias?bias->device_data():nullptr,index);
    }
    void run_ptr(const float* left, const float* right, float* output, float alpha, float beta, const float* bias) {
        if (m==0 || n==0) return;
        if(k==0 || alpha==0) {
            const auto total=static_cast<std::size_t>(m)*n;
            scale_output<<<std::min<std::size_t>((total+255)/256,4096),256,0,stream>>>(output,total,beta);
            checkCuda(cudaGetLastError(), "GEMM scale output");
        } else {
            if(selected>=0 && run_lt_ptr(left,right,output,output,alpha,beta,bias,selected)==CUBLAS_STATUS_SUCCESS)
                return;
            selected=-1;
            check_blas(cublasGemmEx(env->blas, options.transpose_right?CUBLAS_OP_T:CUBLAS_OP_N,
                options.transpose_left?CUBLAS_OP_T:CUBLAS_OP_N, n,m,k,&alpha,
                right,CUDA_R_32F,options.transpose_right?k:n,left,CUDA_R_32F,options.transpose_left?m:k,
                &beta,output,CUDA_R_32F,n,compute,CUBLAS_GEMM_DEFAULT),"plan fallback GEMM");
        }
        detail::gemm::apply_epilogue(output,bias,m,n,fallback_epilogue(options.epilogue),stream);
    }
};

GemmPlan::GemmPlan(std::size_t m,std::size_t n,std::size_t k,GemmOptions o) : impl_(std::make_unique<Impl>(m,n,k,o)) {}
GemmPlan::~GemmPlan() = default;
GemmPlan::GemmPlan(GemmPlan&&) noexcept = default;
GemmPlan& GemmPlan::operator=(GemmPlan&&) noexcept = default;
std::size_t GemmPlan::rows() const noexcept { return impl_?impl_->m:0; }
std::size_t GemmPlan::cols() const noexcept { return impl_?impl_->n:0; }
std::size_t GemmPlan::inner() const noexcept { return impl_?impl_->k:0; }
int GemmPlan::candidate_count() const noexcept { return impl_?impl_->count:0; }
bool GemmPlan::uses_cublaslt() const noexcept { return impl_ && impl_->selected>=0; }
void GemmPlan::execute(const Matrix& left,const Matrix& right,Matrix& output,float alpha,float beta,const Matrix* bias) {
    if(!impl_) throw InvalidArgumentError("Cannot execute a moved-from GEMM plan");
    impl_->validate(left,right,output,bias);
    impl_->run_ptr(left.device_data(),right.device_data(),output.device_data(),alpha,beta,bias?bias->device_data():nullptr);
    if(!output.empty()) output.mark_host_stale();
}
GemmTuningResult GemmPlan::tune(const Matrix& left,const Matrix& right,const Matrix& output,float alpha,float beta,const Matrix* bias,int repeats) {
    if(!impl_) throw InvalidArgumentError("Cannot tune a moved-from GEMM plan");
    impl_->validate(left,right,output,bias);
    if(repeats<1) throw InvalidArgumentError("GEMM tuning repeats must be positive");
    require_not_capturing(impl_->stream);
    GemmTuningResult result;
    if(!impl_->count || output.empty() || impl_->k==0 || alpha==0) return result;
    Matrix scratch(output.rows(),output.cols(),MemoryMode::device_only);
    cudaEvent_t start=nullptr,stop=nullptr;
    checkCuda(cudaEventCreate(&start),"GEMM tuning event");
    try {
        checkCuda(cudaEventCreate(&stop),"GEMM tuning event");
        float best=std::numeric_limits<float>::infinity();
        for(int i=0;i<impl_->count;++i) {
            if(impl_->candidates[i].state!=CUBLAS_STATUS_SUCCESS) continue;
            auto call=[&]{return impl_->run_lt(left,right,output.device_data(),scratch.device_data(),alpha,beta,bias,i);};
            if(call()!=CUBLAS_STATUS_SUCCESS) continue;
            bool ok=true;
            std::array<float,3> times{};
            for(auto& time:times) {
                checkCuda(cudaEventRecord(start,impl_->stream),"GEMM tuning start");
                for(int r=0;r<repeats;++r) if(call()!=CUBLAS_STATUS_SUCCESS) {ok=false;break;}
                checkCuda(cudaEventRecord(stop,impl_->stream),"GEMM tuning stop");
                checkCuda(cudaEventSynchronize(stop),"GEMM tuning sync");
                checkCuda(cudaEventElapsedTime(&time,start,stop),"GEMM tuning elapsed");
                if(!ok)break;
            }
            if(!ok) continue;
            std::sort(times.begin(),times.end());const float ms=times[1];
            ++result.candidates_tested;
            if(ms<best) { best=ms; impl_->selected=i; result.selected_ms=ms/repeats; }
        }
    } catch(...) { if(stop)cudaEventDestroy(stop);cudaEventDestroy(start);throw; }
    cudaEventDestroy(stop);cudaEventDestroy(start);
    return result;
}
namespace {
using CacheKey=std::tuple<int,cudaStream_t,std::size_t,std::size_t,std::size_t,bool,bool,int,int,std::size_t>;
GemmPlan& cached(std::size_t m,std::size_t n,std::size_t k,const GemmOptions& o,
                 int device,cudaStream_t stream) {
    static thread_local std::map<CacheKey,std::unique_ptr<GemmPlan>> plans;
    static thread_local GemmPlan* last=nullptr;
    static thread_local CacheKey last_key;
    CacheKey key{device,stream,m,n,k,o.transpose_left,o.transpose_right,
                 static_cast<int>(o.precision),static_cast<int>(o.epilogue),o.workspace_limit_bytes};
    if(last && key==last_key) return *last;
    auto it=plans.find(key);
    if(it==plans.end()) {
        // Only cold construction needs to change the library execution context.
        // Warm raw dispatch already supplies its stream and skips repeated queries.
        auto old=compute_stream();
        if(stream!=old)detail::set_compute_stream(stream);
        std::unique_ptr<GemmPlan> plan;
        try { plan=std::make_unique<GemmPlan>(m,n,k,o); }
        catch(...) {if(stream!=old)detail::set_compute_stream(old);throw;}
        if(stream!=old)detail::set_compute_stream(old);
        if(plans.size()>=128) {last=nullptr;plans.erase(plans.begin());}
        it=plans.emplace(key,std::move(plan)).first;
    }
    last=it->second.get();last_key=key;
    return *last;
}
}
namespace detail {
struct GemmPlanAccess {
    static void run(GemmPlan& plan,const float* left,const float* right,float* output,const float* bias) {
        plan.impl_->run_ptr(left,right,output,1,0,bias);
    }
};
namespace gemm {
void gemm_cached_raw(const float* left,const float* right,float* output,const float* bias,
                     int m,int n,int k,Epilogue epi,cudaStream_t stream) {
    if(m<0||n<0||k<0)throw InvalidArgumentError("Negative GEMM dimensions");
    GemmOptions options;options.precision=GemmPrecision::tf32;
    options.epilogue=static_cast<GemmEpilogue>(static_cast<int>(epi));
    GemmPlanAccess::run(cached(m,n,k,options,current_device(),stream),left,right,output,bias);
}
}
}
void gemm_into(const Matrix& left,const Matrix& right,Matrix& output,GemmOptions o,float alpha,float beta,const Matrix* bias) {
    const auto m=o.transpose_left?left.cols():left.rows(), k=o.transpose_left?left.rows():left.cols();
    const auto n=o.transpose_right?right.rows():right.cols();
    if(k!=(o.transpose_right?right.cols():right.rows())) throw ShapeMismatchError("GEMM inner dimension mismatch");
    cached(m,n,k,o,current_device(),compute_stream()).execute(left,right,output,alpha,beta,bias);
}
Matrix gemm(const Matrix& left,const Matrix& right,GemmOptions o,float alpha,const Matrix* bias) {
    const auto m=o.transpose_left?left.cols():left.rows(), n=o.transpose_right?right.rows():right.cols();
    checked_dim(m);checked_dim(n);
    if((o.transpose_left?left.rows():left.cols())!=(o.transpose_right?right.cols():right.rows()))
        throw ShapeMismatchError("GEMM inner dimension mismatch");
    Matrix output(m,n,MemoryMode::device_only);
    gemm_into(left,right,output,o,alpha,0,bias);
    return output;
}
} // namespace matrix_pro
