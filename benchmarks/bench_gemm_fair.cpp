// Independent, same-policy GEMM measurement. No allocations/transfers in timing.
#include "matrix_pro/matrix_pro.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace matrix_pro;
namespace {
void blas_check(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw CudaError("benchmark cuBLAS status " + std::to_string(s));
}
struct RawBlas {
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt = nullptr;
    void* work = nullptr;
    size_t bytes = 64u << 20;
    RawBlas() {
        blas_check(cublasCreate(&handle));
        blas_check(cublasSetStream(handle, compute_stream()));
        checkCuda(cudaMalloc(&work, bytes), "raw workspace");
        blas_check(cublasSetWorkspace(handle, work, bytes));
        blas_check(cublasLtCreate(&lt));
    }
    ~RawBlas() { cublasDestroy(handle); cublasLtDestroy(lt); cudaFree(work); }
};
struct RawLtPlan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulAlgo_t algo{};
    RawLtPlan(RawBlas& ctx, int m, int n, int k, cublasComputeType_t policy) {
        blas_check(cublasLtMatmulDescCreate(&op, policy, CUDA_R_32F));
        blas_check(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, n, k, n));
        blas_check(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, k, m, k));
        blas_check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n));
        blas_check(cublasLtMatmulPreferenceCreate(&pref));
        blas_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &ctx.bytes, sizeof(ctx.bytes)));
        cublasLtMatmulHeuristicResult_t choices[8]{};
        int count = 0;
        blas_check(cublasLtMatmulAlgoGetHeuristic(ctx.lt, op, a, b, c, c, pref, 8, choices, &count));
        if (count == 0) throw CudaError("raw cuBLASLt has no heuristic");
        algo = choices[0].algo;
    }
    ~RawLtPlan() {
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(c);
        cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(a); cublasLtMatmulDescDestroy(op);
    }
};
struct Shape { int m, n, k; };
struct Measurement {
    std::string engine, policy;
    Shape shape;
    std::vector<double> device_ms, wall_ms;
    double max_error = 0;
};
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto mid = values.size()/2;
    return values.size()%2 ? values[mid] : (values[mid-1]+values[mid])/2;
}
double deviation(const std::vector<double>& values) {
    double avg = std::accumulate(values.begin(), values.end(), 0.0)/values.size(), sum = 0;
    for (double x : values) sum += (x-avg)*(x-avg);
    return std::sqrt(sum/values.size());
}
void validate(const Matrix& a, const Matrix& b, Matrix& c, bool tf32, Measurement& row) {
    c.download();
    const auto& av = a.data(); const auto& bv = b.data(); const auto& cv = c.data();
    const auto s = row.shape;
    // Independent double-precision dot products at deterministic positions.
    const size_t count = std::min<size_t>(s.m*s.n, 97);
    for (size_t q = 0; q < count; ++q) {
        const size_t idx = q*(static_cast<size_t>(s.m)*s.n-1)/std::max<size_t>(1,count-1);
        const size_t i = idx/s.n, j = idx%s.n;
        double reference = 0, scale = 0;
        for (int k = 0; k < s.k; ++k) {
            double product = static_cast<double>(av[i*s.k+k])*bv[k*s.n+j];
            reference += product; scale += std::abs(product);
        }
        const double error = std::abs(cv[idx]-reference);
        row.max_error = std::max(row.max_error, error);
        const double tolerance = 1e-5 + (tf32 ? 0.002 : 2e-6)*scale;
        if (!std::isfinite(cv[idx]) || error > tolerance)
            throw InvalidArgumentError(row.engine+" correctness failed at "+std::to_string(idx));
    }
}
void measure(const std::function<void()>& fn, Measurement& row, int warmup, int repeats, int batch) {
    for (int i=0; i<warmup; ++i) fn();
    synchronize();
    cudaEvent_t start = nullptr, stop = nullptr;
    checkCuda(cudaEventCreate(&start), "event create");
    checkCuda(cudaEventCreate(&stop), "event create");
    for (int r=0; r<repeats; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        checkCuda(cudaEventRecord(start, compute_stream()), "record");
        for (int i=0; i<batch; ++i) fn();
        checkCuda(cudaEventRecord(stop, compute_stream()), "record");
        checkCuda(cudaEventSynchronize(stop), "sync");
        auto t1 = std::chrono::steady_clock::now();
        float elapsed = 0;
        checkCuda(cudaEventElapsedTime(&elapsed,start,stop), "elapsed");
        row.device_ms.push_back(elapsed/batch);
        row.wall_ms.push_back(std::chrono::duration<double,std::milli>(t1-t0).count()/batch);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
}
void array(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (size_t i=0;i<values.size();++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
}
int main(int argc, char** argv) {
    try {
        std::string path = "benchmarks/results/fair_gemm.json", tag = "unspecified";
        int repeats = 30, batch = 32, warmup = 10;
        for (int i=1;i<argc;++i) {
            const std::string flag=argv[i];
            if (i+1>=argc) throw InvalidArgumentError("missing flag value");
            if (flag=="--output") path=argv[++i];
            else if(flag=="--tag") tag=argv[++i];
            else if(flag=="--repeats") repeats=std::stoi(argv[++i]);
            else if(flag=="--batch") batch=std::stoi(argv[++i]);
            else if(flag=="--warmup") warmup=std::stoi(argv[++i]);
            else throw InvalidArgumentError("unknown flag "+flag);
        }
        if(repeats<1||batch<1||warmup<1) throw InvalidArgumentError("counts must be positive");
        std::vector<Shape> shapes = {{16,16,16},{32,32,32},{64,64,64},{128,128,128},
            {256,256,256},{512,512,512},{1024,1024,1024},{2048,2048,2048},
            {31,47,19},{127,257,63},{64,1024,512},{1024,64,512},{1,1024,4096},{1024,1,4096}};
        std::vector<Measurement> rows;
        RawBlas raw;
        std::mt19937 rng(20261001);
        std::uniform_real_distribution<float> dist(-0.5f,0.5f);
        for(auto s:shapes) {
            std::vector<float> av(s.m*s.k), bv(s.k*s.n);
            for(auto& v:av) v=dist(rng); for(auto& v:bv) v=dist(rng);
            Matrix a(s.m,s.k,av), b(s.k,s.n,bv), c(s.m,s.n,MemoryMode::device_only);
            auto run = [&](std::string engine, std::string policy, const std::function<void()>& fn, int operations=1) {
                Measurement row{engine,policy,s};
                const bool reduced = policy=="tf32" || (policy=="legacy_tf32" && s.m>1 && s.n>4 &&
                    (s.m>64 || s.n>64 || s.k>64));
                fn(); c.mark_host_stale(); validate(a,b,c,reduced,row);
                measure(fn,row,warmup,repeats,operations==1?batch:1);
                for(auto& value:row.device_ms)value/=operations;
                for(auto& value:row.wall_ms)value/=operations;
                c.mark_host_stale(); validate(a,b,c,reduced,row);
                std::cout << engine << ' ' << policy << ' ' << s.m << 'x' << s.n << 'x' << s.k
                          << " : " << median(row.device_ms)*1000 << " us\n";
                rows.push_back(std::move(row));
            };
            run("matrixflash_default","legacy_tf32",[&]{multiply_into(a,b,c);});
            for(bool fast:{false,true}) {
                auto policy=fast?CUBLAS_COMPUTE_32F_FAST_TF32:CUBLAS_COMPUTE_32F_PEDANTIC;
                const std::string name=fast?"tf32":"fp32";
                GemmOptions options;options.precision=fast?GemmPrecision::tf32:GemmPrecision::fp32;
                GemmPlan plan(s.m,s.n,s.k,options);
                run("matrixflash_plan",name,[&]{plan.execute(a,b,c);});
                plan.tune(a,b,c);
                run("matrixflash_tuned",name,[&]{plan.execute(a,b,c);});
                // CUDA Graph removes per-GEMM host submission gaps; separate scope.
                plan.execute(a,b,c);synchronize();
                CudaGraph graph;graph.begin_capture();
                for(int i=0;i<batch;++i)plan.execute(a,b,c);
                graph.end_capture();
                run("matrixflash_graph_batch",name,[&]{graph.replay();},batch);
                const float alpha=1,beta=0;
                run("raw_cublas",name,[&]{blas_check(cublasGemmEx(raw.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                    s.n,s.m,s.k,&alpha,b.device_data(),CUDA_R_32F,s.n,a.device_data(),CUDA_R_32F,s.k,
                    &beta,c.device_data(),CUDA_R_32F,s.n,policy,CUBLAS_GEMM_DEFAULT));});
                RawLtPlan lt(raw,s.m,s.n,s.k,policy);
                run("raw_cublaslt",name,[&]{blas_check(cublasLtMatmul(raw.lt,lt.op,&alpha,b.device_data(),lt.a,
                    a.device_data(),lt.b,&beta,c.device_data(),lt.c,c.device_data(),lt.c,&lt.algo,
                    raw.work,raw.bytes,compute_stream()));});
                CudaGraph raw_graph;raw_graph.begin_capture();
                for(int i=0;i<batch;++i)blas_check(cublasLtMatmul(raw.lt,lt.op,&alpha,b.device_data(),lt.a,
                    a.device_data(),lt.b,&beta,c.device_data(),lt.c,c.device_data(),lt.c,&lt.algo,
                    raw.work,raw.bytes,compute_stream()));
                raw_graph.end_capture();
                run("raw_cublaslt_graph_batch",name,[&]{raw_graph.replay();},batch);
            }
        }
        std::ofstream out(path);
        if(!out) throw IoError("cannot write "+path);
        cudaDeviceProp prop{}; checkCuda(cudaGetDeviceProperties(&prop,current_device()),"properties");
        int runtime=0,driver=0; cudaRuntimeGetVersion(&runtime); cudaDriverGetVersion(&driver);
        out << std::setprecision(12) << "{\"schema_version\":1,\"tag\":\"" << tag << "\",\"gpu\":\"" << prop.name
            << "\",\"sm\":" << prop.major*10+prop.minor << ",\"cuda_runtime\":" << runtime
            << ",\"cuda_driver\":" << driver << ",\"warmup\":" << warmup << ",\"repeats\":" << repeats
            << ",\"batch\":" << batch << ",\"seed\":20261001,\"timing\":\"preallocated CUDA stream elapsed; includes host submission gaps; wall includes batch synchronization; no transfers\",\"results\":[";
        for(size_t i=0;i<rows.size();++i) {
            const auto& r=rows[i]; if(i)out << ',';
            out << "\n{\"engine\":\"" << r.engine << "\",\"policy\":\"" << r.policy << "\",\"m\":" << r.shape.m
                << ",\"n\":" << r.shape.n << ",\"k\":" << r.shape.k
                << ",\"device_median_ms\":" << median(r.device_ms) << ",\"wall_median_ms\":" << median(r.wall_ms)
                << ",\"device_stddev_ms\":" << deviation(r.device_ms) << ",\"max_abs_error_sampled\":" << r.max_error
                << ",\"gflops\":" << 2.0*r.shape.m*r.shape.n*r.shape.k/(median(r.device_ms)*1e6)
                << ",\"device_samples_ms\":"; array(out,r.device_ms);
            out << ",\"wall_samples_ms\":"; array(out,r.wall_ms); out << '}';
        }
        out << "\n]}\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
