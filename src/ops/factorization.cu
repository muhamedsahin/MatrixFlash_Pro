// Reusable LU with partial pivoting (cuSOLVER DnSgetrf / DnSgetrs).
//
// Construction factors A once. Row-major device data is transposed into a
// column-major factor buffer because cuSOLVER's dense GETRF expects Fortran
// layout. Pivots, info and a RHS scratch of (n * max_rhs) floats stay alive
// for later solves. solve_into() copies the right-hand side into that scratch,
// calls getrs, and writes the solution back. It does not allocate when the
// RHS column count is within the capacity passed to the constructor.
//
// The factor is bound to the host thread, CUDA device and compute stream that
// created it. A dedicated cusolverDn handle is stored so a later Pipeline
// stream change cannot silently rebind the solve.
//
// slogdet is computed on the host after one sync: sign is the product of the
// diagonal signs and the pivot-row swaps (cuSOLVER pivots are 1-based), and
// log_abs_det accumulates log(|d_ii|) in double so a huge or tiny FP32 product
// does not overflow. A zero pivot (info > 0) is singular: sign 0 and
// log_abs_det = -infinity. A 0×0 matrix is defined as determinant 1.
// Construction must happen before CUDA graph capture because of that sync.
//
// The destructor frees the solver and the owned buffers on the allocating
// device, then the Buffer members see null pointers and do not free twice.

#include "matrix_pro/ops/factorization.hpp"
#include "matrix_pro/core/cuda_utils.hpp"
#include <cusolverDn.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

namespace matrix_pro {
namespace {
void solver_check(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) throw SolverError("Reusable LU: cuSOLVER status " + std::to_string(status));
}
int dim(std::size_t n) {
    if(n>static_cast<std::size_t>(std::numeric_limits<int>::max())) throw InvalidArgumentError("LU dimension exceeds int32");
    return static_cast<int>(n);
}
int square_dim(const Matrix& a) {
    if(a.rows()!=a.cols())throw ShapeMismatchError("LU requires square coefficients");
    cudaStreamCaptureStatus status;
    checkCuda(cudaStreamIsCapturing(compute_stream(),&status),"LU capture query");
    if(status!=cudaStreamCaptureStatusNone)
        throw InvalidArgumentError("Construct LU factors before CUDA Graph capture");
    return dim(a.rows());
}
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t bytes=0) { if(bytes) checkCuda(cudaMalloc(&data,bytes),"LU allocation"); }
    ~Buffer() { if(data)cudaFree(data); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
__global__ void layout_copy(const float* source,float* target,std::size_t rows,std::size_t cols,bool to_column) {
    for(std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
        i<rows*cols;i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        auto r=i/cols,c=i%cols;
        if(to_column)target[c*rows+r]=source[i]; else target[i]=source[c*rows+r];
    }
}
__global__ void diagonal_copy(const float* lu,float* diagonal,int n) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
        diagonal[i]=lu[static_cast<std::size_t>(i)*n+i];
}
unsigned blocks(std::size_t n) { return static_cast<unsigned>(std::min<std::size_t>((n+255)/256,4096)); }
std::size_t checked_square_bytes(std::size_t n) {
    dim(n);
    if(n && n>std::numeric_limits<std::size_t>::max()/sizeof(float)/n) throw InvalidArgumentError("LU allocation overflow");
    return n*n*sizeof(float);
}
}
struct LUFactorization::Impl {
    int n, capacity, device=current_device();
    cudaStream_t stream=compute_stream();
    std::thread::id owner=std::this_thread::get_id();
    // Use a dedicated handle so later Pipeline changes cannot rebind it.
    cusolverDnHandle_t solver=nullptr;
    Buffer factors, pivots, info, rhs;
    int factor_info=0;
    SLogDetResult determinant;
    Impl(const Matrix& a,std::size_t max_rhs) : n(square_dim(a)),capacity(dim(max_rhs)),
        factors(checked_square_bytes(a.rows())),pivots(a.rows()*sizeof(int)),info(sizeof(int)),
        rhs(a.rows()*max_rhs*sizeof(float)) {
        if(a.rows()!=a.cols())throw ShapeMismatchError("LU requires square coefficients");
        if(max_rhs==0)throw InvalidArgumentError("LU max_rhs must be positive");
        if(!n)return;
        try {
            solver_check(cusolverDnCreate(&solver));solver_check(cusolverDnSetStream(solver,stream));
            layout_copy<<<blocks(a.size()),256,0,stream>>>(a.device_data(),static_cast<float*>(factors.data),n,n,true);
            checkCuda(cudaGetLastError(),"LU layout copy");
            int lwork=0;
            solver_check(cusolverDnSgetrf_bufferSize(solver,n,n,static_cast<float*>(factors.data),n,&lwork));
            Buffer work(static_cast<std::size_t>(lwork)*sizeof(float)), diagonal(n*sizeof(float));
            solver_check(cusolverDnSgetrf(solver,n,n,static_cast<float*>(factors.data),n,
                static_cast<float*>(work.data),static_cast<int*>(pivots.data),static_cast<int*>(info.data)));
            diagonal_copy<<<blocks(n),256,0,stream>>>(static_cast<float*>(factors.data),static_cast<float*>(diagonal.data),n);
            checkCuda(cudaGetLastError(),"LU diagonal copy");
            std::vector<float> diag(n); std::vector<int> pivot(n);
            checkCuda(cudaMemcpyAsync(&factor_info,info.data,sizeof(int),cudaMemcpyDeviceToHost,stream),"LU status read");
            checkCuda(cudaMemcpyAsync(diag.data(),diagonal.data,n*sizeof(float),cudaMemcpyDeviceToHost,stream),"LU diagonal read");
            checkCuda(cudaMemcpyAsync(pivot.data(),pivots.data,n*sizeof(int),cudaMemcpyDeviceToHost,stream),"LU pivots read");
            checkCuda(cudaStreamSynchronize(stream),"LU factor sync");
            if(factor_info<0)throw SolverError("LU invalid cuSOLVER argument");
            if(factor_info>0) determinant={0,-std::numeric_limits<double>::infinity()};
            else for(int i=0;i<n;++i) {
                if(!std::isfinite(diag[i]))throw SolverError("LU nonfinite factor");
                if(diag[i]<0)determinant.sign=-determinant.sign;
                if(pivot[i]!=i+1)determinant.sign=-determinant.sign;
                determinant.log_abs_det+=std::log(std::abs(static_cast<double>(diag[i])));
            }
        } catch(...) { if(solver)cusolverDnDestroy(solver);solver=nullptr;throw; }
    }
    ~Impl() {
        int old=current_device();if(old!=device)cudaSetDevice(device);
        // All owned buffers must be freed on their allocating device.
        if(solver)cusolverDnDestroy(solver);
        for(Buffer* b:{&rhs,&info,&pivots,&factors}) { if(b->data)cudaFree(b->data);b->data=nullptr; }
        if(old!=device)cudaSetDevice(old);
    }
    void validate(const Matrix& b,const Matrix& x)const {
        if(owner!=std::this_thread::get_id() || device!=current_device() || stream!=compute_stream())
            throw InvalidArgumentError("LU must execute on its creating thread/device/stream");
        if(b.rows()!=static_cast<std::size_t>(n)||x.rows()!=b.rows()||x.cols()!=b.cols())throw ShapeMismatchError("LU RHS/output mismatch");
        if(b.cols()>static_cast<std::size_t>(capacity))throw InvalidArgumentError("LU RHS exceeds preallocated capacity");
        if(factor_info>0)throw SolverError("LU solve: singular coefficients");
    }
};
LUFactorization::LUFactorization(const Matrix& a,std::size_t capacity) : impl_(std::make_unique<Impl>(a,capacity)) {}
LUFactorization::~LUFactorization()=default;
LUFactorization::LUFactorization(LUFactorization&&) noexcept=default;
LUFactorization& LUFactorization::operator=(LUFactorization&&) noexcept=default;
std::size_t LUFactorization::size()const noexcept{return impl_?impl_->n:0;}
bool LUFactorization::singular()const noexcept{return impl_&&impl_->factor_info>0;}
SLogDetResult LUFactorization::slogdet()const{if(!impl_)throw InvalidArgumentError("Moved-from LU factorization");return impl_->determinant;}
void LUFactorization::solve_into(const Matrix& b,Matrix& x,bool transpose) {
    if(!impl_)throw InvalidArgumentError("Moved-from LU factorization");
    impl_->validate(b,x);if(b.empty())return;
    auto& f=*impl_;
    layout_copy<<<blocks(b.size()),256,0,f.stream>>>(b.device_data(),static_cast<float*>(f.rhs.data),f.n,b.cols(),true);
    checkCuda(cudaGetLastError(),"LU RHS layout");
    solver_check(cusolverDnSgetrs(f.solver,transpose?CUBLAS_OP_T:CUBLAS_OP_N,f.n,dim(b.cols()),
        static_cast<float*>(f.factors.data),f.n,static_cast<int*>(f.pivots.data),static_cast<float*>(f.rhs.data),f.n,static_cast<int*>(f.info.data)));
    layout_copy<<<blocks(b.size()),256,0,f.stream>>>(static_cast<float*>(f.rhs.data),x.device_data(),f.n,b.cols(),false);
    checkCuda(cudaGetLastError(),"LU output layout");x.mark_host_stale();
}
Matrix LUFactorization::solve(const Matrix& b,bool transpose) {
    Matrix x(b.rows(),b.cols(),MemoryMode::device_only);solve_into(b,x,transpose);return x;
}
SLogDetResult slogdet(const Matrix& a){return LUFactorization(a).slogdet();}
} // namespace matrix_pro
