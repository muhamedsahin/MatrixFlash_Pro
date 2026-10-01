// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Planın descriptor sahipliği, heuristic seçimi ve allocation-free execute.

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
          compute(o.precision == GemmPrecision::tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                                                     : CUBLAS_COMPUTE_32F_PEDANTIC) {
        if (o.precision != GemmPrecision::fp32 && o.precision != GemmPrecision::tf32)
            throw InvalidArgumentError("Invalid GEMM precision policy");
        if (static_cast<int>(o.epilogue) < 0 || static_cast<int>(o.epilogue) > 3)
            throw InvalidArgumentError("Invalid GEMM epilogue");
        require_not_capturing(stream);
        if (m == 0 || n == 0 || k == 0)
            return;
        env = environment(device, stream);
        workspace_bytes = std::min(o.workspace_limit_bytes, env->bytes);
        if (!env->lt)
            return;
        try {
            check_blas(cublasLtMatmulDescCreate(&op, compute, CUDA_R_32F), "plan descriptor");
            const cublasOperation_t ta = o.transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N;
            const cublasOperation_t tb = o.transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N;
            check_blas(
                cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)),
                "plan transpose B");
            check_blas(
                cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)),
                "plan transpose A");
            const auto epilogue = lt_epilogue(o.epilogue);
            check_blas(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                                      sizeof(epilogue)),
                       "plan epilogue");
            const int ar = o.transpose_right ? k : n, ac = o.transpose_right ? n : k;
            const int br = o.transpose_left ? m : k, bc = o.transpose_left ? k : m;
            check_blas(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, ar, ac, ar), "plan B layout");
            check_blas(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, br, bc, br), "plan A layout");
            check_blas(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n), "plan C layout");
            cublasLtMatmulPreference_t pref = nullptr;
            check_blas(cublasLtMatmulPreferenceCreate(&pref), "plan preference");
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes, sizeof(workspace_bytes));
            const auto status = cublasLtMatmulAlgoGetHeuristic(env->lt, op, a, b, c, c, pref,
                                                               static_cast<int>(candidates.size()),
                                                               candidates.data(), &count);
            cublasLtMatmulPreferenceDestroy(pref);
            if (status != CUBLAS_STATUS_SUCCESS)
                count = 0;
            for (int i = 0; i < count; ++i)
                if (candidates[i].state == CUBLAS_STATUS_SUCCESS) {
                    selected = i;
                    break;
                }
        } catch (...) {
            destroy_descriptors();
            throw;
        }
    }
    void destroy_descriptors() noexcept {
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (op)
            cublasLtMatmulDescDestroy(op);
        c = nullptr;
        b = nullptr;
        a = nullptr;
        op = nullptr;
    }
    ~Impl() { destroy_descriptors(); }
    void validate(const Matrix& left, const Matrix& right, const Matrix& output,
                  const Matrix* bias) const {
        if (std::this_thread::get_id() != owner || current_device() != device ||
            compute_stream() != stream)
            throw InvalidArgumentError(
                "GEMM plan must execute on its creating thread, device and stream");
        if (left.rows() != static_cast<std::size_t>(options.transpose_left ? k : m) ||
            left.cols() != static_cast<std::size_t>(options.transpose_left ? m : k) ||
            right.rows() != static_cast<std::size_t>(options.transpose_right ? n : k) ||
            right.cols() != static_cast<std::size_t>(options.transpose_right ? k : n) ||
            output.rows() != static_cast<std::size_t>(m) ||
            output.cols() != static_cast<std::size_t>(n))
            throw ShapeMismatchError("GEMM plan operand shape mismatch");
        if (!output.empty() && (output.device_data() == left.device_data() ||
                                output.device_data() == right.device_data()))
            throw InvalidArgumentError("GEMM output cannot alias an input");
        if (options.epilogue != GemmEpilogue::none) {
            if (!bias || bias->size() != static_cast<std::size_t>(n))
                throw ShapeMismatchError("GEMM bias length must equal N");
            if (!output.empty() && output.device_data() == bias->device_data())
                throw InvalidArgumentError("GEMM output cannot alias bias");
        }
    }
    cublasStatus_t run_lt_ptr(const float* left, const float* right, const float* input,
                              float* output, float alpha, float beta, const float* bias,
                              int index) {
        if (bias && options.epilogue != GemmEpilogue::none) {
            const auto status = cublasLtMatmulDescSetAttribute(
                op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
            if (status != CUBLAS_STATUS_SUCCESS)
                return status;
        }
        return cublasLtMatmul(env->lt, op, &alpha, right, a, left, b, &beta, input, c, output, c,
                              &candidates[index].algo, env->workspace, workspace_bytes, stream);
    }
    cublasStatus_t run_lt(const Matrix& left, const Matrix& right, const float* input,
                          float* output, float alpha, float beta, const Matrix* bias, int index) {
        return run_lt_ptr(left.device_data(), right.device_data(), input, output, alpha, beta,
                          bias ? bias->device_data() : nullptr, index);
    }
    void run_ptr(const float* left, const float* right, float* output, float alpha, float beta,
                 const float* bias) {
        if (m == 0 || n == 0)
            return;
        if (k == 0 || alpha == 0) {
            const auto total = static_cast<std::size_t>(m) * n;
            scale_output<<<std::min<std::size_t>((total + 255) / 256, 4096), 256, 0, stream>>>(
                output, total, beta);
            checkCuda(cudaGetLastError(), "GEMM scale output");
        } else {
            if (selected >= 0 && run_lt_ptr(left, right, output, output, alpha, beta, bias,
                                            selected) == CUBLAS_STATUS_SUCCESS)
                return;
            selected = -1;
            check_blas(cublasGemmEx(env->blas, options.transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
                                    options.transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k,
                                    &alpha, right, CUDA_R_32F, options.transpose_right ? k : n,
                                    left, CUDA_R_32F, options.transpose_left ? m : k, &beta, output,
                                    CUDA_R_32F, n, compute, CUBLAS_GEMM_DEFAULT),
                       "plan fallback GEMM");
        }
        detail::gemm::apply_epilogue(output, bias, m, n, fallback_epilogue(options.epilogue),
                                     stream);
    }
};
