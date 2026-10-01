// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Scratch üzerinde aday zamanlama; capture sırasında tune yapılmaz.

GemmTuningResult GemmPlan::tune(const Matrix& left, const Matrix& right, const Matrix& output,
                                float alpha, float beta, const Matrix* bias, int repeats) {
    if (!impl_)
        throw InvalidArgumentError("Cannot tune a moved-from GEMM plan");
    impl_->validate(left, right, output, bias);
    if (repeats < 1)
        throw InvalidArgumentError("GEMM tuning repeats must be positive");
    require_not_capturing(impl_->stream);
    GemmTuningResult result;
    if (!impl_->count || output.empty() || impl_->k == 0 || alpha == 0)
        return result;
    Matrix scratch(output.rows(), output.cols(), MemoryMode::device_only);
    cudaEvent_t start = nullptr, stop = nullptr;
    checkCuda(cudaEventCreate(&start), "GEMM tuning event");
    try {
        checkCuda(cudaEventCreate(&stop), "GEMM tuning event");
        float best = std::numeric_limits<float>::infinity();
        for (int i = 0; i < impl_->count; ++i) {
            if (impl_->candidates[i].state != CUBLAS_STATUS_SUCCESS)
                continue;
            auto call = [&] {
                return impl_->run_lt(left, right, output.device_data(), scratch.device_data(),
                                     alpha, beta, bias, i);
            };
            if (call() != CUBLAS_STATUS_SUCCESS)
                continue;
            bool ok = true;
            std::array<float, 3> times{};
            for (auto& time : times) {
                checkCuda(cudaEventRecord(start, impl_->stream), "GEMM tuning start");
                for (int r = 0; r < repeats; ++r)
                    if (call() != CUBLAS_STATUS_SUCCESS) {
                        ok = false;
                        break;
                    }
                checkCuda(cudaEventRecord(stop, impl_->stream), "GEMM tuning stop");
                checkCuda(cudaEventSynchronize(stop), "GEMM tuning sync");
                checkCuda(cudaEventElapsedTime(&time, start, stop), "GEMM tuning elapsed");
                if (!ok)
                    break;
            }
            if (!ok)
                continue;
            std::sort(times.begin(), times.end());
            const float ms = times[1];
            ++result.candidates_tested;
            if (ms < best) {
                best = ms;
                impl_->selected = i;
                result.selected_ms = ms / repeats;
            }
        }
    } catch (...) {
        if (stop)
            cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return result;
}
