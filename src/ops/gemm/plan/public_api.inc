// Dahili uygulama parçası: src/ops/gemm/gemm_plan.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: GemmPlan yaşam döngüsü, durum sorguları ve execute erişimi.

GemmPlan::GemmPlan(std::size_t m, std::size_t n, std::size_t k, GemmOptions o)
    : impl_(std::make_unique<Impl>(m, n, k, o)) {
}
GemmPlan::~GemmPlan() = default;
GemmPlan::GemmPlan(GemmPlan&&) noexcept = default;
GemmPlan& GemmPlan::operator=(GemmPlan&&) noexcept = default;
std::size_t GemmPlan::rows() const noexcept {
    return impl_ ? impl_->m : 0;
}
std::size_t GemmPlan::cols() const noexcept {
    return impl_ ? impl_->n : 0;
}
std::size_t GemmPlan::inner() const noexcept {
    return impl_ ? impl_->k : 0;
}
int GemmPlan::candidate_count() const noexcept {
    return impl_ ? impl_->count : 0;
}
bool GemmPlan::uses_cublaslt() const noexcept {
    return impl_ && impl_->selected >= 0;
}
void GemmPlan::execute(const Matrix& left, const Matrix& right, Matrix& output, float alpha,
                       float beta, const Matrix* bias) {
    if (!impl_)
        throw InvalidArgumentError("Cannot execute a moved-from GEMM plan");
    impl_->validate(left, right, output, bias);
    impl_->run_ptr(left.device_data(), right.device_data(), output.device_data(), alpha, beta,
                   bias ? bias->device_data() : nullptr);
    if (!output.empty())
        output.mark_host_stale();
}
