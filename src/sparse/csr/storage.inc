// Dahili uygulama parçası: src/sparse/sparse.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Kopya/taşıma, host erişimi ve dense çıkış.

SparseCSR::SparseCSR(const SparseCSR& other)
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_), row_ptr_(nullptr, release_int),
      col_idx_(nullptr, release_int), values_(nullptr, release_float) {
    if (nnz_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        col_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
        checkCuda(cudaMemcpyAsync(row_ptr_.get(), other.row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "csr copy");
        checkCuda(cudaMemcpyAsync(col_idx_.get(), other.col_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "csr copy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "csr copy");
    } else if (rows_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        checkCuda(cudaMemcpyAsync(row_ptr_.get(), other.row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "csr copy");
    }
}

SparseCSR& SparseCSR::operator=(const SparseCSR& other) {
    if (this != &other)
        *this = SparseCSR(other);
    return *this;
}

SparseCSR::SparseCSR(SparseCSR&& other) noexcept
    : rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_), row_ptr_(std::move(other.row_ptr_)),
      col_idx_(std::move(other.col_idx_)), values_(std::move(other.values_)) {
    other.rows_ = other.cols_ = other.nnz_ = 0;
}

SparseCSR& SparseCSR::operator=(SparseCSR&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        nnz_ = other.nnz_;
        row_ptr_ = std::move(other.row_ptr_);
        col_idx_ = std::move(other.col_idx_);
        values_ = std::move(other.values_);
        other.rows_ = other.cols_ = other.nnz_ = 0;
    }
    return *this;
}

float SparseCSR::sparsity() const noexcept {
    if (rows_ == 0 || cols_ == 0)
        return 0.0f;
    return 1.0f - static_cast<float>(nnz_) / static_cast<float>(rows_ * cols_);
}

std::vector<int> SparseCSR::host_row_ptr() const {
    std::vector<int> host(rows_ + 1, 0);
    if (rows_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), row_ptr_.get(), (rows_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

std::vector<int> SparseCSR::host_col_idx() const {
    std::vector<int> host(nnz_, 0);
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), col_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

std::vector<float> SparseCSR::host_values() const {
    std::vector<float> host(nnz_, 0.0f);
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(host.data(), values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToHost, compute_stream()),
                  "csr read");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr sync");
    }
    return host;
}

Matrix SparseCSR::to_dense() const {
    Matrix out(rows_, cols_);
    if (nnz_ == 0) {
        out.upload();
        return out;
    }
    const std::vector<int> rp = host_row_ptr();
    const std::vector<int> ci = host_col_idx();
    const std::vector<float> vs = host_values();
    for (std::size_t r = 0; r < rows_; ++r)
        for (int j = rp[r]; j < rp[r + 1]; ++j)
            out.data()[r * cols_ + static_cast<std::size_t>(ci[j])] += vs[j];
    out.upload();
    return out;
}

void SparseCSR::synchronize() const {
    matrix_pro::synchronize();
}
