// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: CSC depolama, transpose, spmv ve dense multiply.

// SparseCSC Implementation
SparseCSC::SparseCSC(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      col_ptr_(static_cast<int*>(allocate_device_memory((cols + 1) * sizeof(int))), release_int),
      row_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      values_(static_cast<float*>(allocate_device_memory(nnz * sizeof(float))), release_float) {
}

SparseCSC::~SparseCSC() = default;

SparseCSC::SparseCSC(const SparseCSC& other) : SparseCSC(other.rows_, other.cols_, other.nnz_) {
    if (cols_ + 1 > 0) {
        checkCuda(cudaMemcpyAsync(col_ptr_.get(), other.col_ptr_.get(), (cols_ + 1) * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
    }
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(row_idx_.get(), other.row_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
    }
}

SparseCSC& SparseCSC::operator=(const SparseCSC& other) {
    if (this != &other) {
        *this = SparseCSC(other);
    }
    return *this;
}

SparseCSC::SparseCSC(SparseCSC&& other) noexcept = default;
SparseCSC& SparseCSC::operator=(SparseCSC&& other) noexcept = default;

float SparseCSC::sparsity() const {
    return 1.0f - static_cast<float>(nnz_) / (rows_ * cols_);
}

SparseCSC SparseCSC::from_dense(const Matrix& dense, float threshold) {
    std::vector<float> host_dense(dense.rows() * dense.cols());
    checkCuda(copy_sync(host_dense.data(), dense.device_data(), host_dense.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
              "from_dense memcpy");

    std::vector<int> col_ptr(dense.cols() + 1, 0);
    std::vector<int> row_idx;
    std::vector<float> values;

    for (std::size_t c = 0; c < dense.cols(); ++c) {
        for (std::size_t r = 0; r < dense.rows(); ++r) {
            float val = host_dense[r * dense.cols() + c];
            if (std::abs(val) > threshold) {
                row_idx.push_back(r);
                values.push_back(val);
            }
        }
        col_ptr[c + 1] = row_idx.size();
    }

    SparseCSC csc(dense.rows(), dense.cols(), values.size());
    checkCuda(copy_sync(csc.col_ptr_.get(), col_ptr.data(), (dense.cols() + 1) * sizeof(int),
                        cudaMemcpyHostToDevice),
              "memcpy");
    checkCuda(copy_sync(csc.row_idx_.get(), row_idx.data(), values.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
              "memcpy");
    checkCuda(copy_sync(csc.values_.get(), values.data(), values.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
              "memcpy");

    return csc;
}

SparseCSC SparseCSC::from_csr(const SparseCSR& csr) {
    return csr_to_csc(csr);
}

Matrix SparseCSC::to_dense() const {
    Matrix dense(rows_, cols_);
    dense.fill(0.0f);
    std::vector<int> col_ptr(cols_ + 1);
    std::vector<int> row_idx(nnz_);
    std::vector<float> values(nnz_);
    checkCuda(copy_sync(col_ptr.data(), col_ptr_.get(), (cols_ + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost),
              "memcpy");
    checkCuda(copy_sync(row_idx.data(), row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost),
              "memcpy");
    checkCuda(copy_sync(values.data(), values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy");

    std::vector<float> host_dense(rows_ * cols_, 0.0f);
    for (std::size_t c = 0; c < cols_; ++c) {
        for (int j = col_ptr[c]; j < col_ptr[c + 1]; ++j) {
            host_dense[row_idx[j] * cols_ + c] = values[j];
        }
    }
    checkCuda(copy_sync(dense.device_data(), host_dense.data(), rows_ * cols_ * sizeof(float),
                        cudaMemcpyHostToDevice),
              "memcpy");
    dense.mark_host_stale();
    return dense;
}

SparseCSR SparseCSC::to_csr() const {
    return csc_to_csr(*this);
}

SparseCSC SparseCSC::transpose() const {
    SparseCSR csr = to_csr();
    SparseCSC result(cols_, rows_, csr.nnz());
    checkCuda(copy_sync(result.col_ptr_.get(), csr.row_ptr(), (rows_ + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice),
              "CSC transpose ptr");
    checkCuda(copy_sync(result.row_idx_.get(), csr.col_idx(), csr.nnz() * sizeof(int),
                        cudaMemcpyDeviceToDevice),
              "CSC transpose indices");
    checkCuda(copy_sync(result.values_.get(), csr.values(), csr.nnz() * sizeof(float),
                        cudaMemcpyDeviceToDevice),
              "CSC transpose values");
    return result;
}

Matrix SparseCSC::spmv(const Matrix& x) const {
    if (x.rows() != cols_ || x.cols() != 1)
        throw ShapeMismatchError("spmv shape error");
    Matrix y(rows_, 1);
    y.fill(0.0f);
    if (cols_ > 0) {
        csc_spmv_kernel<<<detail::capped_grid(cols_, 256), 256, 0, compute_stream()>>>(
            col_ptr_.get(), row_idx_.get(), values_.get(), x.device_data(), y.device_data(), cols_);
    }
    y.mark_host_stale();
    return y;
}

Matrix SparseCSC::sparse_matmul(const Matrix& B) const {
    // Basic fallback implementation
    Matrix dense = to_dense();
    return dense * B;
}
