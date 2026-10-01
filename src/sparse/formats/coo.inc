// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: COO depolama ve coalesce; tekrar eden koordinatlar toplanır.

// SparseCOO Implementation
SparseCOO::SparseCOO(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz),
      row_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      col_idx_(static_cast<int*>(allocate_device_memory(nnz * sizeof(int))), release_int),
      values_(static_cast<float*>(allocate_device_memory(nnz * sizeof(float))), release_float) {
}

SparseCOO::~SparseCOO() = default;

SparseCOO::SparseCOO(const SparseCOO& other) : SparseCOO(other.rows_, other.cols_, other.nnz_) {
    if (nnz_ > 0) {
        checkCuda(cudaMemcpyAsync(row_idx_.get(), other.row_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
        checkCuda(cudaMemcpyAsync(col_idx_.get(), other.col_idx_.get(), nnz_ * sizeof(int),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
        checkCuda(cudaMemcpyAsync(values_.get(), other.values_.get(), nnz_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice, compute_stream()),
                  "memcpy");
    }
}

SparseCOO& SparseCOO::operator=(const SparseCOO& other) {
    if (this != &other) {
        *this = SparseCOO(other);
    }
    return *this;
}

SparseCOO::SparseCOO(SparseCOO&& other) noexcept = default;
SparseCOO& SparseCOO::operator=(SparseCOO&& other) noexcept = default;

const int* SparseCOO::row_indices() const noexcept {
    return row_idx_.get();
}
const int* SparseCOO::col_indices() const noexcept {
    return col_idx_.get();
}
const float* SparseCOO::values() const noexcept {
    return values_.get();
}
float SparseCOO::sparsity() const {
    return 1.0f - static_cast<float>(nnz_) / (rows_ * cols_);
}

SparseCOO SparseCOO::from_dense(const Matrix& dense, float threshold) {
    std::vector<float> host_dense(dense.rows() * dense.cols());
    checkCuda(copy_sync(host_dense.data(), dense.device_data(), host_dense.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
              "from_dense memcpy");

    std::vector<int> row_idx, col_idx;
    std::vector<float> values;

    for (std::size_t r = 0; r < dense.rows(); ++r) {
        for (std::size_t c = 0; c < dense.cols(); ++c) {
            float val = host_dense[r * dense.cols() + c];
            if (std::abs(val) > threshold) {
                row_idx.push_back(r);
                col_idx.push_back(c);
                values.push_back(val);
            }
        }
    }
    return from_triplets(dense.rows(), dense.cols(), row_idx, col_idx, values);
}

SparseCOO SparseCOO::from_triplets(std::size_t rows, std::size_t cols,
                                   const std::vector<int>& row_idx, const std::vector<int>& col_idx,
                                   const std::vector<float>& values) {
    if (row_idx.size() != values.size() || col_idx.size() != values.size())
        throw ShapeMismatchError("COO triplet lengths differ");
    for (std::size_t i = 0; i < values.size(); ++i)
        if (row_idx[i] < 0 || col_idx[i] < 0 || static_cast<std::size_t>(row_idx[i]) >= rows ||
            static_cast<std::size_t>(col_idx[i]) >= cols)
            throw OutOfRangeError("COO triplet index out of range");
    SparseCOO coo(rows, cols, values.size());
    checkCuda(copy_sync(coo.row_idx_.get(), row_idx.data(), values.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
              "memcpy");
    checkCuda(copy_sync(coo.col_idx_.get(), col_idx.data(), values.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
              "memcpy");
    checkCuda(copy_sync(coo.values_.get(), values.data(), values.size() * sizeof(float),
                        cudaMemcpyHostToDevice),
              "memcpy");
    return coo;
}

Matrix SparseCOO::to_dense() const {
    Matrix dense(rows_, cols_);
    dense.fill(0.0f);
    if (nnz_ > 0) {
        coo_to_dense_kernel<<<detail::capped_grid(nnz_, 256), 256, 0, compute_stream()>>>(
            row_idx_.get(), col_idx_.get(), values_.get(), dense.device_data(), cols_, nnz_);
    }
    dense.mark_host_stale();
    return dense;
}

SparseCSR SparseCOO::to_csr() const {
    return coo_to_csr(*this);
}

SparseCOO SparseCOO::coalesce() const {
    // For now, assume sorted or simply download, sort, merge, upload.
    // A robust implementation would use thrust::sort_by_key on device.
    // Here we download for simplicity of merging.
    std::vector<int> h_r(nnz_), h_c(nnz_);
    std::vector<float> h_v(nnz_);
    checkCuda(copy_sync(h_r.data(), row_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost),
              "memcpy");
    checkCuda(copy_sync(h_c.data(), col_idx_.get(), nnz_ * sizeof(int), cudaMemcpyDeviceToHost),
              "memcpy");
    checkCuda(copy_sync(h_v.data(), values_.get(), nnz_ * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy");

    struct Entry {
        int r, c;
        float v;
    };
    std::vector<Entry> entries(nnz_);
    for (size_t i = 0; i < nnz_; ++i)
        entries[i] = {h_r[i], h_c[i], h_v[i]};

    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.r != b.r)
            return a.r < b.r;
        return a.c < b.c;
    });

    std::vector<int> out_r, out_c;
    std::vector<float> out_v;
    for (const auto& e : entries) {
        if (!out_r.empty() && out_r.back() == e.r && out_c.back() == e.c) {
            out_v.back() += e.v;
        } else {
            out_r.push_back(e.r);
            out_c.push_back(e.c);
            out_v.push_back(e.v);
        }
    }
    return from_triplets(rows_, cols_, out_r, out_c, out_v);
}
