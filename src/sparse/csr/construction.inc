// Dahili uygulama parçası: src/sparse/sparse.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: CSR kurma, dense/COO dönüşümü ve boş CSR row pointer sıfırlama.

SparseCSR::SparseCSR(std::size_t rows, std::size_t cols, std::size_t nnz)
    : rows_(rows), cols_(cols), nnz_(nnz), row_ptr_(nullptr, release_int),
      col_idx_(nullptr, release_int), values_(nullptr, release_float) {
    if (rows > INT_MAX || cols > INT_MAX || nnz > INT_MAX)
        throw InvalidArgumentError("CSR dimensions/nnz exceed int32 storage");
    if (nnz_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        col_idx_.reset(static_cast<int*>(allocate_device_memory(nnz_ * sizeof(int))));
        values_.reset(static_cast<float*>(allocate_device_memory(nnz_ * sizeof(float))));
    } else if (rows_ > 0) {
        row_ptr_.reset(static_cast<int*>(allocate_device_memory((rows_ + 1) * sizeof(int))));
        zero_device_memory(row_ptr_.get(), (rows_ + 1) * sizeof(int));
    }
}

SparseCSR SparseCSR::from_dense(const Matrix& dense, float threshold) {
    Matrix copy = dense;
    copy.download();
    const std::size_t rows = dense.rows(), cols = dense.cols();
    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> col_idx;
    std::vector<float> values;
    col_idx.reserve(dense.size() / 8 + 1);
    values.reserve(dense.size() / 8 + 1);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const float v = copy.data()[r * cols + c];
            if (v > threshold || v < -threshold) {
                col_idx.push_back(static_cast<int>(c));
                values.push_back(v);
            }
        }
        row_ptr[r + 1] = static_cast<int>(col_idx.size());
    }
    SparseCSR out(rows, cols, values.size());
    if (!values.empty()) {
        checkCuda(cudaMemcpyAsync(out.row_ptr_.get(), row_ptr.data(), (rows + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()),
                  "csr upload");
        checkCuda(cudaMemcpyAsync(out.col_idx_.get(), col_idx.data(), values.size() * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()),
                  "csr col upload");
        checkCuda(cudaMemcpyAsync(out.values_.get(), values.data(), values.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, compute_stream()),
                  "csr val upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "csr upload sync");
    }
    return out;
}

SparseCSR SparseCSR::from_coo(std::size_t rows, std::size_t cols, const std::vector<int>& row_idx,
                              const std::vector<int>& col_idx, const std::vector<float>& values) {
    if (row_idx.size() != col_idx.size() || row_idx.size() != values.size())
        throw ShapeMismatchError("from_coo triplet size mismatch");
    if (rows > INT_MAX || cols > INT_MAX || values.size() > INT_MAX)
        throw InvalidArgumentError("COO dimensions/nnz exceed int32 storage");
    std::vector<int> order(row_idx.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return row_idx[a] < row_idx[b] || (row_idx[a] == row_idx[b] && col_idx[a] < col_idx[b]);
    });
    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> cols_sorted;
    std::vector<float> vals_sorted;
    int previous_row = -1, previous_col = -1;
    for (int o : order) {
        if (row_idx[o] < 0 || static_cast<std::size_t>(row_idx[o]) >= rows || col_idx[o] < 0 ||
            static_cast<std::size_t>(col_idx[o]) >= cols)
            throw OutOfRangeError("from_coo index out of range");
        if (row_idx[o] == previous_row && col_idx[o] == previous_col)
            vals_sorted.back() += values[o];
        else {
            cols_sorted.push_back(col_idx[o]);
            vals_sorted.push_back(values[o]);
            row_ptr[static_cast<std::size_t>(row_idx[o]) + 1]++;
            previous_row = row_idx[o];
            previous_col = col_idx[o];
        }
    }
    for (std::size_t r = 0; r < rows; ++r)
        row_ptr[r + 1] += row_ptr[r];
    SparseCSR out(rows, cols, vals_sorted.size());
    if (!vals_sorted.empty()) {
        checkCuda(cudaMemcpyAsync(out.row_ptr_.get(), row_ptr.data(), (rows + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice, compute_stream()),
                  "coo upload");
        checkCuda(cudaMemcpyAsync(out.col_idx_.get(), cols_sorted.data(),
                                  vals_sorted.size() * sizeof(int), cudaMemcpyHostToDevice,
                                  compute_stream()),
                  "coo col upload");
        checkCuda(cudaMemcpyAsync(out.values_.get(), vals_sorted.data(),
                                  vals_sorted.size() * sizeof(float), cudaMemcpyHostToDevice,
                                  compute_stream()),
                  "coo val upload");
        checkCuda(cudaStreamSynchronize(compute_stream()), "coo upload sync");
    }
    return out;
}
