// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: COO/CSR/CSC dönüşümleri; mevcut host sıralaması korunur.

// Conversions
SparseCSR coo_to_csr(const SparseCOO& coo) {
    std::vector<int> rows(coo.nnz()), cols(coo.nnz());
    std::vector<float> values(coo.nnz());
    checkCuda(copy_sync(rows.data(), coo.row_indices(), rows.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
              "COO rows");
    checkCuda(copy_sync(cols.data(), coo.col_indices(), cols.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
              "COO cols");
    checkCuda(copy_sync(values.data(), coo.values(), values.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
              "COO values");
    return SparseCSR::from_coo(coo.rows(), coo.cols(), rows, cols, values);
}

SparseCOO csr_to_coo(const SparseCSR& csr) {
    std::vector<int> h_ptr(csr.rows() + 1);
    checkCuda(copy_sync(h_ptr.data(), csr.row_ptr(), (csr.rows() + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost),
              "memcpy");

    std::vector<int> h_row_idx(csr.nnz());
    for (size_t r = 0; r < csr.rows(); ++r) {
        for (int j = h_ptr[r]; j < h_ptr[r + 1]; ++j) {
            h_row_idx[j] = r;
        }
    }

    SparseCOO coo(csr.rows(), csr.cols(), csr.nnz());
    checkCuda(copy_sync(const_cast<int*>(coo.row_indices()), h_row_idx.data(),
                        csr.nnz() * sizeof(int), cudaMemcpyHostToDevice),
              "memcpy");
    checkCuda(copy_sync(const_cast<int*>(coo.col_indices()), csr.col_idx(), csr.nnz() * sizeof(int),
                        cudaMemcpyDeviceToDevice),
              "memcpy");
    checkCuda(copy_sync(const_cast<float*>(coo.values()), csr.values(), csr.nnz() * sizeof(float),
                        cudaMemcpyDeviceToDevice),
              "memcpy");
    return coo;
}

SparseCSC csr_to_csc(const SparseCSR& csr) {
    SparseCSC csc(csr.rows(), csr.cols(), csr.nnz());

    int* counts = static_cast<int*>(allocate_device_memory((csr.cols() + 1) * sizeof(int)));
    checkCuda(cudaMemsetAsync(counts, 0, (csr.cols() + 1) * sizeof(int), compute_stream()),
              "memset");

    if (csr.nnz() > 0) {
        count_nnz_per_col_kernel<<<detail::capped_grid(csr.nnz(), 256), 256, 0, compute_stream()>>>(
            csr.col_idx(), counts, csr.nnz());
    }

    std::vector<int> h_counts(csr.cols() + 1, 0);
    checkCuda(
        copy_sync(h_counts.data(), counts, (csr.cols() + 1) * sizeof(int), cudaMemcpyDeviceToHost),
        "memcpy");

    std::vector<int> h_ptr(csr.cols() + 1, 0);
    for (size_t i = 0; i < csr.cols(); ++i)
        h_ptr[i + 1] = h_ptr[i] + h_counts[i];

    checkCuda(
        copy_sync(counts, h_ptr.data(), (csr.cols() + 1) * sizeof(int), cudaMemcpyHostToDevice),
        "memcpy"); // Now counts acts as col_ptr

    int* write_pos = static_cast<int*>(allocate_device_memory(csr.cols() * sizeof(int)));
    checkCuda(cudaMemsetAsync(write_pos, 0, csr.cols() * sizeof(int), compute_stream()), "memset");

    // Copy col_ptr to CSC
    checkCuda(copy_sync(csc.col_ptr_.get(), counts, (csr.cols() + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice),
              "memcpy");

    if (csr.rows() > 0) {
        csr_to_csc_scatter_kernel<<<detail::capped_grid(csr.rows(), 256), 256, 0,
                                    compute_stream()>>>(csr.row_ptr(), csr.col_idx(), csr.values(),
                                                        csc.col_ptr_.get(), csc.row_idx_.get(),
                                                        csc.values_.get(), write_pos, csr.rows());
    }

    free_device_memory(counts);
    free_device_memory(write_pos);
    return csc;
}

SparseCSR csc_to_csr(const SparseCSC& csc) {
    std::vector<int> ptr(csc.cols() + 1), rows(csc.nnz()), cols(csc.nnz());
    std::vector<float> values(csc.nnz());
    checkCuda(
        copy_sync(ptr.data(), csc.col_ptr_.get(), ptr.size() * sizeof(int), cudaMemcpyDeviceToHost),
        "CSC ptr");
    checkCuda(copy_sync(rows.data(), csc.row_idx_.get(), rows.size() * sizeof(int),
                        cudaMemcpyDeviceToHost),
              "CSC rows");
    checkCuda(copy_sync(values.data(), csc.values_.get(), values.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
              "CSC values");
    for (std::size_t col = 0; col < csc.cols(); ++col)
        for (int i = ptr[col]; i < ptr[col + 1]; ++i)
            cols[i] = static_cast<int>(col);
    return SparseCSR::from_coo(csc.rows(), csc.cols(), rows, cols, values);
}
