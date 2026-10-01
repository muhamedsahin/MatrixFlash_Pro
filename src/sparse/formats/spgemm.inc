// Dahili uygulama parçası: src/sparse/sparse_formats.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: cuSPARSE SpGEMM descriptor ve workspace yaşam döngüsü.

// SpGEMM and Addition
SparseCSR spgemm(const SparseCSR& A, const SparseCSR& B) {
    if (A.cols() != B.rows())
        throw ShapeMismatchError("spgemm inner dimension mismatch");
    cusparseHandle_t handle = get_cusparse_handle();
    CHECK_CUSPARSE(cusparseSetStream(handle, compute_stream()));
    cusparseSpMatDescr_t matA, matB, matC;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows(), A.cols(), A.nnz(),
                                     const_cast<int*>(A.row_ptr()), const_cast<int*>(A.col_idx()),
                                     const_cast<float*>(A.values()), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B.rows(), B.cols(), B.nnz(),
                                     const_cast<int*>(B.row_ptr()), const_cast<int*>(B.col_idx()),
                                     const_cast<float*>(B.values()), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A.rows(), B.cols(), 0, nullptr, nullptr, nullptr,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    size_t bufferSize1 = 0, bufferSize2 = 0;
    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;

    float alpha = 1.0f, beta = 0.0f;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));

    dBuffer1 = allocate_device_memory(bufferSize1);
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1,
        dBuffer1));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
        matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));

    dBuffer2 = allocate_device_memory(bufferSize2);
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                                          &beta, matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));

    int64_t C_rows, C_cols, C_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));

    SparseCSR C(A.rows(), B.cols(), C_nnz);

    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, C.row_ptr(), C.col_idx(), C.values()));

    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta,
                                       matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);

    if (dBuffer1)
        free_device_memory(dBuffer1);
    if (dBuffer2)
        free_device_memory(dBuffer2);

    return C;
}
