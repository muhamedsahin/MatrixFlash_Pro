// Dahili uygulama parçası: src/core/cuda_utils.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Vendor solver handle erişim noktaları; cuDNN koşullu derlenir.

cusolverDnHandle_t& cusolver_handle() {
    return thread_solvers().solver_for(current_device());
}

#ifdef MATRIX_PRO_HAS_CUDNN
cudnnHandle_t& cudnn_handle() {
    return thread_solvers().cudnn_for(current_device());
}
#endif
