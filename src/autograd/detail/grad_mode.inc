// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Thread-local NoGrad durumu; iç içe kapsam önceki durumu geri yükler.

namespace {
thread_local bool tl_grad_enabled = true;
}

bool is_grad_enabled() {
    return tl_grad_enabled;
}
void set_grad_enabled(bool enabled) {
    tl_grad_enabled = enabled;
}

NoGradGuard::NoGradGuard() : prev_state_(tl_grad_enabled) {
    tl_grad_enabled = false;
}
NoGradGuard::~NoGradGuard() {
    tl_grad_enabled = prev_state_;
}

// Variable extensions
