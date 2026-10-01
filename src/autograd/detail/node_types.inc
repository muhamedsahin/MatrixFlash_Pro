// Dahili uygulama parçası: src/autograd/autograd.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Matrix ve Tensor tape düğümleri; ebeveynler grafiğin yaşam süresini tutar.

struct Variable::Node {
    Matrix value;
    Matrix gradient;
    bool requires_gradient = false;
    bool retain_grad = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_function;
};

struct VarTensor::Node {
    Tensor value;
    Tensor gradient;
    bool requires_gradient = false;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_function;
};
