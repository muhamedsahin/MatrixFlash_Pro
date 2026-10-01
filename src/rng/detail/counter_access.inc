// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Global sayacı her çağrı için mevcut miktarda ilerletir.

unsigned long long next_counter(std::size_t n) {
    return global_counter().fetch_add(n);
}
