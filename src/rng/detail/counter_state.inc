// Dahili uygulama parçası: src/rng/rng.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Global atomic seed ve counter; tüm dağılımlar tek durumu paylaşır.

std::atomic<unsigned long long>& global_seed() {
    static std::atomic<unsigned long long> seed{12345ULL};
    return seed;
}
std::atomic<unsigned long long>& global_counter() {
    static std::atomic<unsigned long long> counter{0ULL};
    return counter;
}
