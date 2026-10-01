// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Pinned host scratch ve SM sayısına göre sınırlı reduction grid.

constexpr unsigned reduce_block = 256;

// Pinned host scratch for device->host partial reads: pageable copies serialize
// the stream, pinned copies go straight over DMA. Falls back to normal memory
// when cudaMallocHost is unavailable (host without WDDM/pinning support).
class PinnedFloatScratch {
  public:
    explicit PinnedFloatScratch(std::size_t count) : fallback_(count) {
        if (cudaMallocHost(&pinned_, count * sizeof(float)) != cudaSuccess) {
            pinned_ = nullptr;
            cudaGetLastError(); // clear the sticky error; fallback covers us
        }
    }
    ~PinnedFloatScratch() {
        if (pinned_ != nullptr)
            cudaFreeHost(pinned_);
    }
    PinnedFloatScratch(const PinnedFloatScratch&) = delete;
    PinnedFloatScratch& operator=(const PinnedFloatScratch&) = delete;
    float* data() noexcept { return pinned_ != nullptr ? pinned_ : fallback_.data(); }

  private:
    float* pinned_ = nullptr;
    std::vector<float> fallback_;
};

// Reduction grids are capped: the kernels are grid-stride loops, so a small
// multiple of the SM count saturates the GPU while keeping the partial buffer
// (and the host-side combine pass) tiny instead of scaling with element count.
unsigned reduce_grid(std::size_t count) {
    const unsigned needed = static_cast<unsigned>((count + reduce_block - 1) / reduce_block);
    const unsigned max_blocks = static_cast<unsigned>(matrix_pro::sm_count()) * 8u;
    return needed < max_blocks ? needed : max_blocks;
}
