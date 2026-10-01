// Dahili uygulama parçası: src/ops/statistics.cu.
// Kapsam sahibi .cu dosyasında açılır; bu dosya tek başına derlenmez.
// Görev: Blok içi toplam/norm/min/max ve indeks indirgeme çekirdekleri.

__global__ void reduce_partial_kernel(const float* input, std::size_t count, float* partial) {
    __shared__ float ssum[reduce_block];
    __shared__ float ssq[reduce_block];
    __shared__ float sl1[reduce_block];
    __shared__ float smin[reduce_block];
    __shared__ float smax[reduce_block];
    const auto tid = threadIdx.x;

    float sum = 0.0f, sum_sq = 0.0f, sum_l1 = 0.0f;
    float local_min = FLT_MAX, local_max = -FLT_MAX;

    // float4 prefix: 4x fewer global loads on aligned buffers.
    const std::size_t n4 = count / 4;
    if (n4 > 0 && (reinterpret_cast<uintptr_t>(input) & 0xFu) == 0u) {
        const float4* in4 = reinterpret_cast<const float4*>(input);
        for (std::size_t index = blockIdx.x * reduce_block + tid; index < n4;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float4 v = in4[index];
            sum += v.x + v.y + v.z + v.w;
            sum_l1 += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
            sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            local_min = fminf(local_min, fminf(fminf(v.x, v.y), fminf(v.z, v.w)));
            local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        for (std::size_t index = n4 * 4 + blockIdx.x * reduce_block + tid; index < count;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float value = input[index];
            sum += value;
            sum_l1 += fabsf(value);
            sum_sq += value * value;
            local_min = fminf(local_min, value);
            local_max = fmaxf(local_max, value);
        }
    } else {
        for (std::size_t index = blockIdx.x * reduce_block + tid; index < count;
             index += static_cast<std::size_t>(gridDim.x) * reduce_block) {
            const float value = input[index];
            sum += value;
            sum_l1 += fabsf(value);
            sum_sq += value * value;
            local_min = fminf(local_min, value);
            local_max = fmaxf(local_max, value);
        }
    }

    ssum[tid] = sum;
    ssq[tid] = sum_sq;
    sl1[tid] = sum_l1;
    smin[tid] = local_min;
    smax[tid] = local_max;
    __syncthreads();

    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
            ssq[tid] += ssq[tid + stride];
            sl1[tid] += sl1[tid + stride];
            smin[tid] = fminf(smin[tid], smin[tid + stride]);
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * 5U;
        partial[offset + 0] = ssum[0];
        partial[offset + 1] = ssq[0];
        partial[offset + 2] = sl1[0];
        partial[offset + 3] = smin[0];
        partial[offset + 4] = smax[0];
    }
}

__global__ void arg_partial_kernel(const float* input, std::size_t count, int find_max,
                                   float* partial_values, std::size_t* partial_indices) {
    __shared__ float svalue[reduce_block];
    __shared__ std::size_t sindex[reduce_block];

    const auto tid = threadIdx.x;
    float best = find_max ? -FLT_MAX : FLT_MAX;
    std::size_t best_index = 0;

    for (std::size_t index = blockIdx.x * reduce_block + tid; index < count;
         index += gridDim.x * reduce_block) {
        const float value = input[index];
        if (find_max ? (value > best) : (value < best)) {
            best = value;
            best_index = index;
        }
    }

    svalue[tid] = best;
    sindex[tid] = best_index;
    __syncthreads();

    for (unsigned stride = reduce_block / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float va = svalue[tid];
            const float vb = svalue[tid + stride];
            const std::size_t ia = sindex[tid];
            const std::size_t ib = sindex[tid + stride];
            const bool better = find_max ? (vb > va) : (vb < va);
            const bool tie = (vb == va);
            if (better || (tie && ib < ia)) {
                svalue[tid] = vb;
                sindex[tid] = ib;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_values[blockIdx.x] = svalue[0];
        partial_indices[blockIdx.x] = sindex[0];
    }
}
