/**
 * @file collectives.cpp
 * @brief Halka ve agac all-reduce. Indeks aritmetigi p=2 icin elle izlendi.
 */

#include "matrix_pro/core/errors.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/distributed/collectives.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace matrix_pro {
namespace {

void check_ranks(const CollectiveBuffer* ranks, int rank_count) {
    if (ranks == nullptr || rank_count <= 0) {
        throw MatrixProError("collective: empty rank list");
    }
    for (int rank = 0; rank < rank_count; ++rank) {
        if (ranks[rank].data == nullptr || ranks[rank].count == 0) {
            throw MatrixProError("collective: rank buffer is empty");
        }
    }
}

void reduce_into(float* dst, const float* src, std::size_t count, ReduceOp op) {
    if (op == ReduceOp::Max) {
        elementwise_maximum(dst, src, static_cast<int>(count));
        return;
    }
    for (std::size_t i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

void scale_all(std::vector<std::vector<float>>& buffers, float scale) {
    for (std::vector<float>& buffer : buffers) {
        elementwise_scale(buffer.data(), static_cast<int>(buffer.size()), scale);
    }
}

void ring_allreduce(std::vector<std::vector<float>>& buffers, ReduceOp op) {
    const int ranks = static_cast<int>(buffers.size());
    const std::size_t count = buffers[0].size();
    if (ranks == 1) {
        if (op == ReduceOp::Mean) {
            scale_all(buffers, 1.0f);
        }
        return;
    }
    const ReduceOp chunk_op = op == ReduceOp::Mean ? ReduceOp::Sum : op;
    const std::size_t chunk = count / static_cast<std::size_t>(ranks);
    const std::size_t main = chunk * static_cast<std::size_t>(ranks);
    if (chunk > 0) {
        for (int step = 0; step < ranks - 1; ++step) {
            const std::vector<std::vector<float>> snapshot = buffers;
            for (int rank = 0; rank < ranks; ++rank) {
                const int send_index = (rank - step + ranks) % ranks;
                const int destination = (rank + 1) % ranks;
                const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
                if (chunk_op == ReduceOp::Max) {
                    for (std::size_t i = 0; i < chunk; ++i) {
                        buffers[static_cast<std::size_t>(destination)][offset + i] =
                            std::max(snapshot[static_cast<std::size_t>(destination)][offset + i],
                                     snapshot[static_cast<std::size_t>(rank)][offset + i]);
                    }
                } else {
                    for (std::size_t i = 0; i < chunk; ++i) {
                        buffers[static_cast<std::size_t>(destination)][offset + i] =
                            snapshot[static_cast<std::size_t>(destination)][offset + i] +
                            snapshot[static_cast<std::size_t>(rank)][offset + i];
                    }
                }
            }
        }
        for (int step = 0; step < ranks - 1; ++step) {
            const std::vector<std::vector<float>> snapshot = buffers;
            for (int rank = 0; rank < ranks; ++rank) {
                const int send_index = (rank - step + 1 + ranks) % ranks;
                const int destination = (rank + 1) % ranks;
                const std::size_t offset = static_cast<std::size_t>(send_index) * chunk;
                std::memcpy(buffers[static_cast<std::size_t>(destination)].data() + offset,
                            snapshot[static_cast<std::size_t>(rank)].data() + offset, chunk * sizeof(float));
            }
        }
    }
    if (main < count) {
        const std::size_t tail = count - main;
        std::vector<float> acc(tail);
        if (chunk_op == ReduceOp::Max) {
            std::memcpy(acc.data(), buffers[0].data() + main, tail * sizeof(float));
            for (int rank = 1; rank < ranks; ++rank) {
                for (std::size_t i = 0; i < tail; ++i) {
                    acc[i] = std::max(acc[i], buffers[static_cast<std::size_t>(rank)][main + i]);
                }
            }
        } else {
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int rank = 0; rank < ranks; ++rank) {
                for (std::size_t i = 0; i < tail; ++i) {
                    acc[i] += buffers[static_cast<std::size_t>(rank)][main + i];
                }
            }
        }
        for (int rank = 0; rank < ranks; ++rank) {
            std::memcpy(buffers[static_cast<std::size_t>(rank)].data() + main, acc.data(), tail * sizeof(float));
        }
    }
    if (op == ReduceOp::Mean) {
        scale_all(buffers, 1.0f / static_cast<float>(ranks));
    }
}

void tree_allreduce(std::vector<std::vector<float>>& buffers, ReduceOp op) {
    const int ranks = static_cast<int>(buffers.size());
    const ReduceOp chunk_op = op == ReduceOp::Mean ? ReduceOp::Sum : op;
    for (int rank = ranks - 1; rank > 0; --rank) {
        const int parent = (rank - 1) / 2;
        reduce_into(buffers[static_cast<std::size_t>(parent)].data(), buffers[static_cast<std::size_t>(rank)].data(),
                    buffers[0].size(), chunk_op);
    }
    for (int rank = 1; rank < ranks; ++rank) {
        const int parent = (rank - 1) / 2;
        std::memcpy(buffers[static_cast<std::size_t>(rank)].data(), buffers[static_cast<std::size_t>(parent)].data(),
                    buffers[0].size() * sizeof(float));
    }
    if (op == ReduceOp::Mean) {
        scale_all(buffers, 1.0f / static_cast<float>(ranks));
    }
}

bool all_cuda(const CollectiveBuffer* ranks, int rank_count) {
    for (int rank = 0; rank < rank_count; ++rank) {
        if (ranks[rank].device.type != DeviceType::Cuda) {
            return false;
        }
    }
    return true;
}

bool all_cpu(const CollectiveBuffer* ranks, int rank_count) {
    for (int rank = 0; rank < rank_count; ++rank) {
        if (ranks[rank].device.type != DeviceType::Cpu) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<float>> download(const CollectiveBuffer* ranks, int rank_count) {
    std::vector<std::vector<float>> host(static_cast<std::size_t>(rank_count));
    for (int rank = 0; rank < rank_count; ++rank) {
        host[static_cast<std::size_t>(rank)].resize(ranks[rank].count);
        if (ranks[rank].device.type == DeviceType::Cpu) {
            std::memcpy(host[static_cast<std::size_t>(rank)].data(), ranks[rank].data, ranks[rank].count * sizeof(float));
        } else if (ranks[rank].device.type == DeviceType::Cuda) {
            hw::cuda_copy_d2h(ranks[rank].device.index, host[static_cast<std::size_t>(rank)].data(), ranks[rank].data,
                              ranks[rank].count);
            hw::cuda_sync(ranks[rank].device.index);
        } else {
            throw MatrixProError("collective: only CPU and CUDA buffers are executable");
        }
    }
    return host;
}

void upload(CollectiveBuffer* ranks, int rank_count, const std::vector<std::vector<float>>& host) {
    for (int rank = 0; rank < rank_count; ++rank) {
        if (ranks[rank].device.type == DeviceType::Cpu) {
            std::memcpy(ranks[rank].data, host[static_cast<std::size_t>(rank)].data(), ranks[rank].count * sizeof(float));
        } else {
            hw::cuda_copy_h2d(ranks[rank].device.index, ranks[rank].data, host[static_cast<std::size_t>(rank)].data(),
                              ranks[rank].count);
            hw::cuda_sync(ranks[rank].device.index);
        }
    }
}

}  // namespace

void all_reduce(CollectiveBuffer* ranks, int rank_count, ReduceOp op, CollectiveAlgorithm algorithm) {
    check_ranks(ranks, rank_count);
    const std::size_t count = ranks[0].count;
    for (int rank = 1; rank < rank_count; ++rank) {
        if (ranks[rank].count != count) {
            throw MatrixProError("all_reduce: rank lengths differ");
        }
    }
    if (all_cuda(ranks, rank_count) && algorithm == CollectiveAlgorithm::Ring) {
        std::vector<float*> pointers(static_cast<std::size_t>(rank_count));
        std::vector<int> devices(static_cast<std::size_t>(rank_count));
        for (int rank = 0; rank < rank_count; ++rank) {
            pointers[static_cast<std::size_t>(rank)] = ranks[rank].data;
            devices[static_cast<std::size_t>(rank)] = ranks[rank].device.index;
        }
        const int reduce = op == ReduceOp::Max ? 2 : op == ReduceOp::Mean ? 1 : 0;
        if (hw::cuda_nccl_allreduce(pointers.data(), devices.data(), rank_count, count, reduce)) {
            return;
        }
        hw::cuda_allreduce(pointers.data(), devices.data(), rank_count, count, reduce, 0);
        return;
    }
    if (!all_cpu(ranks, rank_count) && !all_cuda(ranks, rank_count)) {
        throw MatrixProError("all_reduce: mixed CPU and CUDA ranks are not a single collective");
    }
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    if (algorithm == CollectiveAlgorithm::Tree) {
        tree_allreduce(host, op);
    } else {
        ring_allreduce(host, op);
    }
    upload(ranks, rank_count, host);
}

void broadcast(CollectiveBuffer* ranks, int rank_count, int root) {
    check_ranks(ranks, rank_count);
    if (root < 0 || root >= rank_count) {
        throw MatrixProError("broadcast: root out of range");
    }
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    for (int rank = 0; rank < rank_count; ++rank) {
        if (rank == root) {
            continue;
        }
        if (host[static_cast<std::size_t>(rank)].size() != host[static_cast<std::size_t>(root)].size()) {
            throw MatrixProError("broadcast: rank lengths differ");
        }
        host[static_cast<std::size_t>(rank)] = host[static_cast<std::size_t>(root)];
    }
    upload(ranks, rank_count, host);
}

void all_gather(CollectiveBuffer* ranks, int rank_count, std::size_t count_per_rank) {
    check_ranks(ranks, rank_count);
    const std::size_t full = count_per_rank * static_cast<std::size_t>(rank_count);
    for (int rank = 0; rank < rank_count; ++rank) {
        if (ranks[rank].count < full) {
            throw MatrixProError("all_gather: buffer shorter than rank_count * count");
        }
    }
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    std::vector<float> gathered(full);
    for (int rank = 0; rank < rank_count; ++rank) {
        std::memcpy(gathered.data() + static_cast<std::size_t>(rank) * count_per_rank,
                    host[static_cast<std::size_t>(rank)].data(), count_per_rank * sizeof(float));
    }
    for (int rank = 0; rank < rank_count; ++rank) {
        std::memcpy(host[static_cast<std::size_t>(rank)].data(), gathered.data(), full * sizeof(float));
    }
    upload(ranks, rank_count, host);
}

void reduce_scatter(CollectiveBuffer* ranks, int rank_count, ReduceOp op) {
    check_ranks(ranks, rank_count);
    if (ranks[0].count % static_cast<std::size_t>(rank_count) != 0) {
        throw MatrixProError("reduce_scatter: length must be divisible by rank count");
    }
    const std::size_t chunk = ranks[0].count / static_cast<std::size_t>(rank_count);
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    ring_allreduce(host, op == ReduceOp::Mean ? ReduceOp::Sum : op);
    for (int rank = 0; rank < rank_count; ++rank) {
        std::vector<float> piece(host[static_cast<std::size_t>(rank)].begin() + static_cast<std::ptrdiff_t>(rank * chunk),
                                 host[static_cast<std::size_t>(rank)].begin() + static_cast<std::ptrdiff_t>((rank + 1) * chunk));
        std::memcpy(host[static_cast<std::size_t>(rank)].data(), piece.data(), chunk * sizeof(float));
    }
    if (op == ReduceOp::Mean) {
        for (std::vector<float>& buffer : host) {
            elementwise_scale(buffer.data(), static_cast<int>(chunk), 1.0f / static_cast<float>(rank_count));
        }
    }
    upload(ranks, rank_count, host);
}

void scatter(CollectiveBuffer* ranks, int rank_count, int root, std::size_t count_per_rank) {
    check_ranks(ranks, rank_count);
    if (root < 0 || root >= rank_count) {
        throw MatrixProError("scatter: root out of range");
    }
    if (ranks[root].count < count_per_rank * static_cast<std::size_t>(rank_count)) {
        throw MatrixProError("scatter: root buffer is too small");
    }
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    const std::vector<float> source = host[static_cast<std::size_t>(root)];
    for (int rank = 0; rank < rank_count; ++rank) {
        std::memcpy(host[static_cast<std::size_t>(rank)].data(),
                    source.data() + static_cast<std::size_t>(rank) * count_per_rank, count_per_rank * sizeof(float));
    }
    upload(ranks, rank_count, host);
}

void gather(CollectiveBuffer* ranks, int rank_count, int root, std::size_t count_per_rank) {
    check_ranks(ranks, rank_count);
    if (root < 0 || root >= rank_count) {
        throw MatrixProError("gather: root out of range");
    }
    if (ranks[root].count < count_per_rank * static_cast<std::size_t>(rank_count)) {
        throw MatrixProError("gather: root buffer is too small");
    }
    std::vector<std::vector<float>> host = download(ranks, rank_count);
    std::vector<float> gathered(count_per_rank * static_cast<std::size_t>(rank_count));
    for (int rank = 0; rank < rank_count; ++rank) {
        std::memcpy(gathered.data() + static_cast<std::size_t>(rank) * count_per_rank,
                    host[static_cast<std::size_t>(rank)].data(), count_per_rank * sizeof(float));
    }
    std::memcpy(host[static_cast<std::size_t>(root)].data(), gathered.data(), gathered.size() * sizeof(float));
    upload(ranks, rank_count, host);
}

}  // namespace matrix_pro
