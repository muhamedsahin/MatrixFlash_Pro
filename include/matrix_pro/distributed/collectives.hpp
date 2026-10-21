#pragma once

/**
 * @file collectives.hpp
 * @brief Ring ve agac all-reduce, yayin, gather ve scatter.
 *
 * Her rank kendi tamponuna sahiptir. CPU tamponlari dogrudan islenir.
 * CUDA tamponlari ayni algoritmayi cihaz uzerinde, P2P kopya ile kosar.
 * NCCL yuklu ve her rank ayri aygittaysa NCCL secilir; degilse yerlesik halka.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/device.hpp"

#include <cstddef>

namespace matrix_pro {

enum class ReduceOp { Sum = 0, Mean = 1, Max = 2 };
enum class CollectiveAlgorithm { Ring = 0, Tree = 1 };

struct CollectiveBuffer {
    float* data = nullptr;
    std::size_t count = 0;
    DeviceId device{};
};

void all_reduce(CollectiveBuffer* ranks, int rank_count, ReduceOp op, CollectiveAlgorithm algorithm);
void broadcast(CollectiveBuffer* ranks, int rank_count, int root);
void all_gather(CollectiveBuffer* ranks, int rank_count, std::size_t count_per_rank);
void reduce_scatter(CollectiveBuffer* ranks, int rank_count, ReduceOp op);
void scatter(CollectiveBuffer* ranks, int rank_count, int root, std::size_t count_per_rank);
void gather(CollectiveBuffer* ranks, int rank_count, int root, std::size_t count_per_rank);

}  // namespace matrix_pro
