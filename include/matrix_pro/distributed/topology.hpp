#pragma once

/**
 * @file topology.hpp
 * @brief NUMA dugumleri ve CUDA P2P / NVLink baglantilari.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <vector>

namespace matrix_pro {

struct PeerLink {
    int device_a = 0;
    int device_b = 0;
    bool peer_access = false;
    bool nvlink_class = false;
};

struct MachineTopology {
    int logical_cpus = 1;
    int numa_nodes = 1;
    int cuda_devices = 0;
    std::vector<PeerLink> links;
};

[[nodiscard]] MachineTopology discover_topology();

}  // namespace matrix_pro
