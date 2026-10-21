#include "matrix_pro/detail/cuda_hooks.hpp"
#include "matrix_pro/distributed/topology.hpp"

#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace matrix_pro {
namespace {

int count_numa_nodes() {
#if defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformation(nullptr, &length);
    if (length == 0) {
        return 1;
    }
    std::vector<unsigned char> raw(length);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(raw.data());
    if (!GetLogicalProcessorInformation(info, &length)) {
        return 1;
    }
    int nodes = 0;
    const std::size_t records = length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (std::size_t i = 0; i < records; ++i) {
        if (info[i].Relationship == RelationNumaNode) {
            ++nodes;
        }
    }
    return nodes > 0 ? nodes : 1;
#else
    return 1;
#endif
}

}  // namespace

MachineTopology discover_topology() {
    MachineTopology topology;
    const unsigned cores = std::thread::hardware_concurrency();
    topology.logical_cpus = cores == 0 ? 1 : static_cast<int>(cores);
    topology.numa_nodes = count_numa_nodes();
    topology.cuda_devices = hw::cuda_device_count();
    for (int a = 0; a < topology.cuda_devices; ++a) {
        for (int b = a + 1; b < topology.cuda_devices; ++b) {
            PeerLink link;
            link.device_a = a;
            link.device_b = b;
            link.peer_access = hw::cuda_peer_access(a, b);
            link.nvlink_class = hw::cuda_nvlink(a, b);
            if (link.peer_access) {
                hw::cuda_enable_peer(a, b);
                hw::cuda_enable_peer(b, a);
            }
            topology.links.push_back(link);
        }
    }
    return topology;
}

}  // namespace matrix_pro
