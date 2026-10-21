/**
 * @file cpuid.cpp
 * @brief OS'in gercekten actigi SIMD bitlerini okur. AVX kayitlari XCR0 ile dogrulanir.
 */

#include "matrix_pro/core/device.hpp"

#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace matrix_pro {
namespace {

void cpuid(int leaf, int subleaf, int out[4]) {
#if defined(_MSC_VER)
    __cpuidex(out, leaf, subleaf);
#else
    __cpuid_count(static_cast<unsigned>(leaf), static_cast<unsigned>(subleaf),
                  reinterpret_cast<unsigned&>(out[0]), reinterpret_cast<unsigned&>(out[1]),
                  reinterpret_cast<unsigned&>(out[2]), reinterpret_cast<unsigned&>(out[3]));
#endif
}

unsigned long long read_xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned eax = 0;
    unsigned edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
#endif
}

int numa_node_count() {
#if defined(_WIN32)
    // GetNumaHighestNodeNumber is resolved in topology.cpp. Here a conservative 1
    // is replaced by discover_topology(); ISA detection only needs a positive count.
#endif
    return 1;
}

}  // namespace

CpuIsa detect_cpu_isa() {
    CpuIsa isa;
    const unsigned hc = std::thread::hardware_concurrency();
    isa.logical_cores = hc == 0 ? 1 : static_cast<int>(hc);
    isa.numa_nodes = numa_node_count();

    int info[4] = {};
    cpuid(0, 0, info);
    const int max_leaf = info[0];
    if (max_leaf < 1) {
        return isa;
    }
    cpuid(1, 0, info);
    const bool osxsave = (info[2] & (1 << 27)) != 0;
    if (!osxsave) {
        return isa;
    }
    const unsigned long long xcr = read_xcr0();
    const bool ymm = (xcr & 0x6ull) == 0x6ull;
    const bool zmm = (xcr & 0xE6ull) == 0xE6ull;
    if (max_leaf >= 7 && ymm) {
        int ext[4] = {};
        cpuid(7, 0, ext);
        isa.avx2 = (ext[1] & (1 << 5)) != 0;
        isa.fma = (info[2] & (1 << 12)) != 0;
        isa.avx512f = zmm && (ext[1] & (1 << 16)) != 0;
        isa.amx_bf16 = (ext[3] & (1 << 22)) != 0;
        isa.amx_int8 = (ext[3] & (1 << 25)) != 0;
    }
#if defined(__ARM_NEON)
    isa.neon = true;
#endif
    return isa;
}

}  // namespace matrix_pro
