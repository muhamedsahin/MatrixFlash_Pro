#pragma once

namespace matrix_pro {

// Controls where a Matrix/Tensor keeps its data.
//
// - host_and_device (default): keeps both a host mirror and the device buffer,
//   matching the historical behaviour of the library. Host access through
//   data()/at() is always valid.
// - device_only: allocates the device buffer only. The host mirror is created
//   lazily the first time download() (or any host accessor) is used, saving RAM
//   and avoiding redundant host<->device traffic for tensor-heavy workloads.
enum class MemoryMode {
    host_and_device,
    device_only
};

} // namespace matrix_pro