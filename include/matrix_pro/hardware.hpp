#pragma once

/**
 * @file hardware.hpp
 * @brief Faz 2 donanim katmaninin tek basligi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/backend_registry.hpp"
#include "matrix_pro/core/device.hpp"
#include "matrix_pro/core/heterogeneous_scheduler.hpp"
#include "matrix_pro/core/thread_pool.hpp"
#include "matrix_pro/cpu/gemm.hpp"
#include "matrix_pro/distributed/collectives.hpp"
#include "matrix_pro/distributed/topology.hpp"
