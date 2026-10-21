#pragma once

/**
 * @file thread_pool.hpp
 * @brief MatrixFlash CPU cekirdekleri icin surec capinda paralel dongu havuzu.
 *
 * Calisanlar ve cagiran thread ayni atomik parca sayacindan is alir; hizli
 * thread daha cok parca kosar. `parallel_for` tum parcalar bitmeden ve hicbir
 * calisan isi gormeyi birakmadan donmez, bu yuzden `user` cagiranin
 * yigininda yasayabilir. Bos calisanlar uyur, cekirdek yakmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

namespace matrix_pro {

class WorkStealingPool {
public:
    static WorkStealingPool& instance();

    /// Bir parallel_for'a katilan thread sayisi: calisanlar + cagiran.
    [[nodiscard]] int worker_count() const;

    /// [begin, end) araligini `grain` genisliginde parcalara boler ve
    /// `fn(sub_begin, sub_end, user)` cagrisini her parca icin tam bir kez yapar.
    /// `fn` icinden atilan ilk istisna cagirana yeniden firlatilir.
    /// Ic ice cagri ve havuz baska bir thread'de mesgulken yapilan cagri
    /// satir ici (seri) kosar; kilitlenme olmaz.
    void parallel_for(int begin, int end, int grain, void (*fn)(int, int, void*), void* user);

private:
    WorkStealingPool();
    ~WorkStealingPool();
    WorkStealingPool(const WorkStealingPool&) = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    struct Impl;
    Impl* impl_;
};

}  // namespace matrix_pro
