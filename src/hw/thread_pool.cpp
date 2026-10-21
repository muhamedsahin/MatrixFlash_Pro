/**
 * @file thread_pool.cpp
 * @brief Paylasilan atomik parca sayaci ile kendi kendini zamanlayan havuz.
 *
 * Her `parallel_for` cagiranin yigininda bir `Job` kurar. Cagiran thread ve
 * calisanlar parcalari `next.fetch_add` ile alir: her parca tam bir kez kosar,
 * hizli thread daha cok parca alir (dinamik yuk dengeleme, is calmanin duz
 * donguler icin esdegeri). Cagiran donmeden once
 *   (1) tum parcalarin bittigini (`completed == chunks`),
 *   (2) hicbir calisanin Job'a artik dokunmadigini (`busy == 0`)
 * bekler. (2) Dekker deseniyle kurulur: calisan once `busy`'yi artirir sonra
 * `current`'i okur; cagiran once `current`'i sifirlar sonra `busy`'yi okur.
 * Iki taraf da seq_cst oldugu icin ya cagiran calisani gorur ya da calisan
 * bos `current` gorur; yigindaki Job'a donusten sonra erisim olamaz.
 *
 * Bos calisanlar kisa bir donusten sonra kosul degiskeninde uyur; havuz
 * bosken cekirdek yakmaz. Nesil sayaci uyku kilidi altinda artirilir, bu
 * yuzden uyanma kaybolmaz. Kutuphane C++17 oldugu icin `atomic::wait`
 * kullanilmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "matrix_pro/core/thread_pool.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define MATRIX_PRO_HAS_PAUSE 1
#endif

namespace matrix_pro {
namespace {

/// Bu thread havuz calisani mi ya da su an bir parallel_for yurutuyor mu.
/// Ic ice cagri satir ici kosar; aksi halde calisan kendi isini bekler.
thread_local bool t_inside_pool = false;

constexpr int kSpinBeforeSleep = 4096;
constexpr int kSpinBeforeYield = 256;

inline void cpu_relax() noexcept {
#if defined(MATRIX_PRO_HAS_PAUSE)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

/// Ilk `kSpinBeforeYield` turda pause, sonra isletim sistemine birakir.
inline void backoff(int& spins) noexcept {
    if (spins < kSpinBeforeYield) {
        ++spins;
        cpu_relax();
    } else {
        std::this_thread::yield();
    }
}

struct Job {
    void (*fn)(int, int, void*) = nullptr;
    void* user = nullptr;
    int begin = 0;
    int end = 0;
    int grain = 1;
    int chunks = 0;
    std::atomic<int> next{0};
    std::atomic<int> completed{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;

    /// Parca kalmayana kadar alir ve kosar. Istisna ilk yakalayan tarafindan saklanir;
    /// kalan parcalar atlanir ama sayilir, boylece cagiran beklemeden cikmaz.
    void run_chunks() noexcept {
        for (;;) {
            const int chunk = next.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= chunks) {
                return;
            }
            const int lo = begin + chunk * grain;
            const int hi = end - lo > grain ? lo + grain : end;
            if (!failed.load(std::memory_order_relaxed)) {
                try {
                    fn(lo, hi, user);
                } catch (...) {
                    bool expected = false;
                    if (failed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                        error = std::current_exception();
                    }
                }
            }
            completed.fetch_add(1, std::memory_order_release);
        }
    }
};

}  // namespace

struct WorkStealingPool::Impl {
    std::vector<std::thread> threads;
    std::atomic<Job*> current{nullptr};
    std::atomic<std::uint64_t> generation{0};
    std::atomic<int> busy{0};
    std::atomic<bool> stop{false};
    std::mutex submit;
    std::mutex sleep_mutex;
    std::condition_variable wake;

    /// Yeni nesli yayinlar ve uyuyan calisanlari uyandirir.
    void publish() {
        {
            std::lock_guard<std::mutex> guard(sleep_mutex);
            generation.fetch_add(1, std::memory_order_acq_rel);
        }
        wake.notify_all();
    }

    void loop() {
        t_inside_pool = true;
        std::uint64_t seen = generation.load(std::memory_order_acquire);
        for (;;) {
            std::uint64_t now = seen;
            for (int spin = 0; spin < kSpinBeforeSleep && now == seen; ++spin) {
                cpu_relax();
                now = generation.load(std::memory_order_acquire);
            }
            if (now == seen) {
                std::unique_lock<std::mutex> lock(sleep_mutex);
                wake.wait(lock, [&] { return generation.load(std::memory_order_acquire) != seen; });
                continue;
            }
            seen = now;
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            busy.fetch_add(1, std::memory_order_seq_cst);
            if (Job* job = current.load(std::memory_order_seq_cst)) {
                job->run_chunks();
            }
            busy.fetch_sub(1, std::memory_order_release);
        }
    }
};

WorkStealingPool& WorkStealingPool::instance() {
    static WorkStealingPool pool;
    return pool;
}

WorkStealingPool::WorkStealingPool() : impl_(new Impl) {
    unsigned hardware = std::thread::hardware_concurrency();
    if (hardware < 1) {
        hardware = 1;
    }
    if (hardware > 32) {
        hardware = 32;
    }
    // Cagiran thread da parca kostugu icin calisan sayisi bir eksiktir.
    const unsigned workers = hardware - 1;
    impl_->threads.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
        impl_->threads.emplace_back([this] { impl_->loop(); });
    }
}

WorkStealingPool::~WorkStealingPool() {
    impl_->stop.store(true, std::memory_order_release);
    impl_->publish();
    for (std::thread& thread : impl_->threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    delete impl_;
}

int WorkStealingPool::worker_count() const { return static_cast<int>(impl_->threads.size()) + 1; }

void WorkStealingPool::parallel_for(int begin, int end, int grain, void (*fn)(int, int, void*), void* user) {
    if (end <= begin) {
        return;
    }
    if (fn == nullptr) {
        throw std::invalid_argument("matrix_pro: parallel_for icin fonksiyon gerekli");
    }
    if (grain < 1) {
        grain = 1;
    }
    const long long span = static_cast<long long>(end) - static_cast<long long>(begin);
    const long long chunks = (span + grain - 1) / grain;
    if (chunks <= 1 || impl_->threads.empty() || t_inside_pool) {
        fn(begin, end, user);
        return;
    }
    // Baska bir thread havuzu kullaniyorsa beklemek yerine satir ici kos:
    // o is tum calisanlari zaten doldurur, bu cagiran kendi cekirdegini kullanir.
    std::unique_lock<std::mutex> lock(impl_->submit, std::try_to_lock);
    if (!lock.owns_lock()) {
        fn(begin, end, user);
        return;
    }

    Job job;
    job.fn = fn;
    job.user = user;
    job.begin = begin;
    job.end = end;
    job.grain = grain;
    job.chunks = static_cast<int>(chunks);

    t_inside_pool = true;
    impl_->current.store(&job, std::memory_order_seq_cst);
    impl_->publish();

    job.run_chunks();
    int spins = 0;
    while (job.completed.load(std::memory_order_acquire) < job.chunks) {
        backoff(spins);
    }
    impl_->current.store(nullptr, std::memory_order_seq_cst);
    spins = 0;
    while (impl_->busy.load(std::memory_order_seq_cst) != 0) {
        backoff(spins);
    }
    t_inside_pool = false;

    if (job.error) {
        std::rethrow_exception(job.error);
    }
}

}  // namespace matrix_pro
