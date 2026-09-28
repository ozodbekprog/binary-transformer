// SPDX-License-Identifier: MIT
// ternary_simd.cpp — multiplication-less ternar matmul yadrolari.
#include "core/ternary_simd.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define BT_X86 1
#endif

#if defined(BT_X86)
#  include <immintrin.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  define BT_NEON 1
#  include <arm_neon.h>
#endif

#if defined(_MSC_VER)
#  include <intrin.h>
#  define BT_POPCNT64(v) static_cast<int>(__popcnt64(v))
#else
#  define BT_POPCNT64(v) __builtin_popcountll(v)
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define BT_TARGET(x) __attribute__((target(x)))
#else
#  define BT_TARGET(x)
#endif

namespace bt::cpu {
namespace {

// ---------------------------------------------------------------------------
// Scalar yadro — barcha platformalarda ishlaydi (dispatch fallback).
// ---------------------------------------------------------------------------
std::int64_t dot_scalar(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                        const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                        std::size_t words) noexcept {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < words; ++i) {
        sum += BT_POPCNT64(a_pos[i] & b_pos[i]);
        sum += BT_POPCNT64(a_neg[i] & b_neg[i]);
        sum -= BT_POPCNT64(a_pos[i] & b_neg[i]);
        sum -= BT_POPCNT64(a_neg[i] & b_pos[i]);
    }
    return sum;
}

#if defined(BT_X86)
// ---------------------------------------------------------------------------
// AVX2 yadro — XNOR/AND + nibble-LUT popcount (PSHUFB) + SAD akkumulyatsiya.
// 256 bit (4 so'z) bir iteratsiyada; popcount protsessori yo'q, LUT ishlatiladi.
// ---------------------------------------------------------------------------
BT_TARGET("avx2") static inline __m256i pc_avx2(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo),
                           _mm256_shuffle_epi8(lut, hi));
}

BT_TARGET("avx2")
std::int64_t dot_avx2(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                      const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                      std::size_t words) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i offset = _mm256_set1_epi8(32);  // manfiy baytlarni surish

    __m256i acc = _mm256_setzero_si256();
    std::size_t i = 0;
    std::size_t processed_bytes = 0;
    for (; i + 4 <= words; i += 4) {
        const __m256i ap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_pos + i));
        const __m256i an = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_neg + i));
        const __m256i bp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_pos + i));
        const __m256i bn = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_neg + i));

        __m256i c = _mm256_add_epi8(pc_avx2(_mm256_and_si256(ap, bp)),
                                    pc_avx2(_mm256_and_si256(an, bn)));
        c = _mm256_sub_epi8(c, pc_avx2(_mm256_and_si256(ap, bn)));
        c = _mm256_sub_epi8(c, pc_avx2(_mm256_and_si256(an, bp)));
        c = _mm256_add_epi8(c, offset);  // baytlarni [16..48] diapazonga surish
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, zero));
        processed_bytes += 32;
    }

    alignas(32) std::int64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::int64_t total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    total -= static_cast<std::int64_t>(32) * static_cast<std::int64_t>(processed_bytes);
    total += dot_scalar(a_pos + i, a_neg + i, b_pos + i, b_neg + i, words - i);
    return total;
}

// ---------------------------------------------------------------------------
// AVX-512 yadro (VPOPCNTDQ) — mavjud bo'lsa eng tez yo'l.
// ---------------------------------------------------------------------------
BT_TARGET("avx512f,avx512vpopcntdq")
std::int64_t dot_avx512(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                        const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                        std::size_t words) noexcept {
    __m512i acc = _mm512_setzero_si512();
    std::size_t i = 0;
    for (; i + 8 <= words; i += 8) {
        const __m512i ap = _mm512_loadu_si512(a_pos + i);
        const __m512i an = _mm512_loadu_si512(a_neg + i);
        const __m512i bp = _mm512_loadu_si512(b_pos + i);
        const __m512i bn = _mm512_loadu_si512(b_neg + i);

        __m512i c = _mm512_popcnt_epi64(_mm512_and_si512(ap, bp));
        c = _mm512_add_epi64(c, _mm512_popcnt_epi64(_mm512_and_si512(an, bn)));
        c = _mm512_sub_epi64(c, _mm512_popcnt_epi64(_mm512_and_si512(ap, bn)));
        c = _mm512_sub_epi64(c, _mm512_popcnt_epi64(_mm512_and_si512(an, bp)));
        acc = _mm512_add_epi64(acc, c);
    }
    alignas(64) std::int64_t lanes[8];
    _mm512_store_si512(lanes, acc);
    std::int64_t total = 0;
    for (int k = 0; k < 8; ++k) total += lanes[k];
    total += dot_scalar(a_pos + i, a_neg + i, b_pos + i, b_neg + i, words - i);
    return total;
}
#endif  // BT_X86

#if defined(BT_NEON)
// ---------------------------------------------------------------------------
// ARM NEON yadro — VCNT (per-byte popcount) + SAD.
// ---------------------------------------------------------------------------
std::int64_t dot_neon(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                      const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                      std::size_t words) noexcept {
    std::int32x4_t acc = vdupq_n_s32(0);
    std::size_t i = 0;
    for (; i + 2 <= words; i += 2) {
        const uint64x2_t ap = vld1q_u64(a_pos + i);
        const uint64x2_t an = vld1q_u64(a_neg + i);
        const uint64x2_t bp = vld1q_u64(b_pos + i);
        const uint64x2_t bn = vld1q_u64(b_neg + i);

        const uint8x16_t pp = vcntq_u8(vreinterpretq_u8_u64(vandq_u64(ap, bp)));
        const uint8x16_t nn = vcntq_u8(vreinterpretq_u8_u64(vandq_u64(an, bn)));
        const uint8x16_t pn = vcntq_u8(vreinterpretq_u8_u64(vandq_u64(ap, bn)));
        const uint8x16_t np = vcntq_u8(vreinterpretq_u8_u64(vandq_u64(an, bp)));

        const int8x16_t c = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(pp),
                                              vreinterpretq_s8_u8(nn)),
                                     vaddq_s8(vreinterpretq_s8_u8(pn),
                                              vreinterpretq_s8_u8(np)));
        acc = vaddq_s32(acc, vpaddlq_s16(vmovl_s8(vget_low_s8(c))));
        acc = vaddq_s32(acc, vpaddlq_s16(vmovl_s8(vget_high_s8(c))));
    }
    std::int64_t total = vaddvq_s32(acc);
    total += dot_scalar(a_pos + i, a_neg + i, b_pos + i, b_neg + i, words - i);
    return total;
}
#endif  // BT_NEON

inline std::int64_t dot_dispatch(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                                 const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                                 std::size_t words, Kernel kernel) noexcept {
    switch (kernel) {
#if defined(BT_X86)
        case Kernel::AVX512: return dot_avx512(a_pos, a_neg, b_pos, b_neg, words);
        case Kernel::AVX2:   return dot_avx2(a_pos, a_neg, b_pos, b_neg, words);
#endif
#if defined(BT_NEON)
        case Kernel::NEON:   return dot_neon(a_pos, a_neg, b_pos, b_neg, words);
#endif
        case Kernel::Scalar:
        default:             return dot_scalar(a_pos, a_neg, b_pos, b_neg, words);
    }
}

// ---------------------------------------------------------------------------
// ThreadPool — doimiy worker'lar. Har layer uchun thread yaratish 10-100x
// qimmat bo'lgani uchun bitta pool qayta ishlatiladi (transformersda
// yuzlab matmul chaqiruvi bo'ladi).
// ---------------------------------------------------------------------------
class ThreadPool {
public:
    static ThreadPool& instance() {
        static ThreadPool pool;
        return pool;
    }

    // Bitta job'ni barcha worker'larga yuboradi va tugashini kutadi.
    void run(const std::function<void()>& job) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stop_) return;
            job_ = &job;
            active_ = workers_.size();
            ++gen_;  // yangi avlod: har worker aynan bir marta ishlaydi
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&] { return active_ == 0; });
        job_ = nullptr;
    }

    std::size_t size() const { return workers_.size(); }

private:
    ThreadPool() {
        const unsigned hw = std::thread::hardware_concurrency();
        const std::size_t n = hw == 0 ? 1 : hw;
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void worker_loop() {
        std::size_t seen = 0;
        for (;;) {
            const std::function<void()>* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                job = job_;
            }
            (*job)();  // job o'z ulushini bajaradi (ichida blok taqsimoti bor)
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--active_ == 0) done_cv_.notify_one();
            }
        }
    }

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    const std::function<void()>* job_ = nullptr;
    std::size_t active_ = 0;
    std::size_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// Ommaviy API
// ---------------------------------------------------------------------------

const char* kernel_name(Kernel k) noexcept {
    switch (k) {
        case Kernel::Scalar: return "scalar";
        case Kernel::AVX2:   return "avx2";
        case Kernel::AVX512: return "avx512";
        case Kernel::NEON:   return "neon";
        case Kernel::Auto:
        default:             return "auto";
    }
}

Kernel detect_best() noexcept {
#if defined(BT_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vpopcntdq") &&
        __builtin_cpu_supports("avx512f")) {
        return Kernel::AVX512;
    }
    if (__builtin_cpu_supports("avx2")) {
        return Kernel::AVX2;
    }
#endif
#if defined(BT_NEON)
    return Kernel::NEON;
#else
    return Kernel::Scalar;
#endif
}

int hardware_threads() noexcept {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

void pack_into(TernaryMatrix& out, const std::int8_t* values) {
    bt::pack_ternary(values, out.rows, out.cols, out.stride64,
                     out.pos.data(), out.neg.data());
}

void pack_into32(TernaryMatrix32& out, const std::int8_t* values) {
    bt::pack_ternary32(values, out.rows, out.cols, out.stride32,
                       out.pos.data(), out.neg.data());
}

std::int64_t dot_ternary(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                         const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                         std::size_t words, Kernel kernel) {
    if (kernel == Kernel::Auto) kernel = detect_best();
    return dot_dispatch(a_pos, a_neg, b_pos, b_neg, words, kernel);
}

void ternary_matmul_ref(const TernaryMatrix& A, const TernaryMatrix& B, std::int32_t* C) {
    for (std::size_t m = 0; m < A.rows; ++m) {
        for (std::size_t n = 0; n < B.cols; ++n) {
            const std::int64_t s = dot_scalar(
                A.pos.data() + m * A.stride64, A.neg.data() + m * A.stride64,
                B.pos.data() + n * B.stride64, B.neg.data() + n * B.stride64,
                A.stride64);
            C[m * B.cols + n] = static_cast<std::int32_t>(s);
        }
    }
}

void ternary_matmul(const TernaryMatrix& A, const TernaryMatrix& B, std::int32_t* C,
                    const TileConfig& tile, Kernel kernel, int threads) {
    if (A.cols != B.rows || A.stride64 != B.stride64) {
        std::memset(C, 0, A.rows * B.cols * sizeof(std::int32_t));
        return;
    }
    if (kernel == Kernel::Auto) kernel = detect_best();
    if (threads <= 0) threads = hardware_threads();

    std::memset(C, 0, A.rows * B.cols * sizeof(std::int32_t));

    const std::size_t m_block = std::max<std::size_t>(1, tile.m_block);
    const std::size_t k_block = std::max<std::size_t>(1, tile.k_block_words);

    std::atomic<std::size_t> next_row{0};
    auto worker = [&]() {
        for (;;) {
            const std::size_t m0 = next_row.fetch_add(m_block);
            if (m0 >= A.rows) break;
            const std::size_t m1 = std::min(m0 + m_block, A.rows);

            for (std::size_t kb = 0; kb < A.stride64; kb += k_block) {
                const std::size_t klen = std::min(k_block, A.stride64 - kb);
                for (std::size_t m = m0; m < m1; ++m) {
                    const std::uint64_t* ap = A.pos.data() + m * A.stride64 + kb;
                    const std::uint64_t* an = A.neg.data() + m * A.stride64 + kb;
                    std::int32_t* crow = C + m * B.cols;
                    for (std::size_t n = 0; n < B.cols; ++n) {
                        const std::uint64_t* bp = B.pos.data() + n * B.stride64 + kb;
                        const std::uint64_t* bn = B.neg.data() + n * B.stride64 + kb;
                        crow[n] += static_cast<std::int32_t>(
                            dot_dispatch(ap, an, bp, bn, klen, kernel));
                    }
                }
            }
        }
    };

    const int nthreads = std::min<int>(threads, static_cast<int>(ceil_div(A.rows, m_block)));
    if (nthreads <= 1) {
        worker();
        return;
    }
    // Doimiy thread pool: har layer uchun thread yaratish o'rniga qayta ishlatamiz.
    std::atomic<int> slots{nthreads};
    ThreadPool& pool = ThreadPool::instance();
    pool.run([&]() {
        if (slots.fetch_sub(1) <= 0) return;  // ortiqcha worker'lar darhol chiqadi
        worker();
    });
}

}  // namespace bt::cpu
