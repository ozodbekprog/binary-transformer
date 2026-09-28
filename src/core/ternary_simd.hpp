// SPDX-License-Identifier: MIT
// ternary_simd.hpp — Multiplication-less ternar/binar matmul yadrolari.
//
// Asosiy g'oya (bit-plane usuli):
//   a ∈ {-1,0,+1} -> (pos_a, neg_a) bitlari
//   a·b = [pos_a & pos_b] + [neg_a & neg_b] - [pos_a & neg_b] - [neg_a & pos_b]
//   => barcha ko'paytirishlar XNOR/AND + POPCNT bilan bajariladi.
//
// Yadrolar: scalar (portativ), AVX2 (LUT-popcount), AVX-512 (VPOPCNTDQ, ixtiyoriy),
// ARM NEON (cnt). Runtime dispatch `detect_best()` orqali.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/bitpack.hpp"

namespace bt::cpu {

// ---------------------------------------------------------------------------
// Matritsa ko'rinishlari
// ---------------------------------------------------------------------------

// Ternar matritsa [rows x cols], bit-planes uint64 so'zlarda (K bo'ylab packed).
struct TernaryMatrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t stride64 = 0;              // words64(cols)
    std::vector<std::uint64_t> pos, neg;   // rows*stride64 so'z

    void resize(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        stride64 = words64(c);
        pos.assign(rows * stride64, 0);
        neg.assign(rows * stride64, 0);
    }
    // Ustunlar K bo'ylab packed (matmul'ning B tomoni) [K x N]:
    // stride rows (K) bo'yicha hisoblanadi.
    void resize_packed_cols(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        stride64 = words64(r);
        pos.assign(rows * stride64, 0);
        neg.assign(rows * stride64, 0);
    }
    std::size_t bytes() const noexcept {
        return (pos.size() + neg.size()) * sizeof(std::uint64_t);
    }
};

// GPU yuklash uchun uint32 variant (GLSL bitCount 32-bit).
struct TernaryMatrix32 {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t stride32 = 0;              // words32(cols)
    std::vector<std::uint32_t> pos, neg;

    void resize(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        stride32 = words32(c);
        pos.assign(rows * stride32, 0);
        neg.assign(rows * stride32, 0);
    }
    void resize_packed_cols(std::size_t r, std::size_t c) {
        rows = r;
        cols = c;
        stride32 = words32(r);
        pos.assign(rows * stride32, 0);
        neg.assign(rows * stride32, 0);
    }
    std::size_t bytes() const noexcept {
        return (pos.size() + neg.size()) * sizeof(std::uint32_t);
    }
};

// ---------------------------------------------------------------------------
// Tiling / blocking konfiguratsiyasi — cache miss va branch miss'ni kamaytiradi.
// ---------------------------------------------------------------------------

struct TileConfig {
    std::size_t k_block_words = 16;  // bir marta ishlanadigan K so'zlari (64-bit)
    std::size_t n_block = 4;         // N (ustunlar) bloki
    std::size_t m_block = 64;        // M (qatorlar) bloki (thread bo'linishi)
};

enum class Kernel { Auto, Scalar, AVX2, AVX512, NEON };

const char* kernel_name(Kernel k) noexcept;
Kernel detect_best() noexcept;          // CPU imkoniyatlariga qarab
int hardware_threads() noexcept;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// A [M x K] ternar, B [K x N] ternar (B ustunlari K bo'ylab packed) ->
// C [M x N] int32. `threads` <= 0 => hardware_threads().
void ternary_matmul(const TernaryMatrix& A,
                    const TernaryMatrix& B,
                    std::int32_t* C,
                    const TileConfig& tile = {},
                    Kernel kernel = Kernel::Auto,
                    int threads = 0);

// Taqqoslash uchun sekin, lekin aniq implementatsiya (self-test uchun).
void ternary_matmul_ref(const TernaryMatrix& A,
                        const TernaryMatrix& B,
                        std::int32_t* C);

// int8 qiymatlardan to'g'ridan-to'g'ri packing.
void pack_into(TernaryMatrix& out, const std::int8_t* values);
void pack_into32(TernaryMatrix32& out, const std::int8_t* values);

// Bitta qator uchun yadro (GPU bilan solishtirish / test uchun).
std::int64_t dot_ternary(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                         const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                         std::size_t words, Kernel kernel = Kernel::Auto);

}  // namespace bt::cpu
