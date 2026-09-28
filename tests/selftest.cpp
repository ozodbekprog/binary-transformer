// SPDX-License-Identifier: MIT
// selftest.cpp — packing, yadrolar va reference taqqoslash (ci uchun).
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/ternary_simd.hpp"

namespace {

struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    std::int8_t ternary() {
        const auto r = next() & 3u;
        return r == 0 ? 1 : (r == 3 ? -1 : 0);
    }
};

int failures = 0;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::printf("[FAIL] %s\n", what);
        ++failures;
    } else {
        std::printf("[ok]   %s\n", what);
    }
}

}  // namespace

int main() {
    std::printf("bt_selftest — packing + yadro tekshiruvi\n");

    // 1) Packing roundtrip
    {
        Rng rng(7);
        const std::size_t M = 13, K = 129;
        std::vector<std::int8_t> vals(M * K);
        for (auto& v : vals) v = rng.ternary();
        bt::cpu::TernaryMatrix A;
        A.resize(M, K);
        bt::cpu::pack_into(A, vals.data());
        bool ok = true;
        for (std::size_t r = 0; r < M && ok; ++r) {
            for (std::size_t c = 0; c < K && ok; ++c) {
                if (bt::unpack_ternary(A.pos.data(), A.neg.data(), r, c, A.stride64) !=
                    vals[r * K + c]) {
                    ok = false;
                }
            }
        }
        expect(ok, "ternar packing roundtrip (64-bit)");
    }

    // 2) CPU yadrolar vs reference
    {
        Rng rng(1234);
        const std::size_t M = 67, K = 259, N = 5;
        std::vector<std::int8_t> buf(std::max(M * K, K * N));
        for (auto& v : buf) v = rng.ternary();

        bt::cpu::TernaryMatrix A, B;
        A.resize(M, K);
        B.resize_packed_cols(K, N);  // B: ustunlar K bo'ylab packed
        bt::cpu::pack_into(A, buf.data());
        for (auto& v : buf) v = rng.ternary();
        // B ustunlari K bo'ylab packed bo'lishi kerak:
        std::vector<std::int8_t> col(K), bmat(K * N);
        for (std::size_t c = 0; c < N; ++c) {
            for (std::size_t r = 0; r < K; ++r) {
                col[r] = rng.ternary();
                bmat[r * N + c] = col[r];
            }
            bt::pack_ternary_column(col.data(), K, B.stride64,
                                    B.pos.data() + c * B.stride64,
                                    B.neg.data() + c * B.stride64);
        }
        (void)bmat;

        std::vector<std::int32_t> ref(M * N), got(M * N);
        bt::cpu::ternary_matmul_ref(A, B, ref.data());

        const bt::cpu::Kernel kernels[] = {
            bt::cpu::Kernel::Scalar, bt::cpu::Kernel::Auto};
        for (const auto k : kernels) {
            bt::cpu::ternary_matmul(A, B, got.data(), {}, k, 3);
            expect(ref == got, k == bt::cpu::Kernel::Scalar
                                  ? "scalar matmul == reference"
                                  : "auto(avx2/neon) matmul == reference");
        }

        // K blokirovkasi (tiling) bilan ham bir xil natija
        bt::cpu::TileConfig tc;
        tc.k_block_words = 3;  // ataylab kichik blok
        bt::cpu::ternary_matmul(A, B, got.data(), tc, bt::cpu::Kernel::Auto, 2);
        expect(ref == got, "tiling (k_block=3) matmul == reference");
    }

    // 3) Katta o'lcham (d_model uslubida) AVX2 tail ishlashi
    {
        Rng rng(99);
        const std::size_t M = 200, K = 2048;
        std::vector<std::int8_t> vals(M * K);
        for (auto& v : vals) v = rng.ternary();
        bt::cpu::TernaryMatrix A;
        A.resize(M, K);
        bt::cpu::pack_into(A, vals.data());

        std::vector<std::int8_t> col(K);
        for (auto& v : col) v = rng.ternary();
        bt::cpu::TernaryMatrix B;
        B.resize_packed_cols(K, 1);
        bt::pack_ternary_column(col.data(), K, B.stride64, B.pos.data(), B.neg.data());

        std::vector<std::int32_t> ref(M), got(M);
        bt::cpu::ternary_matmul_ref(A, B, ref.data());
        bt::cpu::ternary_matmul(A, B, got.data(), {}, bt::cpu::Kernel::Auto, 4);
        expect(ref == got, "d=2048 matvec == reference");
    }

    std::printf(failures == 0 ? "\nSELFTEST: PASS\n" : "\nSELFTEST: FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
