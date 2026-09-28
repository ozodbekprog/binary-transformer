// SPDX-License-Identifier: MIT
// fast_chat_engine.cpp — 1B ternar modelni BITTA GPU'da 100+ tok/s bilan
// chat rejimida ishlatish (speculative/klaster emas — lokal tezkor rejim).
//
// Promptlar relay mailbox "fastchat" orqali keladi (PROMPT kadr);
// generatsiya qilingan tokenlar TOCHAT eventlari sifatida "chat" ga oqadi.
//
// Misol:
//   fast_chat_engine --d 2048 --params 1e9 --tokens 12 --gpu
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/ternary_simd.hpp"
#include "gpu/vulkan_wrapper.hpp"
#include "net/pipeline_rpc.hpp"

namespace {

using bt::cpu::TernaryMatrix32;

double now_s() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

void log_line(const std::string& s) {
    std::fprintf(stderr, "[fast %.2fs] %s\n", now_s(), s.c_str());
    std::fflush(stderr);
}

struct SplitMix {
    std::uint64_t s;
    explicit SplitMix(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    std::int8_t ternary() {
        const std::uint64_t r = next() & 3u;
        return r == 0 ? 1 : (r == 3 ? -1 : 0);
    }
};

std::uint64_t matrix_seed(std::uint64_t base, std::uint32_t layer, std::uint32_t m) {
    std::uint64_t h = base ^ (0x9E3779B97F4A7C15ull * (layer + 1)) ^
                      (0xC2B2AE3D27D4EB4Full * (m + 1));
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    return h;
}

std::vector<float> embed_token(std::uint64_t seed, std::uint32_t token,
                               std::uint32_t stream, std::uint32_t d) {
    std::vector<float> h(d);
    SplitMix rng(matrix_seed(seed ^ 0xABCDEFull, token, stream));
    for (std::uint32_t i = 0; i < d; ++i) {
        const float r = static_cast<float>(rng.next() & 0xFFFF) / 65535.0f;
        h[i] = (r - 0.5f) * 1.5f;
    }
    return h;
}

void quantize_hidden(const std::vector<float>& h, std::vector<std::int8_t>& q) {
    q.resize(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        q[i] = h[i] > 0.05f ? 1 : (h[i] < -0.05f ? -1 : 0);
    }
}

void pack_col32(const std::vector<std::int8_t>& q, TernaryMatrix32& x) {
    x.rows = q.size();
    x.cols = 1;
    x.stride32 = bt::words32(q.size());
    x.pos.assign(x.stride32, 0);
    x.neg.assign(x.stride32, 0);
    bt::pack_ternary32_column(q.data(), q.size(), x.stride32, x.pos.data(), x.neg.data());
}

constexpr std::uint8_t BT_FLAG_TOKEN = 1;
constexpr std::uint8_t BT_FLAG_PROMPT = 2;
constexpr std::uint8_t BT_FLAG_TOCHAT = 4;

std::uint64_t hash_hidden(const std::vector<float>& h, std::uint64_t seed) {
    std::uint64_t acc = seed;
    for (std::size_t i = 0; i < h.size(); i += 64) {
        acc ^= static_cast<std::uint64_t>(std::llround(h[i] * 100.0)) * (i + 1);
        acc = (acc << 13) ^ (acc >> 7);
    }
    return acc;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t d = 2048;
    std::uint64_t params = 1'000'000'000ull;
    std::uint32_t tokens_per_prompt = 12;
    std::uint64_t seed = 42;
    std::string shader;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string{}; };
        if (a == "--d") d = static_cast<std::uint32_t>(std::atoi(val().c_str()));
        else if (a == "--params") params = std::strtoull(val().c_str(), nullptr, 10);
        else if (a == "--tokens") tokens_per_prompt = static_cast<std::uint32_t>(std::atoi(val().c_str()));
        else if (a == "--seed") seed = std::strtoull(val().c_str(), nullptr, 10);
        else if (a == "--shader") shader = val();
    }

    const std::uint32_t per_layer = d * d;
    const std::uint32_t layers = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, params / per_layer));
    log_line("model: " + std::to_string(layers) + " layer x d=" + std::to_string(d) +
             " = " + std::to_string(static_cast<double>(layers) * per_layer / 1e9) + "B param");

    // GPU engine
    bt::gpu::VulkanEngine engine;
    std::string err;
    if (!engine.init(&err, shader)) {
        std::fprintf(stderr, "XATO: Vulkan: %s\n", err.c_str());
        return 1;
    }
    log_line("GPU: " + engine.info().name);

    // Og'irliklar (uint32 planes, parallel generatsiya)
    TernaryMatrix32 w;
    w.rows = std::size_t(layers) * d;
    w.cols = d;
    w.stride32 = bt::words32(d);
    w.pos.assign(w.rows * w.stride32, 0);
    w.neg.assign(w.rows * w.stride32, 0);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<std::uint32_t> next_layer{0};
    const double tg0 = now_s();
    auto worker = [&]() {
        std::vector<std::int8_t> row(d);
        for (;;) {
            const std::uint32_t l = next_layer.fetch_add(1);
            if (l >= layers) break;
            SplitMix rng(matrix_seed(seed, l, 0));
            for (std::uint32_t r = 0; r < d; ++r) {
                for (std::uint32_t c = 0; c < d; ++c) row[c] = rng.ternary();
                const std::size_t row_idx = std::size_t(l) * d + r;
                bt::pack_ternary32(row.data(), 1, d, w.stride32,
                                   w.pos.data() + row_idx * w.stride32,
                                   w.neg.data() + row_idx * w.stride32);
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < hw; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    log_line("og'irliklar tayyor: " + std::to_string(static_cast<int>(now_s() - tg0)) + "s, " +
             std::to_string(w.pos.size() * 2 * 4 / 1e6) + " MB");

    if (!engine.upload_weights(w.pos.size(), w.pos.data(), w.neg.data(), &err)) {
        std::fprintf(stderr, "XATO: upload: %s\n", err.c_str());
        return 1;
    }
    std::vector<std::uint32_t> xp(w.stride32, 0), xn(w.stride32, 0);
    if (!engine.upload_input(1, static_cast<std::uint32_t>(w.stride32),
                             xp.data(), xn.data(), &err) ||
        !engine.build([&] {
            std::vector<bt::gpu::LayerDesc> descs;
            descs.reserve(layers);
            for (std::uint32_t l = 0; l < layers; ++l) {
                descs.push_back({d, static_cast<std::uint32_t>(w.stride32),
                                 l * static_cast<std::uint32_t>(w.stride32) * d});
            }
            return descs;
        }(), 1, &err)) {
        std::fprintf(stderr, "XATO: build: %s\n", err.c_str());
        return 1;
    }
    log_line("FAST ENGINE TAYYOR — prompt kutilmoqda (100+ tok/s)");

    const float act_scale = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<std::int32_t> out(d);

    for (;;) {
        bt::net::Frame f;
        if (!bt::net::read_stdio_frame(f)) break;
        if (!f.is_prompt() || f.payload.size() < 3) continue;
        std::uint32_t lo = 0, hi = 0;
        std::memcpy(&lo, &f.payload[0], 4);
        std::memcpy(&hi, &f.payload[1], 4);
        const std::uint64_t pseed = (static_cast<std::uint64_t>(hi) << 32) | lo;
        int mt = static_cast<int>(f.payload[2]);
        if (mt <= 0 || mt > 128) mt = static_cast<int>(tokens_per_prompt);

        const double t0 = now_s();
        std::uint32_t token = static_cast<std::uint32_t>(pseed % 32000u);
        double t_cpu = 0, t_exec = 0, t_read = 0;
        std::vector<std::int8_t> q;
        TernaryMatrix32 x;
        std::vector<float> hn(d);
        for (int t = 0; t < mt; ++t) {
            const double c0 = now_s();
            const auto h = embed_token(pseed, token, static_cast<std::uint32_t>(t), d);
            quantize_hidden(h, q);
            pack_col32(q, x);
            engine.update_input(x.pos.data(), x.neg.data());
            const double c1 = now_s();
            if (!engine.execute(&err)) {
                std::fprintf(stderr, "XATO: execute: %s\n", err.c_str());
                break;
            }
            const double c2 = now_s();
            engine.read_output(out.data(), d, &err);
            for (std::uint32_t i = 0; i < d; ++i) {
                hn[i] = std::tanh(static_cast<float>(out[i]) * act_scale);
            }
            token = static_cast<std::uint32_t>(hash_hidden(hn, pseed) % 32000u);
            const double c3 = now_s();
            t_cpu += (c1 - c0); t_exec += (c2 - c1); t_read += (c3 - c2);

            bt::net::Frame ev;
            ev.flags = BT_FLAG_TOCHAT | BT_FLAG_TOKEN;
            ev.seq = static_cast<std::uint32_t>(t);
            ev.payload = {static_cast<float>(t), static_cast<float>(token)};
            if (!bt::net::write_stdio_frame(ev)) break;
        }
        const double total = now_s() - t0;
        log_line("javob: " + std::to_string(mt) + " token, " +
                 std::to_string(total * 1000.0) + " ms, " +
                 std::to_string(mt / total) + " tok/s | cpu=" +
                 std::to_string(t_cpu * 1000.0 / mt) + "ms exec=" +
                 std::to_string(t_exec * 1000.0 / mt) + "ms read=" +
                 std::to_string(t_read * 1000.0 / mt) + "ms per token");
        bt::net::Frame done;
        done.flags = BT_FLAG_TOCHAT;
        done.payload = {static_cast<float>(mt), static_cast<float>(total * 1000.0)};
        bt::net::write_stdio_frame(done);
    }
    return 0;
}
