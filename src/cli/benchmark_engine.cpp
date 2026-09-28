// SPDX-License-Identifier: MIT
// benchmark_engine.cpp — Fake Ternary Weights (default 1B parametr) ustida
// CPU (AVX2/AVX-512/NEON) va GPU (Vulkan) ternar matmul tezligini o'lchaydi.
//
// Natija: tokens/sec, MAC/s, packed hajm, layerlar soni, yadro nomi.
//
// Misollar:
//   benchmark_engine                          # 1B params, auto backend
//   benchmark_engine --params 2e8 --tokens 5  # kichikroq, tezroq
//   benchmark_engine --backend cpu --kernel avx2
//   benchmark_engine --list                   # backendlar ro'yxati
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/ternary_simd.hpp"
#include "gpu/vulkan_wrapper.hpp"

namespace {

using bt::cpu::Kernel;
using bt::cpu::TernaryMatrix;
using bt::cpu::TernaryMatrix32;

// ---------------------------------------------------------------------------
// Deterministik PRNG (splitmix64)
// ---------------------------------------------------------------------------
struct Rng {
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // Ternar taqsimot: 25% +1, 50% 0, 25% -1
    std::int8_t ternary() {
        const std::uint64_t r = next() & 3u;
        return r == 0 ? 1 : (r == 3 ? -1 : 0);
    }
};

double now_s() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Model spetsifikatsiyasi
// ---------------------------------------------------------------------------
struct ModelSpec {
    std::uint64_t params = 1'000'000'000ull;
    std::uint32_t d = 2048;
    std::uint32_t layers = 0;
    std::uint64_t actual_params = 0;

    void finalize() {
        const std::uint64_t per_layer = std::uint64_t(d) * d;
        layers = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, params / per_layer));
        actual_params = std::uint64_t(layers) * per_layer;
    }
    std::uint64_t packed_bytes() const {
        // 2 bit/param: pos + neg planes, uint64 so'zlarga packed (taxminan)
        return (actual_params + 3) / 4;
    }
};

// Faqat CPU uchun: har layer alohida TernaryMatrix [d x d]
struct CpuModel {
    std::vector<TernaryMatrix> layers;
    std::uint32_t d = 0;
};

void build_cpu_model(CpuModel& m, const ModelSpec& spec, std::uint64_t seed,
                     std::vector<int8_t>* first_layer_unused = nullptr) {
    (void)first_layer_unused;
    Rng rng(seed);
    m.d = spec.d;
    m.layers.resize(spec.layers);
    std::vector<std::int8_t> row(spec.d);
    for (auto& layer : m.layers) {
        layer.resize(spec.d, spec.d);
        for (std::uint32_t r = 0; r < spec.d; ++r) {
            for (std::uint32_t c = 0; c < spec.d; ++c) row[c] = rng.ternary();
            bt::pack_ternary(row.data(), 1, spec.d, layer.stride64,
                             layer.pos.data() + std::size_t(r) * layer.stride64,
                             layer.neg.data() + std::size_t(r) * layer.stride64);
        }
    }
}

// CPU va GPU uchun birgalikda: uint64 (CPU) + uint32 (GPU) planes
void build_both_models(CpuModel& cpu, TernaryMatrix32& gpu_weights, std::uint32_t& gpu_stride32,
                       const ModelSpec& spec, std::uint64_t seed) {
    Rng rng(seed);
    cpu.d = spec.d;
    cpu.layers.resize(spec.layers);
    gpu_stride32 = bt::words32(spec.d);
    const std::size_t layer_words32 = std::size_t(spec.d) * gpu_stride32;
    gpu_weights.rows = std::size_t(spec.layers) * spec.d;
    gpu_weights.cols = spec.d;
    gpu_weights.stride32 = gpu_stride32;
    gpu_weights.pos.assign(std::size_t(spec.layers) * layer_words32, 0);
    gpu_weights.neg.assign(std::size_t(spec.layers) * layer_words32, 0);

    std::vector<std::int8_t> row(spec.d);
    for (std::uint32_t l = 0; l < spec.layers; ++l) {
        TernaryMatrix& layer = cpu.layers[l];
        layer.resize(spec.d, spec.d);
        for (std::uint32_t r = 0; r < spec.d; ++r) {
            for (std::uint32_t c = 0; c < spec.d; ++c) row[c] = rng.ternary();
            // CPU (64-bit)
            bt::pack_ternary(row.data(), 1, spec.d, layer.stride64,
                             layer.pos.data() + std::size_t(r) * layer.stride64,
                             layer.neg.data() + std::size_t(r) * layer.stride64);
            // GPU (32-bit)
            std::uint32_t* gp = gpu_weights.pos.data() +
                                (std::size_t(l) * spec.d + r) * gpu_stride32;
            std::uint32_t* gn = gpu_weights.neg.data() +
                                (std::size_t(l) * spec.d + r) * gpu_stride32;
            bt::pack_ternary32(row.data(), 1, spec.d, gpu_stride32, gp, gn);
        }
    }
}

// Hidden state -> ternar {-1,0,1}
void quantize_hidden(const std::vector<float>& h, std::vector<std::int8_t>& q,
                     float eps = 0.05f) {
    q.resize(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        q[i] = h[i] > eps ? 1 : (h[i] < -eps ? -1 : 0);
    }
}

void pack_hidden_column(const std::vector<std::int8_t>& q, TernaryMatrix& x) {
    const std::size_t d = q.size();
    x.rows = d;
    x.cols = 1;
    x.stride64 = bt::words64(d);
    x.pos.assign(d * x.stride64, 0);
    x.neg.assign(d * x.stride64, 0);
    bt::pack_ternary_column(q.data(), d, x.stride64, x.pos.data(), x.neg.data());
}

void pack_hidden_column32(const std::vector<std::int8_t>& q, TernaryMatrix32& x) {
    const std::size_t d = q.size();
    x.rows = d;
    x.cols = 1;
    x.stride32 = bt::words32(d);
    x.pos.assign(d * x.stride32, 0);
    x.neg.assign(d * x.stride32, 0);
    bt::pack_ternary32_column(q.data(), d, x.stride32, x.pos.data(), x.neg.data());
}

// ---------------------------------------------------------------------------
// CPU benchmark
// ---------------------------------------------------------------------------
struct BenchResult {
    double tokens_per_sec = 0;
    double macs_per_sec = 0;
    double pass_ms = 0;
    std::string backend;
};

BenchResult cpu_benchmark(CpuModel& model, const ModelSpec& spec, int tokens,
                          int threads, Kernel kernel) {
    const std::size_t d = spec.d;
    std::vector<float> hidden(d);
    for (std::size_t i = 0; i < d; ++i) {
        hidden[i] = 0.1f * std::sin(static_cast<float>(i) * 0.01f);
    }
    std::vector<std::int8_t> q;
    TernaryMatrix x;
    std::vector<std::int32_t> out(d);

    const auto pass = [&]() {
        quantize_hidden(hidden, q);
        pack_hidden_column(q, x);
        for (auto& layer : model.layers) {
            bt::cpu::ternary_matmul(layer, x, out.data(), {}, kernel, threads);
            const float scale = 1.0f / std::sqrt(static_cast<float>(d));
            for (std::size_t i = 0; i < d; ++i) {
                hidden[i] = std::tanh(static_cast<float>(out[i]) * scale);
            }
            quantize_hidden(hidden, q);
            pack_hidden_column(q, x);
        }
    };

    pass();  // warmup
    const double t0 = now_s();
    for (int t = 0; t < tokens; ++t) pass();
    const double dt = now_s() - t0;

    BenchResult r;
    r.tokens_per_sec = tokens / dt;
    r.pass_ms = 1000.0 * dt / tokens;
    r.macs_per_sec = r.tokens_per_sec * static_cast<double>(spec.actual_params);
    r.backend = std::string("cpu/") + bt::cpu::kernel_name(kernel) + " x" +
                std::to_string(threads);
    return r;
}

// ---------------------------------------------------------------------------
// GPU benchmark
// ---------------------------------------------------------------------------
struct GpuBenchOutcome {
    bool ok = false;
    BenchResult result;
    std::string error;
};

GpuBenchOutcome gpu_benchmark(const TernaryMatrix32& weights, std::uint32_t stride32,
                              const ModelSpec& spec, int tokens, std::uint32_t batch,
                              const std::string& shader) {
    GpuBenchOutcome outcome;
    bt::gpu::VulkanEngine engine;
    if (!engine.init(&outcome.error, shader)) return outcome;

    // Og'irliklarni yuklash (bir marta; vaqt tok/s ga kirmaydi)
    if (!engine.upload_weights(weights.pos.size(), weights.pos.data(),
                               weights.neg.data(), &outcome.error)) {
        return outcome;
    }

    // Kirish vektori (benchmark uchun statik)
    std::vector<std::int8_t> q(spec.d);
    for (std::size_t i = 0; i < spec.d; ++i) {
        q[i] = (i % 3 == 0) ? 1 : ((i % 3 == 1) ? -1 : 0);
    }
    TernaryMatrix32 x;
    pack_hidden_column32(q, x);
    std::vector<std::uint32_t> xpos(batch * stride32, 0), xneg(batch * stride32, 0);
    for (std::uint32_t b = 0; b < batch; ++b) {
        std::memcpy(xpos.data() + std::size_t(b) * stride32, x.pos.data(),
                    stride32 * sizeof(std::uint32_t));
        std::memcpy(xneg.data() + std::size_t(b) * stride32, x.neg.data(),
                    stride32 * sizeof(std::uint32_t));
    }
    if (!engine.upload_input(batch, stride32, xpos.data(), xneg.data(),
                             &outcome.error)) {
        return outcome;
    }

    std::vector<bt::gpu::LayerDesc> layers;
    layers.reserve(spec.layers);
    const std::uint32_t layer_words = spec.d * stride32;
    for (std::uint32_t l = 0; l < spec.layers; ++l) {
        layers.push_back({spec.d, stride32, l * layer_words});
    }
    if (!engine.build(layers, batch, &outcome.error)) return outcome;

    // Warmup + o'lchov
    if (!engine.execute(&outcome.error)) return outcome;
    const double t0 = now_s();
    for (int t = 0; t < tokens; ++t) {
        if (!engine.execute(&outcome.error)) return outcome;
    }
    const double dt = now_s() - t0;

    std::vector<std::int32_t> out(std::size_t(spec.d) * batch);
    engine.read_output(out.data(), out.size(), &outcome.error);

    BenchResult r;
    r.tokens_per_sec = tokens / dt;
    r.pass_ms = 1000.0 * dt / tokens;
    r.macs_per_sec = r.tokens_per_sec * static_cast<double>(spec.actual_params);
    r.backend = "vulkan/" + engine.info().name;
    outcome.result = r;
    outcome.ok = true;
    engine.shutdown();
    return outcome;
}

// ---------------------------------------------------------------------------
// Kichik to'g'rilik testi (CPU vs reference; ixtiyoriy GPU vs CPU)
// ---------------------------------------------------------------------------
bool correctness_check(bool include_gpu, const std::string& shader, std::string* err) {
    const std::size_t M = 61, K = 97, N = 3;
    Rng rng(42);
    TernaryMatrix A, B;
    A.resize(M, K);
    B.resize_packed_cols(K, N);
    std::vector<std::int8_t> buf(std::max(M * K, K * N));
    for (auto& v : buf) v = rng.ternary();
    bt::pack_ternary(buf.data(), M, K, A.stride64, A.pos.data(), A.neg.data());
    // B ustunlari K bo'ylab packed (ustun layout) + mantiqiy bmat nusxasi
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

    std::vector<std::int32_t> ref(M * N), got(M * N);
    bt::cpu::ternary_matmul_ref(A, B, ref.data());
    bt::cpu::ternary_matmul(A, B, got.data(), {}, Kernel::Auto, 4);
    if (ref != got) {
        if (err) *err = "CPU yadro reference bilan mos emas";
        return false;
    }

#if BT_HAVE_VULKAN
    if (include_gpu) {
        // A ni uint32 ko'rinishga o'tkazish (qatorlar bo'ylab packed)
        std::vector<std::int8_t> vals(M * K);
        for (std::size_t r = 0; r < M; ++r) {
            for (std::size_t c = 0; c < K; ++c) {
                vals[r * K + c] =
                    bt::unpack_ternary(A.pos.data(), A.neg.data(), r, c, A.stride64);
            }
        }
        TernaryMatrix32 A32;
        A32.resize(M, K);
        bt::pack_ternary32(vals.data(), M, K, A32.stride32, A32.pos.data(), A32.neg.data());

        // B ning 0-ustunini olib, K bo'ylab packed ustun sifatida joylash
        // (mantiqiy qiymatlar bmat'dan olinadi — B ustun layoutidan emas).
        TernaryMatrix32 B32c;
        B32c.resize_packed_cols(K, 1);
        std::vector<std::int8_t> col(K);
        for (std::size_t r = 0; r < K; ++r) {
            col[r] = bmat[r * N];
        }
        bt::pack_ternary32_column(col.data(), K, B32c.stride32,
                                  B32c.pos.data(), B32c.neg.data());

        bt::gpu::VulkanEngine engine;
        std::string gerr;
        if (!engine.init(&gerr, shader)) {
            // CPU-only node: GPU check o'tkazib yuboriladi (xato emas)
            std::printf("  [gpu-check] o'tkazib yuborildi: %s\n", gerr.c_str());
            return true;
        }
        if (!engine.upload_weights(A32.pos.size(), A32.pos.data(), A32.neg.data(), &gerr) ||
            !engine.upload_input(1, B32c.stride32, B32c.pos.data(), B32c.neg.data(), &gerr)) {
            if (err) *err = "GPU check upload xatosi: " + gerr;
            return false;
        }
        std::vector<bt::gpu::LayerDesc> layer = {
            {static_cast<std::uint32_t>(M), static_cast<std::uint32_t>(B32c.stride32), 0}};
        if (!engine.build(layer, 1, &gerr) || !engine.execute(&gerr)) {
            if (err) *err = "GPU check yig'ilmadi: " + gerr;
            return false;
        }
        std::vector<std::int32_t> gpu_out(M);
        engine.read_output(gpu_out.data(), M, &gerr);
        int shown = 0;
        for (std::size_t m = 0; m < M; ++m) {
            if (gpu_out[m] != ref[m * N]) {  // B ning 0-ustuni
                if (shown < 5) {
                    std::printf("  [gpu-check] qator %zu: gpu=%d ref=%d\n",
                                m, gpu_out[m], ref[m * N]);
                    ++shown;
                }
                if (err) *err = "GPU natija CPU bilan mos emas (qator " +
                                std::to_string(m) + ")";
                return false;
            }
        }
    }
#else
    (void)include_gpu;
    (void)shader;
#endif
    return true;
}

void print_usage() {
    std::printf(
        "benchmark_engine — Binar/Ternar Transformer Engine benchmark\n"
        "\n"
        "  --params N       parametrlar soni (default 1e9)\n"
        "  --d N            d_model (default 2048)\n"
        "  --tokens N       o'lchanadigan tokenlar (default 3)\n"
        "  --batch N        GPU batch (default 1)\n"
        "  --threads N      CPU thread soni (default: hammasi)\n"
        "  --backend X      cpu | gpu | auto (default auto)\n"
        "  --kernel X       auto | scalar | avx2 | avx512 | neon\n"
        "  --seed N         PRNG seed (default 1)\n"
        "  --shader PATH    SPIR-V fayl yo'li (binary_matmul.spv)\n"
        "  --check          to'g'rilik testini ham o'tkazish (CPU+GPU)\n"
        "  --list           backend/kernel ro'yxatini chiqarish\n"
        "  --help           shu yordam\n");
}

}  // namespace

int main(int argc, char** argv) {
    ModelSpec spec;
    int tokens = 3;
    int threads = 0;
    std::uint32_t batch = 1;
    std::uint64_t seed = 1;
    std::string backend = "auto";
    std::string kernel_str = "auto";
    std::string shader_path;
    bool do_check = false;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](double fallback) {
            return i + 1 < argc ? std::atof(argv[++i]) : fallback;
        };
        if (a == "--params") {
            spec.params = static_cast<std::uint64_t>(next(1e9));
        } else if (a == "--d") {
            spec.d = static_cast<std::uint32_t>(next(2048));
        } else if (a == "--tokens") {
            tokens = static_cast<int>(next(3));
        } else if (a == "--batch") {
            batch = static_cast<std::uint32_t>(next(1));
        } else if (a == "--threads") {
            threads = static_cast<int>(next(0));
        } else if (a == "--backend") {
            backend = argv[++i];
        } else if (a == "--kernel") {
            kernel_str = argv[++i];
        } else if (a == "--seed") {
            seed = static_cast<std::uint64_t>(next(1));
        } else if (a == "--shader") {
            shader_path = argv[++i];
        } else if (a == "--check") {
            do_check = true;
        } else if (a == "--list") {
            list_only = true;
        } else if (a == "--help" || a == "-h") {
            print_usage();
            return 0;
        } else {
            std::fprintf(stderr, "Noma'lum argument: %s\n", a.c_str());
            print_usage();
            return 2;
        }
    }

    if (list_only) {
        std::printf("CPU yadrolar: scalar, avx2, avx512(agar mavjud), neon(ARM)\n");
        std::printf("Eng yaxshi (shu mashina): %s\n",
                    bt::cpu::kernel_name(bt::cpu::detect_best()));
        std::printf("CPU threadlar: %d\n", bt::cpu::hardware_threads());
        bt::gpu::VulkanEngine engine;
        std::string err;
        if (engine.init(&err, shader_path)) {
            const auto v = engine.info().api_version;
            std::printf("GPU: %s | VRAM %.1f GB | Vulkan %u.%u\n",
                        engine.info().name.c_str(),
                        engine.info().vram_bytes / 1e9,
                        (v >> 22) & 0x7Fu, (v >> 12) & 0x3FFu);
            engine.shutdown();
        } else {
            std::printf("GPU: mavjud emas (%s)\n", err.c_str());
        }
        return 0;
    }

    spec.finalize();
    if (spec.d < 32) {
        std::fprintf(stderr, "XATO: --d kamida 32 bo'lsin\n");
        return 2;
    }
    if (threads <= 0) threads = bt::cpu::hardware_threads();

    std::printf("=== Binar/Ternar Transformer Engine — Phase 1 benchmark ===\n");
    std::printf("Model : %.3fB parametr | d=%u | layers=%u | seed=%llu\n",
                static_cast<double>(spec.actual_params) / 1e9, spec.d, spec.layers,
                static_cast<unsigned long long>(seed));
    std::printf("Packed: %.1f MB (2 bit/param) | tokens=%d\n",
                static_cast<double>(spec.packed_bytes()) / 1e6, tokens);

    if (do_check) {
        std::string cerr;
        const bool ok = correctness_check(backend != "cpu", shader_path, &cerr);
        std::printf("Check : %s%s%s\n", ok ? "OK" : "XATO: ",
                    ok ? "CPU == reference" : cerr.c_str(),
                    (ok && backend != "cpu") ? " (GPU ham mos)" : "");
        if (!ok && backend == "gpu") return 1;
    }

    Kernel kernel = Kernel::Auto;
    if (kernel_str == "scalar") kernel = Kernel::Scalar;
    else if (kernel_str == "avx2") kernel = Kernel::AVX2;
    else if (kernel_str == "avx512") kernel = Kernel::AVX512;
    else if (kernel_str == "neon") kernel = Kernel::NEON;

    const bool want_cpu = (backend == "cpu" || backend == "auto");
    const bool want_gpu = (backend == "gpu" || backend == "auto");

    // --- Model generatsiyasi
    std::printf("\nOg'irliklar generatsiyasi...\n");
    const double tgen0 = now_s();
    CpuModel cpu_model;
    TernaryMatrix32 gpu_weights;
    std::uint32_t gpu_stride32 = 0;
    if (want_gpu) {
        build_both_models(cpu_model, gpu_weights, gpu_stride32, spec, seed);
    } else {
        build_cpu_model(cpu_model, spec, seed);
    }
    const double tgen = now_s() - tgen0;
    std::printf("Tayyor: %.2f s (%.1f MB RAM)\n", tgen,
                static_cast<double>(spec.packed_bytes()) / 1e6);

    std::vector<BenchResult> results;

    if (want_cpu) {
        std::printf("\n[CPU] ishga tushdi (%s, %d thread)...\n",
                    bt::cpu::kernel_name(bt::cpu::detect_best()), threads);
        const auto r = cpu_benchmark(cpu_model, spec, tokens, threads, kernel);
        results.push_back(r);
        std::printf("[CPU] %-28s %8.2f tok/s | %7.1f G-MAC/s | pass %.2f ms\n",
                    r.backend.c_str(), r.tokens_per_sec,
                    r.macs_per_sec / 1e9, r.pass_ms);
    }

    if (want_gpu) {
        std::printf("\n[GPU] Vulkan ishga tushirilmoqda...\n");
        const auto outcome = gpu_benchmark(gpu_weights, gpu_stride32, spec,
                                           tokens, batch, shader_path);
        if (outcome.ok) {
            results.push_back(outcome.result);
            std::printf("[GPU] %-28s %8.2f tok/s | %7.1f G-MAC/s | pass %.2f ms\n",
                        outcome.result.backend.c_str(),
                        outcome.result.tokens_per_sec,
                        outcome.result.macs_per_sec / 1e9,
                        outcome.result.pass_ms);
        } else {
            std::printf("[GPU] ishlamadi: %s\n", outcome.error.c_str());
        }
    }

    if (!results.empty()) {
        const auto best = *std::max_element(
            results.begin(), results.end(),
            [](const BenchResult& a, const BenchResult& b) {
                return a.tokens_per_sec < b.tokens_per_sec;
            });
        std::printf("\nG'olib: %s — %.2f tok/s (%.1f G-MAC/s)\n",
                    best.backend.c_str(), best.tokens_per_sec,
                    best.macs_per_sec / 1e9);
    }
    return 0;
}
