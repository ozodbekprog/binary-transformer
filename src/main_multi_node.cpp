// SPDX-License-Identifier: MIT
// main_multi_node.cpp — Phase 2: 70B-class ternar modelni Pipeline Parallelism
// bilan 4 node bo'ylab ishga tushiruvchi distributed engine.
//
// Arxitektura:
//   controller (Node1) -> worker1 (Node2) -> worker2 (Node3) -> worker3 (Node4)
//   Worker'lar hidden state (1 x d float32) ni keyingisiga uzatadi;
//   oxirgi worker generatsiya qilingan token ID ni controller'ga qaytaradi.
//
// Transport:
//   --transport tcp    : to'g'ridan-to'g'ri non-blocking sockets (LAN/Tailscale)
//   --transport stdio  : relay bridge (Python HTTP relay) bilan stdin/stdout
//
// Model (sintetik, --synth):
//   layers  : 0..79 (80 qatlam), d_model = 8192
//   layer   : 4 attention-proyeksiya (d x d) + 2 FFN blok (3 x d x d)
//             => 10 matritsa/layer = 671M param/layer ~= 53.7B jami (~13.4 GB,
//                vazifadagi 12-14 GB budjetga mos)
//
// Misol:
//   ./distributed_70b_engine --role worker --layers 12-43 --transport stdio
//   ./distributed_70b_engine --role controller --layers 0-11 --transport stdio \
//        --tokens 8 --seed 42
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

using bt::cpu::Kernel;
using bt::cpu::TernaryMatrix;
using bt::cpu::TernaryMatrix32;

// ---------------------------------------------------------------------------
// Sozlamalar
// ---------------------------------------------------------------------------
struct Config {
    std::string role = "worker";
    int node_id = 0;
    std::uint32_t layer_begin = 0;
    std::uint32_t layer_end = 0;      // inclusive
    std::uint32_t d = 8192;
    std::uint32_t matrices_per_layer = 10;  // 4 attn + 2*(gate,up,down)
    std::uint64_t seed = 42;
    std::string transport = "stdio";
    std::string listen_host = "0.0.0.0";
    std::uint16_t listen_port = 0;
    std::string next_host;
    std::uint16_t next_port = 0;
    int tokens = 8;
    int threads = 0;
    bool synth = true;
    bool final_stage = false;
    std::string model_path;
    std::string node_name = "node";
};

double now_s() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

void log_line(const std::string& s) {
    std::fprintf(stderr, "[%s %.2fs] %s\n", "engine", now_s(), s.c_str());
    std::fflush(stderr);
}

// ---------------------------------------------------------------------------
// Deterministik ternary og'irliklarni parallel generatsiya qilish
// ---------------------------------------------------------------------------
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

// CPU: har (layer, matrix) uchun TernaryMatrix [d x d]
void generate_cpu_weights(const Config& cfg, std::vector<TernaryMatrix>& out) {
    const std::uint32_t layers = cfg.layer_end - cfg.layer_begin + 1;
    const std::uint32_t total = layers * cfg.matrices_per_layer;
    out.resize(total);
    const unsigned hw = cfg.threads > 0 ? static_cast<unsigned>(cfg.threads)
                                        : std::thread::hardware_concurrency();
    const unsigned nthreads = hw == 0 ? 1 : hw;

    std::atomic<std::uint32_t> next_unit{0};
    auto work = [&]() {
        std::vector<std::int8_t> row(cfg.d);
        for (;;) {
            const std::uint32_t unit = next_unit.fetch_add(1);
            if (unit >= total) break;
            const std::uint32_t layer = cfg.layer_begin + unit / cfg.matrices_per_layer;
            const std::uint32_t mi = unit % cfg.matrices_per_layer;
            SplitMix rng(matrix_seed(cfg.seed, layer, mi));
            TernaryMatrix& mat = out[unit];
            mat.resize(cfg.d, cfg.d);
            for (std::uint32_t r = 0; r < cfg.d; ++r) {
                for (std::uint32_t c = 0; c < cfg.d; ++c) row[c] = rng.ternary();
                bt::pack_ternary(row.data(), 1, cfg.d, mat.stride64,
                                 mat.pos.data() + std::size_t(r) * mat.stride64,
                                 mat.neg.data() + std::size_t(r) * mat.stride64);
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(work);
    for (auto& th : pool) th.join();
}

// GPU: barcha matritsalar bitta [rows x d] buferga (uint32 planes)
void generate_gpu_weights(const Config& cfg, TernaryMatrix32& out) {
    const std::uint32_t layers = cfg.layer_end - cfg.layer_begin + 1;
    const std::uint32_t total = layers * cfg.matrices_per_layer;
    out.rows = std::size_t(total) * cfg.d;
    out.cols = cfg.d;
    out.stride32 = bt::words32(cfg.d);
    out.pos.assign(out.rows * out.stride32, 0);
    out.neg.assign(out.rows * out.stride32, 0);

    const unsigned hw = cfg.threads > 0 ? static_cast<unsigned>(cfg.threads)
                                        : std::thread::hardware_concurrency();
    const unsigned nthreads = hw == 0 ? 1 : hw;

    std::atomic<std::uint32_t> next_unit{0};
    auto work = [&]() {
        std::vector<std::int8_t> row(cfg.d);
        for (;;) {
            const std::uint32_t unit = next_unit.fetch_add(1);
            if (unit >= total) break;
            const std::uint32_t layer = cfg.layer_begin + unit / cfg.matrices_per_layer;
            const std::uint32_t mi = unit % cfg.matrices_per_layer;
            SplitMix rng(matrix_seed(cfg.seed, layer, mi));
            for (std::uint32_t r = 0; r < cfg.d; ++r) {
                for (std::uint32_t c = 0; c < cfg.d; ++c) row[c] = rng.ternary();
                const std::size_t row_idx = std::size_t(unit) * cfg.d + r;
                bt::pack_ternary32(row.data(), 1, cfg.d, out.stride32,
                                   out.pos.data() + row_idx * out.stride32,
                                   out.neg.data() + row_idx * out.stride32);
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(work);
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------
// Hidden state konvertatsiyalari
// ---------------------------------------------------------------------------
void quantize_hidden(const std::vector<float>& h, std::vector<std::int8_t>& q) {
    q.resize(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) {
        q[i] = h[i] > 0.05f ? 1 : (h[i] < -0.05f ? -1 : 0);
    }
}

void pack_col64(const std::vector<std::int8_t>& q, TernaryMatrix& x) {
    x.rows = q.size();
    x.cols = 1;
    x.stride64 = bt::words64(q.size());
    x.pos.assign(q.size() * x.stride64, 0);
    x.neg.assign(q.size() * x.stride64, 0);
    bt::pack_ternary_column(q.data(), q.size(), x.stride64, x.pos.data(), x.neg.data());
}

void pack_col32(const std::vector<std::int8_t>& q, TernaryMatrix32& x) {
    x.rows = q.size();
    x.cols = 1;
    x.stride32 = bt::words32(q.size());
    x.pos.assign(q.size() * x.stride32, 0);
    x.neg.assign(q.size() * x.stride32, 0);
    bt::pack_ternary32_column(q.data(), q.size(), x.stride32, x.pos.data(), x.neg.data());
}

float activation(float v, float scale) {
    return std::tanh(v * scale);
}

// ---------------------------------------------------------------------------
// Worker/controller compute: o'z layerlarini bitta matn vektorga qo'llash
// ---------------------------------------------------------------------------
class LayerCompute {
public:
    bool init(const Config& cfg, std::string* err) {
        cfg_ = cfg;
        const std::uint32_t layers = cfg.layer_end - cfg.layer_begin + 1;
        if (layers == 0) {
            *err = "layer oralig'i bo'sh";
            return false;
        }
        log_line("og'irliklar generatsiyasi: " + std::to_string(layers) +
                 " layer x " + std::to_string(cfg.matrices_per_layer) + " matritsa");

        // GPU mavjudmi?
        bt::gpu::VulkanEngine probe;
        std::string gerr;
        use_gpu_ = probe.init(&gerr, "");
        if (use_gpu_) {
            gpu_info_ = probe.info().name;
            probe.shutdown();
            log_line("GPU rejimi: " + gpu_info_);
            generate_gpu_weights(cfg, gpu_weights_);
            if (!engine_.init(&gerr, "")) {
                *err = "Vulkan init: " + gerr;
                return false;
            }
            if (!engine_.upload_weights(gpu_weights_.pos.size(), gpu_weights_.pos.data(),
                                        gpu_weights_.neg.data(), &gerr)) {
                *err = "GPU weight upload: " + gerr;
                return false;
            }
            descs_.clear();
            const std::size_t layer_words = std::size_t(cfg.d) * gpu_weights_.stride32;
            for (std::size_t m = 0; m < std::size_t(layers) * cfg.matrices_per_layer; ++m) {
                descs_.push_back({cfg.d, static_cast<std::uint32_t>(gpu_weights_.stride32),
                                  static_cast<std::uint32_t>(m * layer_words)});
            }
            // X buffer (batch=1) — bir marta yaratiladi, keyin update_input
            std::vector<std::uint32_t> xp(gpu_weights_.stride32, 0);
            std::vector<std::uint32_t> xn(gpu_weights_.stride32, 0);
            if (!engine_.upload_input(1, static_cast<std::uint32_t>(gpu_weights_.stride32),
                                      xp.data(), xn.data(), &gerr)) {
                *err = "GPU input buffer: " + gerr;
                return false;
            }
            if (!engine_.build(descs_, 1, &gerr)) {
                *err = "GPU build: " + gerr;
                return false;
            }
        } else {
            log_line("GPU yo'q (" + gerr + ") — CPU AVX2/AVX-512 rejimi");
            generate_cpu_weights(cfg, cpu_weights_);
        }
        log_line("WEIGHTS READY: " + std::to_string(total_params() / 1e9) +
                 "B param, " + std::to_string(total_bytes() / 1e9) + " GB packed");
        return true;
    }

    std::uint64_t total_params() const {
        const std::uint64_t layers = cfg_.layer_end - cfg_.layer_begin + 1;
        return layers * cfg_.matrices_per_layer * std::uint64_t(cfg_.d) * cfg_.d;
    }
    std::uint64_t total_bytes() const {
        return (total_params() + 3) / 4;  // 2 bit/param
    }
    bool gpu() const { return use_gpu_; }
    const std::string& gpu_name() const { return gpu_info_; }

    // bitta matmuls: out = W * x  (d x d) x (d x 1) -> int32[d]
    void matmul(const TernaryMatrix& w, const TernaryMatrix& x, std::int32_t* out) {
        bt::cpu::ternary_matmul(w, x, out, {}, Kernel::Auto, cfg_.threads);
    }

    // Butun layer stack'ni bitta vektorga qo'llash (token decode simulyatsiyasi)
    void forward(std::vector<float>& hidden, std::int32_t* tmp) {
        const std::uint32_t d = cfg_.d;
        const float act_scale = 1.0f / std::sqrt(static_cast<float>(d));
        std::vector<std::int8_t> q;
        std::vector<float> h = hidden;

        if (use_gpu_) {
            quantize_hidden(h, q);
            TernaryMatrix32 x;
            pack_col32(q, x);
            std::string err;
            engine_.update_input(x.pos.data(), x.neg.data());
            if (!engine_.execute(&err)) {
                log_line("GPU execute xato: " + err);
                return;
            }
            std::vector<std::int32_t> out(d);
            engine_.read_output(out.data(), d, &err);
            for (std::uint32_t i = 0; i < d; ++i) {
                h[i] = activation(static_cast<float>(out[i]), act_scale);
            }
            hidden = h;
            return;
        }

        const std::uint32_t layers = cfg_.layer_end - cfg_.layer_begin + 1;
        TernaryMatrix x;
        std::vector<std::int32_t> out(d);
        std::size_t unit = 0;
        for (std::uint32_t l = 0; l < layers; ++l) {
            // 4 attention proyeksiya (ketma-ket)
            for (int m = 0; m < 4; ++m, ++unit) {
                quantize_hidden(h, q);
                pack_col64(q, x);
                matmul(cpu_weights_[unit], x, out.data());
                for (std::uint32_t i = 0; i < d; ++i) {
                    h[i] = activation(static_cast<float>(out[i]), act_scale);
                }
            }
            // 2 ta FFN blok (har biri 3 matritsa)
            for (int blk = 0; blk < 2; ++blk) {
                std::vector<float> residual = h;
                for (int m = 0; m < 3; ++m, ++unit) {
                    quantize_hidden(h, q);
                    pack_col64(q, x);
                    matmul(cpu_weights_[unit], x, out.data());
                    for (std::uint32_t i = 0; i < d; ++i) {
                        h[i] = activation(static_cast<float>(out[i]), act_scale);
                    }
                }
                for (std::uint32_t i = 0; i < d; ++i) h[i] += residual[i];
            }
        }
        hidden = h;
        (void)tmp;
    }

private:
    Config cfg_;
    bool use_gpu_ = false;
    std::string gpu_info_;
    std::vector<TernaryMatrix> cpu_weights_;
    TernaryMatrix32 gpu_weights_;
    std::vector<bt::gpu::LayerDesc> descs_;
    bt::gpu::VulkanEngine engine_;
};

// ---------------------------------------------------------------------------
// Embedding: token id + stream -> d o'lchamli vektor (sintetik)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Transport adapter: TCP yoki stdio
// ---------------------------------------------------------------------------
class Transport {
public:
    bool init_server(const Config& cfg, std::string* err) {
        if (cfg.transport == "stdio") {
            use_stdio_ = true;
            return true;
        }
        if (!server_.listen_on(cfg.listen_host, cfg.listen_port, err)) return false;
        log_line("tinglash: port " + std::to_string(cfg.listen_port));
        if (!server_.accept_into(up_, 120000, err)) return false;
        log_line("upstream ulandi");
        return true;
    }

    bool init_client(const Config& cfg, std::string* err) {
        if (cfg.transport == "stdio") {
            use_stdio_ = true;
            return true;
        }
        if (!down_.connect_to(cfg.next_host, cfg.next_port, err)) return false;
        if (!down_.wait_connected(30000, err)) return false;
        log_line("downstream ulandi: " + cfg.next_host + ":" + std::to_string(cfg.next_port));
        return true;
    }

    bool recv(bt::net::Frame& f) {
        if (use_stdio_) return bt::net::read_stdio_frame(f);
        return bt::net::recv_frame(up_, f, 600000);
    }
    bool send(bt::net::Frame& f) {
        if (use_stdio_) return bt::net::write_stdio_frame(f);
        return bt::net::send_frame(down_, f, 10000);
    }

private:
    bool use_stdio_ = false;
    bt::net::Socket server_;
    bt::net::Socket up_;
    bt::net::Socket down_;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
void usage() {
    std::printf(
        "distributed_70b_engine — Phase 2 pipeline parallel inference\n"
        "  --role controller|worker   (default worker)\n"
        "  --node-id N                node raqami (1-4)\n"
        "  --layers A-B               layer oralig'i (masalan 12-43)\n"
        "  --transport tcp|stdio      (default stdio — relay bridge bilan)\n"
        "  --listen-port N            tcp server porti (worker)\n"
        "  --next-node-ip HOST:PORT   keyingi node (tcp)\n"
        "  --d N                      d_model (default 8192)\n"
        "  --seed N                   sintetik og'irlik seed'i\n"
        "  --tokens N                 generatsiya tokenlari (controller)\n"
        "  --threads N                CPU threadlari\n"
        "  --final                    oxirgi stage (token sampiling)\n"
        "  --model-path PATH          (ixtiyoriy; hozir sintetik)\n"
        "  --node-name NAME           log nomi\n");
}

bool parse_args(int argc, char** argv, Config& cfg) {
    // --flag=value ko'rinishini ham qo'llab-quvvatlash
    std::vector<std::string> av;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const auto eq = a.find('=');
        if (a.rfind("--", 0) == 0 && eq != std::string::npos) {
            av.push_back(a.substr(0, eq));
            av.push_back(a.substr(eq + 1));
        } else {
            av.push_back(a);
        }
    }
    for (std::size_t i = 0; i < av.size(); ++i) {
        const std::string a = av[i];
        auto val = [&]() -> std::string { return i + 1 < av.size() ? av[++i] : std::string{}; };
        if (a == "--role") cfg.role = val();
        else if (a == "--node-id") cfg.node_id = std::atoi(val().c_str());
        else if (a == "--layers") {
            const std::string v = val();
            const auto dash = v.find('-');
            if (dash == std::string::npos) return false;
            cfg.layer_begin = static_cast<std::uint32_t>(std::atoi(v.substr(0, dash).c_str()));
            cfg.layer_end = static_cast<std::uint32_t>(std::atoi(v.substr(dash + 1).c_str()));
        } else if (a == "--transport") cfg.transport = val();
        else if (a == "--listen-port") cfg.listen_port = static_cast<std::uint16_t>(std::atoi(val().c_str()));
        else if (a == "--next-node-ip") {
            const std::string v = val();
            const auto colon = v.find(':');
            if (colon == std::string::npos) return false;
            cfg.next_host = v.substr(0, colon);
            cfg.next_port = static_cast<std::uint16_t>(std::atoi(v.c_str() + colon + 1));
        } else if (a == "--d") cfg.d = static_cast<std::uint32_t>(std::atoi(val().c_str()));
        else if (a == "--seed") cfg.seed = std::strtoull(val().c_str(), nullptr, 10);
        else if (a == "--tokens") cfg.tokens = std::atoi(val().c_str());
        else if (a == "--threads") cfg.threads = std::atoi(val().c_str());
        else if (a == "--final") cfg.final_stage = true;
        else if (a == "--model-path") cfg.model_path = val();
        else if (a == "--node-name") cfg.node_name = val();
        else if (a == "--help" || a == "-h") { usage(); return false; }
        else { std::fprintf(stderr, "noma'lum argument: %s\n", a.c_str()); return false; }
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        usage();
        return 2;
    }
    if (cfg.layer_end < cfg.layer_begin) {
        std::fprintf(stderr, "XATO: --layers A-B noto'g'ri\n");
        return 2;
    }
    if (cfg.threads <= 0) cfg.threads = bt::cpu::hardware_threads();

    LayerCompute compute;
    std::string err;
    if (!compute.init(cfg, &err)) {
        std::fprintf(stderr, "XATO: %s\n", err.c_str());
        return 1;
    }

    Transport tr;
    const bool is_controller = (cfg.role == "controller");

    if (is_controller) {
        // Controller: keyingi node'ga ulanadi (stdio'da bridge orqali)
        if (!tr.init_client(cfg, &err)) {
            std::fprintf(stderr, "XATO: %s\n", err.c_str());
            return 1;
        }
        log_line("CONTROLLER tayyor: layers " + std::to_string(cfg.layer_begin) + "-" +
                 std::to_string(cfg.layer_end) + ", tokens=" + std::to_string(cfg.tokens));

        // Interaktiv rejim: PROMPT kadrlari kelganda generatsiya qiladi.
        auto generate = [&](std::uint64_t seed, int ntok) {
            const double t0 = now_s();
            std::uint32_t token = static_cast<std::uint32_t>(seed % 32000u);
            std::vector<double> latencies;
            for (int t = 0; t < ntok; ++t) {
                auto h = embed_token(seed, token, static_cast<std::uint32_t>(t), cfg.d);
                const double ts = now_s();
                compute.forward(h, nullptr);
                bt::net::Frame f;
                f.seq = static_cast<std::uint32_t>(t);
                f.stream = 0;
                f.payload = h;
                if (!tr.send(f)) {
                    std::fprintf(stderr, "XATO: send (token %d)\n", t);
                    return false;
                }
                bt::net::Frame back;
                for (;;) {
                    if (!tr.recv(back)) {
                        std::fprintf(stderr, "XATO: token qaytmadi (%d)\n", t);
                        return false;
                    }
                    if (back.is_prompt()) {
                        continue;  // keyingi promptga navbat (signal yo'q — oddiy tashlab)
                    }
                    if (back.is_token()) break;
                }
                token = static_cast<std::uint32_t>(back.payload[1]);
                const double dt = now_s() - ts;
                latencies.push_back(dt);
                log_line("token " + std::to_string(t + 1) + "/" + std::to_string(ntok) +
                         " id=" + std::to_string(token) + " latency=" +
                         std::to_string(static_cast<int>(dt * 1000)) + "ms");
                bt::net::Frame ev;  // chat uchun stream eventi
                ev.flags = bt::net::BT_FLAG_TOCHAT | bt::net::BT_FLAG_TOKEN;
                ev.seq = f.seq;
                ev.payload = {static_cast<float>(t), static_cast<float>(token)};
                tr.send(ev);
            }
            const double total = now_s() - t0;
            double avg = 0;
            for (double l : latencies) avg += l;
            avg = latencies.empty() ? 0 : avg / latencies.size();
            std::fprintf(stderr, "PIPELINE-RESULT tokens=%d total=%.3fs tok_per_s=%.4f "
                        "avg_latency_ms=%.1f\n",
                        ntok, total, ntok / total, avg * 1000.0);
            bt::net::Frame done;
            done.flags = bt::net::BT_FLAG_TOCHAT;
            done.payload = {static_cast<float>(ntok),
                            static_cast<float>(total * 1000.0)};
            tr.send(done);
            std::fflush(stderr);
            return true;
        };

        log_line("CONTROLLER interaktiv tayyor — prompt kutilmoqda");
        for (;;) {
            bt::net::Frame f;
            if (!tr.recv(f)) break;
            if (!f.is_prompt() || f.payload.size() < 3) continue;
            std::uint32_t lo = 0, hi = 0;
            std::memcpy(&lo, &f.payload[0], 4);
            std::memcpy(&hi, &f.payload[1], 4);
            const std::uint64_t seed = (static_cast<std::uint64_t>(hi) << 32) | lo;
            int mt = static_cast<int>(f.payload[2]);
            if (mt <= 0 || mt > 64) mt = 8;
            log_line("PROMPT keldi: seed=" + std::to_string(seed) +
                     " tokens=" + std::to_string(mt));
            generate(seed, mt);
        }
        return 0;
    }

    // Worker: upstream'dan qabul, hisoblash, downstream'ga uzatish
    if (cfg.transport == "stdio") {
        if (!tr.init_server(cfg, &err)) {  // stdio'da server = bridge
            std::fprintf(stderr, "XATO: %s\n", err.c_str());
            return 1;
        }
    } else {
        if (!tr.init_server(cfg, &err)) {
            std::fprintf(stderr, "XATO: %s\n", err.c_str());
            return 1;
        }
        if (!tr.init_client(cfg, &err)) {
            std::fprintf(stderr, "XATO: %s\n", err.c_str());
            return 1;
        }
    }

    log_line("WORKER tayyor: layers " + std::to_string(cfg.layer_begin) + "-" +
             std::to_string(cfg.layer_end) + " gpu=" +
             (compute.gpu() ? compute.gpu_name() : "cpu"));

    std::uint64_t processed = 0;
    const double t_start = now_s();
    for (;;) {
        bt::net::Frame f;
        if (!tr.recv(f)) break;
        if (f.is_token()) {  // token kadri — o'zgarishsiz uzatamiz
            if (!tr.send(f)) break;
            continue;
        }
        const double c0 = now_s();
        std::vector<float> hidden = f.payload;
        compute.forward(hidden, nullptr);
        if (cfg.final_stage) {
            // Sintetik head: hidden -> token id (deterministik sampling)
            std::uint64_t acc = cfg.seed;
            for (std::uint32_t i = 0; i < cfg.d; i += 64) {
                acc ^= (std::uint64_t)std::llround(hidden[i] * 100.0) * (i + 1);
                acc = (acc << 13) ^ (acc >> 7);
            }
            const std::uint32_t token_id = static_cast<std::uint32_t>(acc % 32000u);
            bt::net::Frame tok;
            tok.seq = f.seq;
            tok.stream = f.stream;
            tok.flags = bt::net::BT_FLAG_TOKEN;
            tok.payload = {static_cast<float>(f.stream),
                           static_cast<float>(token_id)};
            if (!tr.send(tok)) break;
        } else {
            f.payload = hidden;
            if (!tr.send(f)) break;
        }
        ++processed;
        const double dt = now_s() - c0;
        log_line("frame " + std::to_string(f.seq) + " hisoblandi: " +
                 std::to_string(static_cast<int>(dt * 1000)) + "ms (jami " +
                 std::to_string(processed) + ", " +
                 std::to_string(processed / (now_s() - t_start + 1e-9)) + " fps)");
    }
    std::fprintf(stderr, "WORKER-DONE processed=%llu avg_stage_ms=%.1f gpu=%s\n",
                static_cast<unsigned long long>(processed),
                processed ? (now_s() - t_start) * 1000.0 / processed : 0.0,
                compute.gpu() ? compute.gpu_name().c_str() : "cpu");
    return 0;
}
