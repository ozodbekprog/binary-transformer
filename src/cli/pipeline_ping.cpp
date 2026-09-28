// SPDX-License-Identifier: MIT
// pipeline_ping.cpp — Pipeline RPC demosi: Node1 -> Node2 -> Node3 -> Node4.
//
// Ishlatish (4 ta terminal yoki 4 ta PC):
//   # Node 4 (oxirgi):
//   pipeline_ping --role last   --listen 9004 --dim 512 --frames 2000
//   # Node 3:
//   pipeline_ping --role middle --listen 9003 --next 127.0.0.1:9004 --dim 512 --frames 2000
//   # Node 2:
//   pipeline_ping --role middle --listen 9002 --next 127.0.0.1:9003 --dim 512 --frames 2000
//   # Node 1 (generator):
//   pipeline_ping --role first --next 127.0.0.1:9002 --dim 512 --frames 2000
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "net/pipeline_rpc.hpp"

namespace {

struct Args {
    std::string role = "first";
    std::string listen_host = "0.0.0.0";
    std::uint16_t listen_port = 0;
    std::string next_host;
    std::uint16_t next_port = 0;
    std::uint32_t dim = 512;
    std::uint64_t frames = 1000;
};

bool parse_hostport(const std::string& s, std::string& host, std::uint16_t& port) {
    const auto pos = s.find(':');
    if (pos == std::string::npos) return false;
    host = s.substr(0, pos);
    port = static_cast<std::uint16_t>(std::atoi(s.c_str() + pos + 1));
    return port != 0;
}

void usage() {
    std::printf(
        "pipeline_ping — ferma pipeline transport demosi\n"
        "  --role first|middle|last   zanjirdagi rol (default first)\n"
        "  --listen PORT              tinglash porti (middle/last uchun)\n"
        "  --next HOST:PORT           keyingi node (first/middle uchun)\n"
        "  --dim N                    hidden-state o'lchami (default 512)\n"
        "  --frames N                 kadrlar soni (default 1000)\n"
        "  --listen-host HOST         bind manzili (default 0.0.0.0)\n");
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : std::string{};
        };
        if (a == "--role") {
            args.role = val();
        } else if (a == "--listen") {
            args.listen_port = static_cast<std::uint16_t>(std::atoi(val().c_str()));
        } else if (a == "--listen-host") {
            args.listen_host = val();
        } else if (a == "--next") {
            if (!parse_hostport(val(), args.next_host, args.next_port)) {
                std::fprintf(stderr, "XATO: --next HOST:PORT ko'rinishida bo'lsin\n");
                return 2;
            }
        } else if (a == "--dim") {
            args.dim = static_cast<std::uint32_t>(std::atoi(val().c_str()));
        } else if (a == "--frames") {
            args.frames = static_cast<std::uint64_t>(std::atoll(val().c_str()));
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "Noma'lum argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (args.role != "first" && args.listen_port == 0) {
        std::fprintf(stderr, "XATO: %s uchun --listen PORT kerak\n", args.role.c_str());
        return 2;
    }
    if (args.role != "last" && args.next_host.empty()) {
        std::fprintf(stderr, "XATO: %s uchun --next HOST:PORT kerak\n", args.role.c_str());
        return 2;
    }

    // Lokal transform (middle/last): oddiy layer simulyatsiyasi (cheap)
    const auto transform = [](bt::net::Frame& f) {
        float acc = 0.0f;
        for (float& v : f.payload) {
            acc += v;
            v = v * 0.5f + 0.01f;
        }
        f.flags = static_cast<std::uint8_t>(f.flags ^ (static_cast<unsigned>(acc) & 1u));
    };

    std::printf("pipeline: role=%s dim=%u frames=%llu", args.role.c_str(), args.dim,
                static_cast<unsigned long long>(args.frames));
    if (args.listen_port) std::printf(" listen=%u", args.listen_port);
    if (args.next_port) std::printf(" next=%s:%u", args.next_host.c_str(), args.next_port);
    std::printf("\n");

    bt::net::PipelineStats stats;
    std::string error;
    const bool ok = bt::net::pipeline_run(
        args.role, args.frames, args.dim, args.listen_host, args.listen_port,
        args.next_host, args.next_port, transform, &stats, &error);
    if (!ok) {
        std::fprintf(stderr, "XATO: %s\n", error.c_str());
        return 1;
    }
    const double mb = static_cast<double>(stats.frames) * args.dim * 4.0 / 1e6;
    std::printf("TAYYOR: %llu kadr | %.3f s | %.0f kadr/s | %.1f MB | %.1f MB/s\n",
                static_cast<unsigned long long>(stats.frames), stats.seconds,
                stats.frames_per_sec(), mb,
                stats.seconds > 0 ? mb / stats.seconds : 0.0);
    return 0;
}
