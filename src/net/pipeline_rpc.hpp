// SPDX-License-Identifier: MIT
// pipeline_rpc.hpp — ultra-yengil, non-blocking pipeline transport (header-only).
//
// Vazifa: Node 1 -> Node 2 -> Node 3 -> Node 4 zanjiri bo'ylab faqat
// [dim] float32 hidden-state vektorlarini uzatish. Har bir node vektorni
// qabul qiladi, ixtiyoriy lokal transform bajaradi va keyingisiga uzatadi.
//
// Protokol (little-endian, 12 bayt sarlavha + payload):
//   magic   : 'B','T'      (2 bayt)
//   version : uint8        (=1)
//   flags   : uint8        (0)
//   seq     : uint32
//   dim     : uint32
//   payload : dim * float32
//
// UDP emas, TCP; non-blocking connect/read/write + poll (timeout bilan).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using bt_sock_t = SOCKET;
#  define BT_INVALID_SOCK INVALID_SOCKET
#  define BT_SOCK_ERRNO WSAGetLastError()
#  define BT_EWOULDBLOCK WSAEWOULDBLOCK
#else
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
using bt_sock_t = int;
#  define BT_INVALID_SOCK (-1)
#  define BT_SOCK_ERRNO errno
#  define BT_EWOULDBLOCK EWOULDBLOCK
#endif

namespace bt::net {

// ---------------------------------------------------------------------------
// Global init (Windows uchun WSAStartup; POSIX'da bo'sh)
// ---------------------------------------------------------------------------
inline void net_global_init() {
#if defined(_WIN32)
    static std::atomic<bool> done{false};
    if (!done.exchange(true)) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
#endif
}

// ---------------------------------------------------------------------------
// Frame — uzatiladigan hidden-state vektori (yoki token event)
// ---------------------------------------------------------------------------
constexpr std::uint8_t BT_FLAG_TOKEN = 1;   // payload[0]=stream, payload[1]=token_id
constexpr std::uint8_t BT_FLAG_PROMPT = 2;  // payload[0..1]=seed bits, payload[2]=max_tokens
constexpr std::uint8_t BT_FLAG_TOCHAT = 4;  // bridge uni 'chat' mailbox'iga yo'naltiradi

struct Frame {
    std::uint32_t seq = 0;      // global kadr raqami
    std::uint32_t stream = 0;   // multi-sequence pipeline oqimi
    std::uint8_t flags = 0;
    std::vector<float> payload;  // dim ta float32

    std::uint32_t dim() const { return static_cast<std::uint32_t>(payload.size()); }
    bool is_token() const { return (flags & BT_FLAG_TOKEN) != 0; }
    bool is_prompt() const { return (flags & BT_FLAG_PROMPT) != 0; }
    bool to_chat() const { return (flags & BT_FLAG_TOCHAT) != 0; }
};

inline std::vector<std::uint8_t> encode_frame(const Frame& f) {
    std::vector<std::uint8_t> out(16 + f.payload.size() * sizeof(float));
    out[0] = 'B';
    out[1] = 'T';
    out[2] = 1;          // version
    out[3] = f.flags;
    const std::uint32_t seq = f.seq;
    const std::uint32_t stream = f.stream;
    const std::uint32_t dim = f.dim();
    std::memcpy(out.data() + 4, &seq, 4);
    std::memcpy(out.data() + 8, &stream, 4);
    std::memcpy(out.data() + 12, &dim, 4);
    if (!f.payload.empty()) {
        std::memcpy(out.data() + 16, f.payload.data(),
                    f.payload.size() * sizeof(float));
    }
    return out;
}

inline bool decode_header(const std::uint8_t* h, std::uint32_t& seq,
                          std::uint32_t& stream, std::uint32_t& dim) {
    if (h[0] != 'B' || h[1] != 'T' || h[2] != 1) return false;
    std::memcpy(&seq, h + 4, 4);
    std::memcpy(&stream, h + 8, 4);
    std::memcpy(&dim, h + 12, 4);
    return dim <= (1u << 24);  // sog'lom chegara (16M float = 64MB)
}

// ---------------------------------------------------------------------------
// Socket — non-blocking TCP yordamchisi
// ---------------------------------------------------------------------------
class Socket {
public:
    Socket() = default;
    explicit Socket(bt_sock_t fd) : fd_(fd) {}
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& o) noexcept : fd_(o.fd_) { o.fd_ = BT_INVALID_SOCK; }
    Socket& operator=(Socket&& o) noexcept {
        if (this != &o) {
            close();
            fd_ = o.fd_;
            o.fd_ = BT_INVALID_SOCK;
        }
        return *this;
    }
    ~Socket() { close(); }

    bool valid() const { return fd_ != BT_INVALID_SOCK; }
    bt_sock_t fd() const { return fd_; }

    static bool set_nonblocking(bt_sock_t fd) {
#if defined(_WIN32)
        u_long mode = 1;
        return ioctlsocket(fd, FIONBIO, &mode) == 0;
#else
        const int flags = fcntl(fd, F_GETFL, 0);
        return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
    }

    static void set_nodelay(bt_sock_t fd) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
                   reinterpret_cast<const char*>(&one), sizeof(one));
    }

    // Server: listen (non-blocking accept)
    bool listen_on(const std::string& host, std::uint16_t port, std::string* error) {
        net_global_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (!valid()) {
            if (error) *error = "socket() xato";
            return false;
        }
        int one = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&one), sizeof(one));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = host.empty() || host == "0.0.0.0"
                                   ? INADDR_ANY
                                   : inet_addr(host.c_str());
        if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            if (error) *error = "bind() xato (port band?)";
            close();
            return false;
        }
        if (::listen(fd_, 1) != 0) {
            if (error) *error = "listen() xato";
            close();
            return false;
        }
        set_nonblocking(fd_);
        return true;
    }

    // Client: non-blocking connect (connect davom etishi mumkin — wait_connected)
    bool connect_to(const std::string& host, std::uint16_t port, std::string* error) {
        net_global_init();
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (!valid()) {
            if (error) *error = "socket() xato";
            return false;
        }
        set_nonblocking(fd_);
        set_nodelay(fd_);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr(host.c_str());
        const int rc = ::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc == 0) return true;
#if defined(_WIN32)
        const bool pending = (BT_SOCK_ERRNO == WSAEWOULDBLOCK ||
                              BT_SOCK_ERRNO == WSAEINPROGRESS);
#else
        const bool pending = (errno == EINPROGRESS || errno == EWOULDBLOCK);
#endif
        if (!pending) {
            if (error) *error = "connect() xato";
            close();
            return false;
        }
        return true;  // davom etadi
    }

    // Non-blocking connect tugashini kutish
    bool wait_connected(int timeout_ms, std::string* error) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        for (;;) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(fd_, &wfds);
            timeval tv{0, 50 * 1000};
            const int rc = select(static_cast<int>(fd_ + 1), nullptr, &wfds, nullptr, &tv);
            if (rc > 0) {
                int soerr = 0;
#if defined(_WIN32)
                int len = sizeof(soerr);
#else
                socklen_t len = sizeof(soerr);
#endif
                getsockopt(fd_, SOL_SOCKET, SO_ERROR,
                           reinterpret_cast<char*>(&soerr), &len);
                if (soerr == 0) return true;
                if (error) *error = "connect yakunlandi xato bilan: " +
                                    std::to_string(soerr);
                return false;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                if (error) *error = "connect timeout";
                return false;
            }
        }
    }

    // Accept (timeout bilan); muvaffaqiyatda out'ga yangi socket
    bool accept_into(Socket& out, int timeout_ms, std::string* error) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        for (;;) {
            sockaddr_in peer{};
#if defined(_WIN32)
            int plen = sizeof(peer);
#else
            socklen_t plen = sizeof(peer);
#endif
            bt_sock_t c = ::accept(fd_, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (c != BT_INVALID_SOCK) {
                set_nodelay(c);
                out = Socket(c);
                return true;
            }
            if (std::chrono::steady_clock::now() > deadline) {
                if (error) *error = "accept timeout";
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    bool send_all(const void* data, std::size_t n, int timeout_ms) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        std::size_t sent = 0;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (sent < n) {
#if defined(_WIN32)
            const int rc = ::send(fd_, reinterpret_cast<const char*>(p + sent),
                                  static_cast<int>(n - sent), 0);
#else
            const ssize_t rc = ::send(fd_, p + sent, n - sent, MSG_NOSIGNAL);
#endif
            if (rc > 0) {
                sent += static_cast<std::size_t>(rc);
                continue;
            }
            const int e = BT_SOCK_ERRNO;
            if (rc < 0 && (e == BT_EWOULDBLOCK
#if defined(_WIN32)
                           || e == WSAEINPROGRESS
#else
                           || e == EAGAIN
#endif
                           )) {
                if (wait_io(true, deadline)) continue;
                return false;
            }
            return false;
        }
        return true;
    }

    bool recv_all(void* data, std::size_t n, int timeout_ms) {
        auto* p = static_cast<std::uint8_t*>(data);
        std::size_t got = 0;
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        while (got < n) {
#if defined(_WIN32)
            const int rc = ::recv(fd_, reinterpret_cast<char*>(p + got),
                                  static_cast<int>(n - got), 0);
#else
            const ssize_t rc = ::recv(fd_, p + got, n - got, 0);
#endif
            if (rc > 0) {
                got += static_cast<std::size_t>(rc);
                continue;
            }
            if (rc == 0) return false;  // peer yopdi
            const int e = BT_SOCK_ERRNO;
            if (e == BT_EWOULDBLOCK
#if !defined(_WIN32)
                || e == EAGAIN
#endif
                ) {
                if (wait_io(false, deadline)) continue;
                return false;
            }
            return false;
        }
        return true;
    }

    void close() {
        if (fd_ != BT_INVALID_SOCK) {
#if defined(_WIN32)
            closesocket(fd_);
#else
            ::close(fd_);
#endif
            fd_ = BT_INVALID_SOCK;
        }
    }

private:
    bool wait_io(bool write, std::chrono::steady_clock::time_point deadline) {
        for (;;) {
            if (std::chrono::steady_clock::now() > deadline) return false;
            timeval tv{0, 50 * 1000};
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd_, &fds);
            const int rc = select(static_cast<int>(fd_ + 1),
                                  write ? nullptr : &fds,
                                  write ? &fds : nullptr, nullptr, &tv);
            if (rc > 0) return true;
            if (rc < 0) return false;
        }
    }

    bt_sock_t fd_ = BT_INVALID_SOCK;
};

// ---------------------------------------------------------------------------
// Frame I/O (encode/decode + socket)
// ---------------------------------------------------------------------------
inline bool send_frame(Socket& s, const Frame& f, int timeout_ms) {
    const auto buf = encode_frame(f);
    return s.send_all(buf.data(), buf.size(), timeout_ms);
}

inline bool recv_frame(Socket& s, Frame& f, int timeout_ms) {
    std::uint8_t header[16];
    if (!s.recv_all(header, sizeof(header), timeout_ms)) return false;
    std::uint32_t seq = 0, stream = 0, dim = 0;
    if (!decode_header(header, seq, stream, dim)) return false;
    f.seq = seq;
    f.stream = stream;
    f.flags = header[3];
    f.payload.resize(dim);
    if (dim > 0 && !s.recv_all(f.payload.data(), dim * sizeof(float), timeout_ms)) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stdio transport — bridge (Python/HTTP relay) bilan ishlash uchun:
// [u32 uzunlik][encode_frame bytes] oqimi.
// ---------------------------------------------------------------------------
inline bool write_stdio_frame(Frame& f) {
    const auto buf = encode_frame(f);
    const std::uint32_t len = static_cast<std::uint32_t>(buf.size());
#if defined(_WIN32)
    std::fwrite(&len, 1, 4, stdout);
    std::fwrite(buf.data(), 1, buf.size(), stdout);
    std::fflush(stdout);
#else
    if (std::fwrite(&len, 1, 4, stdout) != 4) return false;
    if (std::fwrite(buf.data(), 1, buf.size(), stdout) != buf.size()) return false;
    std::fflush(stdout);
#endif
    return true;
}

inline bool read_stdio_frame(Frame& f) {
    std::uint32_t len = 0;
    if (std::fread(&len, 1, 4, stdin) != 4) return false;
    if (len < 16 || len > (1u << 26)) return false;
    std::vector<std::uint8_t> buf(len);
    if (std::fread(buf.data(), 1, len, stdin) != len) return false;
    std::uint32_t seq = 0, stream = 0, dim = 0;
    if (!decode_header(buf.data(), seq, stream, dim)) return false;
    if (16 + dim * 4u != len) return false;
    f.seq = seq;
    f.stream = stream;
    f.flags = buf[3];
    f.payload.resize(dim);
    if (dim > 0) std::memcpy(f.payload.data(), buf.data() + 16, dim * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// PipelineNode — zanjirning bitta bosqichi
//
// role: "first"  — vektor generatsiya qiladi va keyingisiga yuboradi
//       "middle" — qabul qiladi, transform qiladi, uzatadi
//       "last"   — qabul qiladi, transform qiladi, statistikani yig'adi
//
// `listen` bo'sh bo'lsa server ochilmaydi (first uchun); `next` bo'sh bo'lsa
// uzatilmaydi (last uchun).
// ---------------------------------------------------------------------------
struct PipelineStats {
    std::uint64_t frames = 0;
    double seconds = 0.0;
    double frames_per_sec() const { return seconds > 0 ? frames / seconds : 0.0; }
};

inline bool pipeline_run(const std::string& role,
                         std::uint64_t frames,
                         std::uint32_t dim,
                         const std::string& listen_host, std::uint16_t listen_port,
                         const std::string& next_host, std::uint16_t next_port,
                         const std::function<void(Frame&)>& transform,
                         PipelineStats* stats,
                         std::string* error) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    Socket up, down;
    if (role != "first") {
        Socket server;
        if (!server.listen_on(listen_host, listen_port, error)) return false;
        if (!server.accept_into(up, 30000, error)) return false;
    }
    if (role != "last") {
        if (!down.connect_to(next_host, next_port, error)) return false;
        if (!down.wait_connected(10000, error)) return false;
    }

    std::vector<float> seed(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        seed[i] = 0.01f * static_cast<float>((i % 97) + 1);
    }

    for (std::uint64_t i = 0; i < frames; ++i) {
        if (role == "first") {
            Frame f;
            f.seq = static_cast<std::uint32_t>(i);
            f.payload = seed;
            f.payload[static_cast<std::size_t>(i) % dim] += 1.0f;
            if (!send_frame(down, f, 10000)) {
                if (error) *error = "send xato (frame " + std::to_string(i) + ")";
                return false;
            }
        } else {
            Frame f;
            if (!recv_frame(up, f, 60000)) {
                if (error) *error = "recv xato (frame " + std::to_string(i) + ")";
                return false;
            }
            if (transform) transform(f);
            if (role == "middle") {
                if (!send_frame(down, f, 10000)) {
                    if (error) *error = "forward xato (frame " + std::to_string(i) + ")";
                    return false;
                }
            }
        }
    }

    if (stats) {
        stats->frames = frames;
        stats->seconds =
            std::chrono::duration<double>(clock::now() - t0).count();
    }
    return true;
}

}  // namespace bt::net
