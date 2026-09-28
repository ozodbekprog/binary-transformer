// SPDX-License-Identifier: MIT
// bitpack.hpp — Ternar/Binar matritsalarni bit-packing, mmap yuklash.
//
// Ternar qiymatlar {-1, 0, +1} ikki bit-planes ko'rinishida saqlanadi:
//   +1 -> pos bit = 1
//   -1 -> neg bit = 1
//    0 -> ikkalasi ham 0
// Binar {0, 1} — xuddi shu sxemaning xususiy holi (neg plane bo'sh).
//
// Bitlar little-endian tartibda: k indeksli element (k/64) so'zning (k%64) biti.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

#if defined(_WIN32)
#  define BT_WINDOWS 1
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace bt {

constexpr std::size_t ceil_div(std::size_t a, std::size_t b) noexcept {
    return (a + b - 1) / b;
}
constexpr std::size_t words64(std::size_t bits) noexcept {
    return ceil_div(bits, 64u);
}
constexpr std::size_t words32(std::size_t bits) noexcept {
    return ceil_div(bits, 32u);
}

// ---------------------------------------------------------------------------
// Bit o'qish/yozish (scalar; hot loop'da ishlatilmaydi — faqat packing uchun)
// ---------------------------------------------------------------------------

inline void bit_set(std::uint64_t* words, std::size_t k) noexcept {
    words[k >> 6] |= (std::uint64_t{1} << (k & 63u));
}
inline bool bit_get(const std::uint64_t* words, std::size_t k) noexcept {
    return (words[k >> 6] >> (k & 63u)) & 1u;
}
inline void bit32_set(std::uint32_t* words, std::size_t k) noexcept {
    words[k >> 5] |= (std::uint32_t{1} << (k & 31u));
}
inline bool bit32_get(const std::uint32_t* words, std::size_t k) noexcept {
    return (words[k >> 5] >> (k & 31u)) & 1u;
}

// ---------------------------------------------------------------------------
// Packing: int8 {-1,0,1} qatorlarini (rows x cols) bit-planes ga joylash.
// `stride_words` = words64(cols) — qatorlararo qadam.
// ---------------------------------------------------------------------------

inline void pack_ternary(const std::int8_t* values,
                         std::size_t rows, std::size_t cols,
                         std::size_t stride_words,
                         std::uint64_t* pos, std::uint64_t* neg) noexcept {
    std::memset(pos, 0, rows * stride_words * sizeof(std::uint64_t));
    std::memset(neg, 0, rows * stride_words * sizeof(std::uint64_t));
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int8_t* row = values + r * cols;
        std::uint64_t* p = pos + r * stride_words;
        std::uint64_t* n = neg + r * stride_words;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::int8_t v = row[c];
            if (v > 0) {
                p[c >> 6] |= (std::uint64_t{1} << (c & 63u));
            } else if (v < 0) {
                n[c >> 6] |= (std::uint64_t{1} << (c & 63u));
            }
        }
    }
}

inline void pack_ternary32(const std::int8_t* values,
                           std::size_t rows, std::size_t cols,
                           std::size_t stride_words,
                           std::uint32_t* pos, std::uint32_t* neg) noexcept {
    std::memset(pos, 0, rows * stride_words * sizeof(std::uint32_t));
    std::memset(neg, 0, rows * stride_words * sizeof(std::uint32_t));
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int8_t* row = values + r * cols;
        std::uint32_t* p = pos + r * stride_words;
        std::uint32_t* n = neg + r * stride_words;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::int8_t v = row[c];
            if (v > 0) {
                p[c >> 5] |= (std::uint32_t{1} << (c & 31u));
            } else if (v < 0) {
                n[c >> 5] |= (std::uint32_t{1} << (c & 31u));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ustun packing: bitta ustun (K qiymat) ni [K x 1] ko'rinishida — K bo'ylab
// ketma-ket bitlarga joylash. `stride` = words64(K); pos/neg shu ustunga
// tegishli bo'lishi kerak (c*stride offset bilan chaqiriladi).
// ---------------------------------------------------------------------------

inline void pack_ternary_column(const std::int8_t* values, std::size_t k,
                                std::size_t stride_words,
                                std::uint64_t* pos, std::uint64_t* neg) noexcept {
    std::memset(pos, 0, stride_words * sizeof(std::uint64_t));
    std::memset(neg, 0, stride_words * sizeof(std::uint64_t));
    for (std::size_t i = 0; i < k; ++i) {
        const std::int8_t v = values[i];
        if (v > 0) {
            pos[i >> 6] |= (std::uint64_t{1} << (i & 63u));
        } else if (v < 0) {
            neg[i >> 6] |= (std::uint64_t{1} << (i & 63u));
        }
    }
}

inline void pack_ternary32_column(const std::int8_t* values, std::size_t k,
                                  std::size_t stride_words,
                                  std::uint32_t* pos, std::uint32_t* neg) noexcept {
    std::memset(pos, 0, stride_words * sizeof(std::uint32_t));
    std::memset(neg, 0, stride_words * sizeof(std::uint32_t));
    for (std::size_t i = 0; i < k; ++i) {
        const std::int8_t v = values[i];
        if (v > 0) {
            pos[i >> 5] |= (std::uint32_t{1} << (i & 31u));
        } else if (v < 0) {
            neg[i >> 5] |= (std::uint32_t{1} << (i & 31u));
        }
    }
}
inline std::int8_t unpack_ternary(const std::uint64_t* pos,
                                  const std::uint64_t* neg,
                                  std::size_t r, std::size_t c,
                                  std::size_t stride_words) noexcept {
    const std::uint64_t* p = pos + r * stride_words;
    const std::uint64_t* n = neg + r * stride_words;
    const bool bp = bit_get(p, c);
    const bool bn = bit_get(n, c);
    return static_cast<std::int8_t>(static_cast<int>(bp) - static_cast<int>(bn));
}

// ---------------------------------------------------------------------------
// MappedFile — faylni mmap orqali xotiraga ulash (POSIX; Windows fallback).
// Qaytarilgan xotira faqat o'qish uchun; close() dan keyin foydalanilmaydi.
// ---------------------------------------------------------------------------

class MappedFile {
public:
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            data_ = other.data_;
            size_ = other.size_;
#if defined(BT_WINDOWS)
            file_ = other.file_;
            mapping_ = other.mapping_;
            other.file_ = nullptr;
            other.mapping_ = nullptr;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    ~MappedFile() { close(); }

    static MappedFile open(const std::string& path, std::string* err = nullptr) {
        MappedFile m;
#if defined(BT_WINDOWS)
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) {
            if (err) *err = "fayl ochilmadi: " + path;
            return m;
        }
        LARGE_INTEGER sz{};
        GetFileSizeEx(f, &sz);
        m.file_ = f;
        m.size_ = static_cast<std::size_t>(sz.QuadPart);
        HANDLE map = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map) {
            CloseHandle(f);
            m.file_ = nullptr;
            if (err) *err = "CreateFileMapping xato";
            return m;
        }
        m.mapping_ = map;
        m.data_ = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
        if (!m.data_) {
            if (err) *err = "MapViewOfFile xato";
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (err) *err = "fayl ochilmadi: " + path;
            return m;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            if (err) *err = "fayl stat xato yoki bo'sh";
            ::close(fd);
            return m;
        }
        void* p = ::mmap(nullptr, static_cast<std::size_t>(st.st_size),
                         PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            if (err) *err = "mmap xato";
            ::close(fd);
            return m;
        }
        m.fd_ = fd;
        m.data_ = p;
        m.size_ = static_cast<std::size_t>(st.st_size);
#endif
        return m;
    }

    void close() noexcept {
        if (!data_) {
#if !defined(BT_WINDOWS)
            if (fd_ >= 0) {
                ::close(fd_);
                fd_ = -1;
            }
#endif
            return;
        }
#if defined(BT_WINDOWS)
        UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_) CloseHandle(file_);
        mapping_ = nullptr;
        file_ = nullptr;
#else
        ::munmap(data_, size_);
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        data_ = nullptr;
        size_ = 0;
    }

    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool valid() const noexcept { return data_ != nullptr; }

private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(BT_WINDOWS)
    HANDLE file_ = nullptr;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace bt
