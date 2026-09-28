// SPDX-License-Identifier: MIT
// vulkan_wrapper.hpp — Vulkan compute orqali bit-packed ternar matmul.
//
// Kutubxona yuklash dinamik (dlopen / LoadLibrary) — build vaqtida Vulkan
// SDK talab qilinmaydi, faqat header. Header bo'lmasa stub ishlaydi.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bt::gpu {

struct DeviceInfo {
    std::string name;
    std::uint64_t vram_bytes = 0;
    std::uint32_t api_version = 0;
    bool valid = false;
};

// Bitta layer tavsifi (og'irlik so'z-offseti bilan).
struct LayerDesc {
    std::uint32_t m = 0;        // output qatorlar
    std::uint32_t k_words = 0;  // K/32 (stride)
    std::uint32_t w_base = 0;   // wpos/wneg massividagi so'z-offset
};

class VulkanEngine {
public:
    VulkanEngine();
    ~VulkanEngine();
    VulkanEngine(const VulkanEngine&) = delete;
    VulkanEngine& operator=(const VulkanEngine&) = delete;

    // `shader_path` bo'sh bo'lsa: BT_SHADER env → exe yonidagi binary_matmul.spv
    bool init(std::string* error = nullptr, const std::string& shader_path = "");
    bool available() const noexcept;
    const DeviceInfo& info() const noexcept;

    // Og'irliklarni (pos/neg planes, uint32) GPU xotirasiga bir marta yuklash.
    bool upload_weights(std::size_t total_words,
                        const std::uint32_t* wpos,
                        const std::uint32_t* wneg,
                        std::string* error = nullptr);

    // Kirish (hidden state) — barcha layerlar uchun umumiy (benchmark rejimi).
    bool upload_input(std::uint32_t batch, std::uint32_t k_words,
                      const std::uint32_t* xpos, const std::uint32_t* xneg,
                      std::string* error = nullptr);

    // Mavjud input buffer'ga yangi hidden state yozish (per-token).
    bool update_input(const std::uint32_t* xpos, const std::uint32_t* xneg,
                      std::string* error = nullptr);

    // Layerlar ketma-ketligini bitta command buffer'ga yozish.
    bool build(const std::vector<LayerDesc>& layers, std::uint32_t batch,
               std::string* error = nullptr);

    // Bitta "token" (yoki batch) o'tishini bajarish (submit + fence wait).
    bool execute(std::string* error = nullptr);

    // Oxirgi layer natijasini o'qish (m*batch int32).
    bool read_output(std::int32_t* out, std::size_t count,
                     std::string* error = nullptr);

    void shutdown() noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace bt::gpu
