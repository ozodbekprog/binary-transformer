// SPDX-License-Identifier: MIT
// vulkan_wrapper.cpp — Vulkan compute pipeline (dinamik yuklash, SDK'siz).
#include "gpu/vulkan_wrapper.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <type_traits>

#ifndef BT_HAVE_VULKAN
#  if defined(__has_include)
#    if __has_include(<vulkan/vulkan.h>)
#      define BT_HAVE_VULKAN 1
#    else
#      define BT_HAVE_VULKAN 0
#    endif
#  else
#    define BT_HAVE_VULKAN 0
#  endif
#endif

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <unistd.h>
#  if defined(__APPLE__)
#    include <mach-o/dyld.h>
#  endif
#endif

#if BT_HAVE_VULKAN
#  define VK_NO_PROTOTYPES
#  include <vulkan/vulkan.h>
#endif

namespace bt::gpu {
namespace {

#if BT_HAVE_VULKAN

// --- Platforma abstraksiyasi -------------------------------------------------
void* lib_open(const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(name));
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}
void* lib_sym(void* handle, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}
void lib_close(void* handle) {
#if defined(_WIN32)
    if (handle) FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    if (handle) dlclose(handle);
#endif
}

std::string exe_dir() {
    char buf[4096] = {};
#if defined(_WIN32)
    DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return {};
#elif defined(__APPLE__)
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) != 0) return {};
#else
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
#endif
    std::string path(buf);
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string{} : path.substr(0, pos);
}

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    out.resize(size);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
    return static_cast<bool>(f) || f.eof();
}

std::string find_shader(const std::string& explicit_path) {
    std::vector<std::string> candidates;
    if (!explicit_path.empty()) candidates.push_back(explicit_path);
    if (const char* env = std::getenv("BT_SHADER")) candidates.emplace_back(env);
    const std::string dir = exe_dir();
    if (!dir.empty()) candidates.push_back(dir + "/binary_matmul.spv");
    candidates.emplace_back("binary_matmul.spv");
    candidates.emplace_back("bin/binary_matmul.spv");
    candidates.emplace_back("build/bin/binary_matmul.spv");
    for (const auto& c : candidates) {
        std::ifstream f(c, std::ios::binary);
        if (f.good()) return c;
    }
    return {};
}

std::string vk_error(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "ok";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        default: return "VkResult(" + std::to_string(static_cast<int>(r)) + ")";
    }
}

constexpr std::uint64_t kStagingChunk = 16ull * 1024 * 1024;

#endif  // BT_HAVE_VULKAN

}  // namespace

// ===========================================================================
// Impl
// ===========================================================================
struct VulkanEngine::Impl {
    bool initialized = false;
    DeviceInfo info;

#if BT_HAVE_VULKAN
    void* lib = nullptr;

    // global / instance
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkCreateInstance create_instance = nullptr;
    PFN_vkEnumeratePhysicalDevices enumerate_devices = nullptr;
    PFN_vkGetPhysicalDeviceProperties get_phys_props = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_families = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_props = nullptr;
    PFN_vkCreateDevice create_device = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;

    // device
    PFN_vkGetDeviceQueue get_queue = nullptr;
    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements get_buffer_reqs = nullptr;
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkBindBufferMemory bind_buffer = nullptr;
    PFN_vkMapMemory map_memory = nullptr;
    PFN_vkUnmapMemory unmap_memory = nullptr;
    PFN_vkCreateShaderModule create_shader = nullptr;
    PFN_vkCreateDescriptorSetLayout create_dsl = nullptr;
    PFN_vkCreatePipelineLayout create_pl = nullptr;
    PFN_vkCreateComputePipelines create_pipelines = nullptr;
    PFN_vkCreateDescriptorPool create_dpool = nullptr;
    PFN_vkAllocateDescriptorSets alloc_dsets = nullptr;
    PFN_vkUpdateDescriptorSets update_dsets = nullptr;
    PFN_vkCreateCommandPool create_cpool = nullptr;
    PFN_vkAllocateCommandBuffers alloc_cmds = nullptr;
    PFN_vkBeginCommandBuffer begin_cmd = nullptr;
    PFN_vkCmdBindPipeline bind_pipeline = nullptr;
    PFN_vkCmdBindDescriptorSets bind_dsets = nullptr;
    PFN_vkCmdPushConstants push_constants = nullptr;
    PFN_vkCmdDispatch dispatch = nullptr;
    PFN_vkCmdPipelineBarrier pipeline_barrier = nullptr;
    PFN_vkCmdCopyBuffer copy_buffer = nullptr;
    PFN_vkEndCommandBuffer end_cmd = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkWaitForFences wait_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    PFN_vkDestroyShaderModule destroy_shader = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_dsl = nullptr;
    PFN_vkDestroyPipelineLayout destroy_pl = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline = nullptr;
    PFN_vkDestroyDescriptorPool destroy_dpool = nullptr;
    PFN_vkDestroyCommandPool destroy_cpool = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    // buffers
    VkBuffer wpos_buf = VK_NULL_HANDLE; VkDeviceMemory wpos_mem = VK_NULL_HANDLE;
    VkBuffer wneg_buf = VK_NULL_HANDLE; VkDeviceMemory wneg_mem = VK_NULL_HANDLE;
    VkBuffer xpos_buf = VK_NULL_HANDLE; VkDeviceMemory xpos_mem = VK_NULL_HANDLE;
    VkBuffer xneg_buf = VK_NULL_HANDLE; VkDeviceMemory xneg_mem = VK_NULL_HANDLE;
    VkBuffer out_buf = VK_NULL_HANDLE;  VkDeviceMemory out_mem = VK_NULL_HANDLE;
    void* xpos_map = nullptr;
    void* xneg_map = nullptr;
    std::size_t x_bytes = 0;
    void* out_map = nullptr;
    std::size_t out_bytes = 0;

    bool built = false;
    std::uint32_t batch = 1;

    std::uint32_t find_memory(std::uint32_t type_bits, VkMemoryPropertyFlags props) const {
        for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    bool make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags props,
                     VkBuffer* buf, VkDeviceMemory* mem,
                     std::string* error) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r = create_buffer(device, &bci, nullptr, buf);
        if (r != VK_SUCCESS) {
            if (error) *error = "vkCreateBuffer: " + vk_error(r);
            return false;
        }
        VkMemoryRequirements req{};
        get_buffer_reqs(device, *buf, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory(req.memoryTypeBits, props);
        if (mai.memoryTypeIndex == UINT32_MAX) {
            if (error) *error = "mos VkDeviceMemory turi topilmadi";
            return false;
        }
        r = allocate_memory(device, &mai, nullptr, mem);
        if (r != VK_SUCCESS) {
            if (error) *error = "vkAllocateMemory: " + vk_error(r);
            return false;
        }
        r = bind_buffer(device, *buf, *mem, 0);
        if (r != VK_SUCCESS) {
            if (error) *error = "vkBindBufferMemory: " + vk_error(r);
            return false;
        }
        return true;
    }

    // Katta buffer'ni bo'laklab (staging orqali) DEVICE_LOCAL xotiraga yozish.
    bool upload_device(VkBuffer dst, const void* src, std::size_t bytes,
                       std::string* error) {
        std::size_t offset = 0;
        const std::size_t chunk = static_cast<std::size_t>(kStagingChunk);
        while (offset < bytes) {
            const std::size_t n = std::min(chunk, bytes - offset);
            VkBuffer staging = VK_NULL_HANDLE;
            VkDeviceMemory staging_mem = VK_NULL_HANDLE;
            if (!make_buffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             &staging, &staging_mem, error)) {
                return false;
            }
            void* map = nullptr;
            map_memory(device, staging_mem, 0, n, 0, &map);
            std::memcpy(map, static_cast<const std::uint8_t*>(src) + offset, n);
            unmap_memory(device, staging_mem);

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            begin_cmd(cmd, &bi);
            VkBufferCopy region{};
            region.srcOffset = 0;
            region.dstOffset = offset;
            region.size = n;
            copy_buffer(cmd, staging, dst, 1, &region);
            end_cmd(cmd);
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            reset_fences(device, 1, &fence);
            queue_submit(queue, 1, &si, fence);
            wait_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);

            destroy_buffer(device, staging, nullptr);
            free_memory(device, staging_mem, nullptr);
            offset += n;
        }
        return true;
    }

    void record_barrier() {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        pipeline_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
    }

    void destroy_all() {
        if (device) {
            if (cmd) destroy_cpool(device, cpool, nullptr);
            if (out_buf) destroy_buffer(device, out_buf, nullptr);
            if (out_mem) free_memory(device, out_mem, nullptr);
            if (xpos_buf) destroy_buffer(device, xpos_buf, nullptr);
            if (xpos_mem) free_memory(device, xpos_mem, nullptr);
            if (xneg_buf) destroy_buffer(device, xneg_buf, nullptr);
            if (xneg_mem) free_memory(device, xneg_mem, nullptr);
            if (wpos_buf) destroy_buffer(device, wpos_buf, nullptr);
            if (wpos_mem) free_memory(device, wpos_mem, nullptr);
            if (wneg_buf) destroy_buffer(device, wneg_buf, nullptr);
            if (wneg_mem) free_memory(device, wneg_mem, nullptr);
            if (fence && destroy_fence) destroy_fence_stub();
            if (pipeline) destroy_pipeline(device, pipeline, nullptr);
            if (pl) destroy_pl(device, pl, nullptr);
            if (dsl) destroy_dsl(device, dsl, nullptr);
            if (dpool) destroy_dpool(device, dpool, nullptr);
            destroy_device(device, nullptr);
        }
        if (instance) destroy_instance(instance, nullptr);
        lib_close(lib);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
        lib = nullptr;
    }

    PFN_vkDestroyFence destroy_fence = nullptr;
    void destroy_fence_stub() { destroy_fence(device, fence, nullptr); }
#endif  // BT_HAVE_VULKAN
};

// ===========================================================================
// Public API
// ===========================================================================

VulkanEngine::VulkanEngine() : impl_(new Impl()) {}
VulkanEngine::~VulkanEngine() { shutdown(); }

bool VulkanEngine::available() const noexcept { return impl_ && impl_->initialized; }
const DeviceInfo& VulkanEngine::info() const noexcept { return impl_->info; }

bool VulkanEngine::init(std::string* error, const std::string& shader_path) {
#if !BT_HAVE_VULKAN
    (void)error;
    (void)shader_path;
    if (error) *error = "Vulkan backend o'chirilgan (vulkan/vulkan.h topilmadi)";
    return false;
#else
    Impl& d = *impl_;
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        d.destroy_all();
        d.initialized = false;
        return false;
    };

    if (d.initialized) return true;

#if defined(_WIN32)
    d.lib = lib_open("vulkan-1.dll");
#elif defined(__APPLE__)
    d.lib = lib_open("libvulkan.dylib");
    if (!d.lib) d.lib = lib_open("libvulkan.1.dylib");
    if (!d.lib) d.lib = lib_open("libMoltenVK.dylib");
#else
    d.lib = lib_open("libvulkan.so.1");
    if (!d.lib) d.lib = lib_open("libvulkan.so");
#endif
    if (!d.lib) return fail("Vulkan loader topilmadi (libvulkan.so.1 / MoltenVK)");

    d.gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(lib_sym(d.lib, "vkGetInstanceProcAddr"));
    if (!d.gipa) return fail("vkGetInstanceProcAddr topilmadi");
    d.create_instance = reinterpret_cast<PFN_vkCreateInstance>(
        d.gipa(nullptr, "vkCreateInstance"));
    if (!d.create_instance) return fail("vkCreateInstance topilmadi");

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "binary-transformer";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkResult r = d.create_instance(&ici, nullptr, &d.instance);
    if (r != VK_SUCCESS) return fail("vkCreateInstance: " + vk_error(r));

    auto load_i = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::decay_t<decltype(fn)>>(d.gipa(d.instance, name));
        return fn != nullptr;
    };
    if (!load_i(d.enumerate_devices, "vkEnumeratePhysicalDevices") ||
        !load_i(d.get_phys_props, "vkGetPhysicalDeviceProperties") ||
        !load_i(d.get_queue_families, "vkGetPhysicalDeviceQueueFamilyProperties") ||
        !load_i(d.get_memory_props, "vkGetPhysicalDeviceMemoryProperties") ||
        !load_i(d.create_device, "vkCreateDevice") ||
        !load_i(d.gdpa, "vkGetDeviceProcAddr")) {
        return fail("instance funksiyalarini yuklab bo'lmadi");
    }

    std::uint32_t count = 0;
    d.enumerate_devices(d.instance, &count, nullptr);
    if (count == 0) return fail("Vulkan qurilma topilmadi");
    std::vector<VkPhysicalDevice> devices(count);
    d.enumerate_devices(d.instance, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties best_props{};
    std::uint64_t best_vram = 0;
    int best_score = -1;
    for (VkPhysicalDevice dev : devices) {
        std::uint32_t qn = 0;
        d.get_queue_families(dev, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qs(qn);
        d.get_queue_families(dev, &qn, qs.data());
        bool compute = false;
        for (const auto& q : qs) {
            if (q.queueFlags & VK_QUEUE_COMPUTE_BIT) { compute = true; break; }
        }
        if (!compute) continue;
        VkPhysicalDeviceProperties p{};
        d.get_phys_props(dev, &p);
        // Reyting: diskret GPU > integrallangan > virtual > CPU > boshqa.
        // (VkPhysicalDeviceType raqamlari bu tartibda EMAS — ataylab scoring.)
        int score = 0;
        switch (p.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score = 50; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score = 40; break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score = 30; break;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            score = 10; break;
            default:                                     score = 0;  break;
        }
        VkPhysicalDeviceMemoryProperties mp{};
        d.get_memory_props(dev, &mp);
        std::uint64_t vram = 0;
        for (std::uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                vram += mp.memoryHeaps[i].size;
            }
        }
        // Bir xil toifada ko'proq VRAM yutadi
        if (score > best_score || (score == best_score && vram > best_vram)) {
            best = dev;
            best_props = p;
            best_vram = vram;
            best_score = score;
        }
    }
    if (best == VK_NULL_HANDLE) return fail("Compute queue'li Vulkan qurilma yo'q");
    d.phys = best;

    std::uint32_t qn = 0;
    d.get_queue_families(best, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    d.get_queue_families(best, &qn, qs.data());
    std::uint32_t qfam = UINT32_MAX;
    for (std::uint32_t i = 0; i < qn; ++i) {
        const auto flags = qs[i].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT)) {
            qfam = i;
            break;
        }
    }
    if (qfam == UINT32_MAX) {
        for (std::uint32_t i = 0; i < qn; ++i) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
        }
    }
    if (qfam == UINT32_MAX) return fail("Compute queue family topilmadi");

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    r = d.create_device(best, &dci, nullptr, &d.device);
    if (r != VK_SUCCESS) return fail("vkCreateDevice: " + vk_error(r));

    auto load_d = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::decay_t<decltype(fn)>>(d.gdpa(d.device, name));
        if (!fn) fn = reinterpret_cast<std::decay_t<decltype(fn)>>(d.gipa(d.instance, name));
        return fn != nullptr;
    };
    const bool ok = load_d(d.get_queue, "vkGetDeviceQueue") &&
        load_d(d.create_buffer, "vkCreateBuffer") &&
        load_d(d.get_buffer_reqs, "vkGetBufferMemoryRequirements") &&
        load_d(d.allocate_memory, "vkAllocateMemory") &&
        load_d(d.bind_buffer, "vkBindBufferMemory") &&
        load_d(d.map_memory, "vkMapMemory") && load_d(d.unmap_memory, "vkUnmapMemory") &&
        load_d(d.create_shader, "vkCreateShaderModule") &&
        load_d(d.create_dsl, "vkCreateDescriptorSetLayout") &&
        load_d(d.create_pl, "vkCreatePipelineLayout") &&
        load_d(d.create_pipelines, "vkCreateComputePipelines") &&
        load_d(d.create_dpool, "vkCreateDescriptorPool") &&
        load_d(d.alloc_dsets, "vkAllocateDescriptorSets") &&
        load_d(d.update_dsets, "vkUpdateDescriptorSets") &&
        load_d(d.create_cpool, "vkCreateCommandPool") &&
        load_d(d.alloc_cmds, "vkAllocateCommandBuffers") &&
        load_d(d.begin_cmd, "vkBeginCommandBuffer") &&
        load_d(d.bind_pipeline, "vkCmdBindPipeline") &&
        load_d(d.bind_dsets, "vkCmdBindDescriptorSets") &&
        load_d(d.push_constants, "vkCmdPushConstants") &&
        load_d(d.dispatch, "vkCmdDispatch") &&
        load_d(d.pipeline_barrier, "vkCmdPipelineBarrier") &&
        load_d(d.copy_buffer, "vkCmdCopyBuffer") &&
        load_d(d.end_cmd, "vkEndCommandBuffer") &&
        load_d(d.queue_submit, "vkQueueSubmit") &&
        load_d(d.create_fence, "vkCreateFence") &&
        load_d(d.wait_fences, "vkWaitForFences") &&
        load_d(d.reset_fences, "vkResetFences") &&
        load_d(d.destroy_buffer, "vkDestroyBuffer") &&
        load_d(d.free_memory, "vkFreeMemory") &&
        load_d(d.destroy_shader, "vkDestroyShaderModule") &&
        load_d(d.destroy_dsl, "vkDestroyDescriptorSetLayout") &&
        load_d(d.destroy_pl, "vkDestroyPipelineLayout") &&
        load_d(d.destroy_pipeline, "vkDestroyPipeline") &&
        load_d(d.destroy_dpool, "vkDestroyDescriptorPool") &&
        load_d(d.destroy_cpool, "vkDestroyCommandPool") &&
        load_d(d.destroy_device, "vkDestroyDevice") &&
        load_d(d.destroy_instance, "vkDestroyInstance") &&
        load_d(d.destroy_fence, "vkDestroyFence");
    if (!ok) return fail("device funksiyalarini yuklab bo'lmadi");

    d.get_queue(d.device, qfam, 0, &d.queue);
    d.get_memory_props(best, &d.mem_props);

    // VRAM hajmi (DEVICE_LOCAL heap)
    std::uint64_t vram = 0;
    for (std::uint32_t i = 0; i < d.mem_props.memoryHeapCount; ++i) {
        if (d.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vram += d.mem_props.memoryHeaps[i].size;
        }
    }
    d.info.name = best_props.deviceName;
    d.info.vram_bytes = vram;
    d.info.api_version = best_props.apiVersion;
    d.info.valid = true;

    // Command pool + fence
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = qfam;
    r = d.create_cpool(d.device, &cpci, nullptr, &d.cpool);
    if (r != VK_SUCCESS) return fail("vkCreateCommandPool: " + vk_error(r));
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = d.cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    r = d.alloc_cmds(d.device, &cbai, &d.cmd);
    if (r != VK_SUCCESS) return fail("vkAllocateCommandBuffers: " + vk_error(r));
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = d.create_fence(d.device, &fci, nullptr, &d.fence);
    if (r != VK_SUCCESS) return fail("vkCreateFence: " + vk_error(r));

    // Shader SPIR-V
    const std::string spv_path = find_shader(shader_path);
    if (spv_path.empty()) {
        return fail("SPIR-V topilmadi: binary_matmul.spv (glslc bilan yig'ing yoki "
                    "BT_SHADER=... bering)");
    }
    std::vector<std::uint8_t> spirv;
    if (!read_file(spv_path, spirv) || spirv.empty()) {
        return fail("SPIR-V o'qilmadi: " + spv_path);
    }
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size();
    smci.pCode = reinterpret_cast<const std::uint32_t*>(spirv.data());
    VkShaderModule shader = VK_NULL_HANDLE;
    r = d.create_shader(d.device, &smci, nullptr, &shader);
    if (r != VK_SUCCESS) return fail("vkCreateShaderModule: " + vk_error(r));

    // Descriptor set layout: 5 storage buffer
    VkDescriptorSetLayoutBinding bindings[5]{};
    for (std::uint32_t i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 5;
    dslci.pBindings = bindings;
    r = d.create_dsl(d.device, &dslci, nullptr, &d.dsl);
    if (r != VK_SUCCESS) return fail("vkCreateDescriptorSetLayout: " + vk_error(r));

    // Push constants: m, k_words, batch, mode, w_base (20 bayt)
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 5 * sizeof(std::uint32_t);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &d.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    r = d.create_pl(d.device, &plci, nullptr, &d.pl);
    if (r != VK_SUCCESS) return fail("vkCreatePipelineLayout: " + vk_error(r));

    VkComputePipelineCreateInfo cpci2{};
    cpci2.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci2.stage.module = shader;
    cpci2.stage.pName = "main";
    cpci2.layout = d.pl;
    r = d.create_pipelines(d.device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &d.pipeline);
    d.destroy_shader(d.device, shader, nullptr);
    if (r != VK_SUCCESS) return fail("vkCreateComputePipelines: " + vk_error(r));

    // Descriptor pool + set
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 5;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    r = d.create_dpool(d.device, &dpci, nullptr, &d.dpool);
    if (r != VK_SUCCESS) return fail("vkCreateDescriptorPool: " + vk_error(r));
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = d.dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &d.dsl;
    r = d.alloc_dsets(d.device, &dsai, &d.dset);
    if (r != VK_SUCCESS) return fail("vkAllocateDescriptorSets: " + vk_error(r));

    d.initialized = true;
    return true;
#endif  // BT_HAVE_VULKAN
}

bool VulkanEngine::upload_weights(std::size_t total_words,
                                  const std::uint32_t* wpos,
                                  const std::uint32_t* wneg,
                                  std::string* error) {
#if !BT_HAVE_VULKAN
    (void)total_words; (void)wpos; (void)wneg;
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.initialized) {
        if (error) *error = "VulkanEngine init qilinmagan";
        return false;
    }
    const VkDeviceSize bytes = total_words * sizeof(std::uint32_t);
    if (!d.make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &d.wpos_buf, &d.wpos_mem, error)) {
        return false;
    }
    if (!d.make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       &d.wneg_buf, &d.wneg_mem, error)) {
        return false;
    }
    if (!d.upload_device(d.wpos_buf, wpos, bytes, error)) return false;
    if (!d.upload_device(d.wneg_buf, wneg, bytes, error)) return false;
    return true;
#endif
}

bool VulkanEngine::upload_input(std::uint32_t batch, std::uint32_t k_words,
                                const std::uint32_t* xpos, const std::uint32_t* xneg,
                                std::string* error) {
#if !BT_HAVE_VULKAN
    (void)batch; (void)k_words; (void)xpos; (void)xneg;
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.initialized) {
        if (error) *error = "VulkanEngine init qilinmagan";
        return false;
    }
    const VkDeviceSize bytes = VkDeviceSize(batch) * k_words * sizeof(std::uint32_t);
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!d.make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host,
                       &d.xpos_buf, &d.xpos_mem, error)) return false;
    if (!d.make_buffer(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host,
                       &d.xneg_buf, &d.xneg_mem, error)) return false;
    d.map_memory(d.device, d.xpos_mem, 0, bytes, 0, &d.xpos_map);
    d.map_memory(d.device, d.xneg_mem, 0, bytes, 0, &d.xneg_map);
    std::memcpy(d.xpos_map, xpos, static_cast<std::size_t>(bytes));
    std::memcpy(d.xneg_map, xneg, static_cast<std::size_t>(bytes));
    d.x_bytes = static_cast<std::size_t>(bytes);
    d.batch = batch;
    return true;
#endif
}

bool VulkanEngine::update_input(const std::uint32_t* xpos, const std::uint32_t* xneg,
                                std::string* error) {
#if !BT_HAVE_VULKAN
    (void)xpos; (void)xneg;
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.initialized || !d.xpos_map || d.x_bytes == 0) {
        if (error) *error = "input buffer tayyor emas (upload_input chaqirilmagan)";
        return false;
    }
    std::memcpy(d.xpos_map, xpos, d.x_bytes);
    std::memcpy(d.xneg_map, xneg, d.x_bytes);
    return true;
#endif
}

bool VulkanEngine::build(const std::vector<LayerDesc>& layers, std::uint32_t batch,
                         std::string* error) {
#if !BT_HAVE_VULKAN
    (void)layers; (void)batch;
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.initialized) {
        if (error) *error = "VulkanEngine init qilinmagan";
        return false;
    }
    if (layers.empty()) {
        if (error) *error = "Layer ro'yxati bo'sh";
        return false;
    }
    std::uint32_t max_m = 0;
    for (const auto& l : layers) max_m = std::max(max_m, l.m);

    d.out_bytes = std::size_t(max_m) * batch * sizeof(std::int32_t);
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!d.make_buffer(d.out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host,
                       &d.out_buf, &d.out_mem, error)) return false;
    d.map_memory(d.device, d.out_mem, 0, d.out_bytes, 0, &d.out_map);

    VkDescriptorBufferInfo infos[5]{};
    infos[0] = {d.wpos_buf, 0, VK_WHOLE_SIZE};
    infos[1] = {d.wneg_buf, 0, VK_WHOLE_SIZE};
    infos[2] = {d.xpos_buf, 0, VK_WHOLE_SIZE};
    infos[3] = {d.xneg_buf, 0, VK_WHOLE_SIZE};
    infos[4] = {d.out_buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[5]{};
    for (std::uint32_t i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = d.dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    d.update_dsets(d.device, 5, writes, 0, nullptr);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VkResult r = d.begin_cmd(d.cmd, &bi);
    if (r != VK_SUCCESS) {
        if (error) *error = "vkBeginCommandBuffer: " + vk_error(r);
        return false;
    }
    d.bind_pipeline(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
    d.bind_dsets(d.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);

    const std::uint32_t wg = 256;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const LayerDesc& l = layers[i];
        std::uint32_t pc[5] = {l.m, l.k_words, batch, 0u, l.w_base};
        d.push_constants(d.cmd, d.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(pc), pc);
        d.dispatch(d.cmd, (l.m + wg - 1) / wg, 1, 1);
        if (i + 1 < layers.size()) d.record_barrier();
    }
    r = d.end_cmd(d.cmd);
    if (r != VK_SUCCESS) {
        if (error) *error = "vkEndCommandBuffer: " + vk_error(r);
        return false;
    }
    d.built = true;
    return true;
#endif
}

bool VulkanEngine::execute(std::string* error) {
#if !BT_HAVE_VULKAN
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.built) {
        if (error) *error = "build() chaqirilmagan";
        return false;
    }
    d.reset_fences(d.device, 1, &d.fence);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cmd;
    VkResult r = d.queue_submit(d.queue, 1, &si, d.fence);
    if (r != VK_SUCCESS) {
        if (error) *error = "vkQueueSubmit: " + vk_error(r);
        return false;
    }
    r = d.wait_fences(d.device, 1, &d.fence, VK_TRUE, UINT64_MAX);
    if (r != VK_SUCCESS) {
        if (error) *error = "vkWaitForFences: " + vk_error(r);
        return false;
    }
    return true;
#endif
}

bool VulkanEngine::read_output(std::int32_t* out, std::size_t count,
                               std::string* error) {
#if !BT_HAVE_VULKAN
    (void)out; (void)count;
    if (error) *error = "Vulkan backend o'chirilgan";
    return false;
#else
    Impl& d = *impl_;
    if (!d.out_map) {
        if (error) *error = "output buffer tayyor emas";
        return false;
    }
    const std::size_t bytes = count * sizeof(std::int32_t);
    if (bytes > d.out_bytes) {
        if (error) *error = "read_output hajmi buffer'dan katta";
        return false;
    }
    std::memcpy(out, d.out_map, bytes);
    return true;
#endif
}

void VulkanEngine::shutdown() noexcept {
#if BT_HAVE_VULKAN
    if (impl_) {
        impl_->destroy_all();
        impl_->initialized = false;
    }
#endif
    delete impl_;
    impl_ = nullptr;
}

}  // namespace bt::gpu
