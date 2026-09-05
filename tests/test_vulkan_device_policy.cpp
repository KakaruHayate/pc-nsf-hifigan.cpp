// SPDX-License-Identifier: MIT
#include "ggml-vulkan-device-policy.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr auto local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        constexpr auto visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        constexpr auto coherent = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkPhysicalDeviceMemoryProperties memory{};
        memory.memoryHeapCount = 2;
        memory.memoryHeaps[0].size = 1024;
        memory.memoryHeaps[1].size = 128;
        memory.memoryTypeCount = 5;
        // Deliberately enumerate mapped VRAM before device-only VRAM.
        memory.memoryTypes[0] = {local | visible | coherent, 0};
        memory.memoryTypes[1] = {visible | coherent, 0};
        memory.memoryTypes[2] = {local, 0};
        memory.memoryTypes[3] = {local, 1};
        memory.memoryTypes[4] = {local, 0};
        VkMemoryRequirements req{256, 16, 31};
        require(ggml_vk_memory_type_indices(memory, req, local) == std::vector<uint32_t>({2,4,0}),
                "device-only types must precede mapped VRAM; undersized heap must be excluded");
        require(ggml_vk_memory_type_indices(memory, req, local | visible | coherent) == std::vector<uint32_t>({0}),
                "UMA/explicit host-visible requests must retain mapped VRAM");
        require(ggml_vk_memory_type_indices(memory, req, visible | coherent) == std::vector<uint32_t>({0,1}),
                "staging allocations must retain host access");
        req.memoryTypeBits = 3;
        require(ggml_vk_memory_type_indices(memory, req, local) == std::vector<uint32_t>({0}),
                "devices without device-only VRAM must retain a valid fallback");
        req.memoryTypeBits = 2;
        require(ggml_vk_memory_type_indices(memory, req, local).empty(),
                "system memory must not silently satisfy a device-local request");
        require(ggml_vk_memory_type_indices(memory, req, visible | coherent) == std::vector<uint32_t>({1}),
                "explicit system-memory fallback must remain available");
        require(ggml_vk_prefer_device_only_memory(false, false, false), "discrete default");
        require(!ggml_vk_prefer_device_only_memory(true, false, false), "UMA default");
        require(!ggml_vk_prefer_device_only_memory(false, true, false), "explicit mapped preference");
        require(ggml_vk_prefer_device_only_memory(false, true, true), "disable override takes precedence");
        require(!ggml_vk_prefer_device_only_memory(true, true, true), "UMA keeps existing allocation policy");

        VkPhysicalDeviceLimits limits{};
        limits.maxComputeWorkGroupInvocations = 256;
        limits.maxComputeWorkGroupSize[0] = 256;
        limits.maxComputeSharedMemorySize = 32768;
        using variant = ggml_vk_conv_direct_variant;
        require(ggml_vk_select_conv_direct_variant(16, limits) == variant::e16, "small channel tile");
        require(ggml_vk_select_conv_direct_variant(32, limits) == variant::e32, "medium channel tile");
        require(ggml_vk_select_conv_direct_variant(64, limits) == variant::w128, "large channel tile");
        require(ggml_vk_conv_direct_shared_bytes(64,128) == 18560, "w128 shared memory contract");
        require(ggml_vk_conv_direct_shared_bytes(64,64) == 15488, "e64 shared memory contract");
        limits.maxComputeSharedMemorySize = 16384;
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::e64, "16 KiB shared-memory fallback");
        limits.maxComputeSharedMemorySize = 32768;
        limits.maxComputeWorkGroupInvocations = 128;
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::e64, "128-thread fallback");
        limits.maxComputeWorkGroupInvocations = 256;
        limits.maxComputeWorkGroupSize[0] = 128;
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::e64, "local X limit");
        limits.maxComputeSharedMemorySize = ggml_vk_conv_direct_shared_bytes(32,64);
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::e32, "smaller tile fallback");
        limits.maxComputeSharedMemorySize = ggml_vk_conv_direct_shared_bytes(16,64);
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::e16, "minimum tile fallback");
        --limits.maxComputeSharedMemorySize;
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::none, "unsupported shared-memory limit");
        limits.maxComputeSharedMemorySize = 32768;
        limits.maxComputeWorkGroupSize[0] = 64;
        require(ggml_vk_select_conv_direct_variant(256, limits) == variant::none, "unsupported local size");
        std::cout << "Vulkan memory and convolution device policies passed (no GPU required)\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
