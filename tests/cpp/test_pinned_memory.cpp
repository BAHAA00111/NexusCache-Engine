#include <gtest/gtest.h>
#include "nexuscache/pinned_memory.hpp"

TEST(PinnedMemoryTest, PinnedBufferAllocationAndAsyncTransfer) {
    size_t num_elements = 1024;
    size_t bytes = num_elements * sizeof(float);

    // 1. Allocate Pinned Host Memory
    nexuscache::PinnedBuffer host_buffer(bytes);
    EXPECT_EQ(host_buffer.size_bytes(), bytes);

    torch::Tensor host_tensor = host_buffer.to_tensor(torch::kFloat32, {static_cast<int64_t>(num_elements)});
    EXPECT_TRUE(host_tensor.is_pinned());

    // Fill Host Tensor with dummy data
    host_tensor.fill_(3.14159f);

    // 2. Allocate CUDA Device Tensor
    torch::Tensor device_tensor = torch::empty({static_cast<int64_t>(num_elements)}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

    // 3. Perform Asynchronous Host-to-Device Copy
    nexuscache::AsyncTransferManager transfer_mgr(0);
    transfer_mgr.copy_host_to_device_async(host_tensor, device_tensor);
    transfer_mgr.synchronize_stream();

    // Verify GPU Tensor contains copied values
    torch::Tensor host_result = device_tensor.cpu();
    EXPECT_FLOAT_EQ(host_result[0].item<float>(), 3.14159f);
    EXPECT_FLOAT_EQ(host_result[num_elements - 1].item<float>(), 3.14159f);
}