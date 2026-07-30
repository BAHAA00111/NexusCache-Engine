#ifndef NEXUSCACHE_PINNED_MEMORY_HPP
#define NEXUSCACHE_PINNED_MEMORY_HPP

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <cstdint>

namespace nexuscache {


 // RAII Wrapper for Allocation and Deallocation of Pinned (Page-Locked) Host Memory.

class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t bytes, unsigned int flags = cudaHostAllocDefault);
    ~PinnedBuffer();

    // Disable copying to enforce singular ownership of physical host memory allocation
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    // Support move semantics
    PinnedBuffer(PinnedBuffer&& other) noexcept;
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

    void* data() const { return host_ptr_; }
    size_t size_bytes() const { return bytes_; }

    
    //Zero-copies or maps the pinned buffer into a PyTorch Host CPU Tensor.
    torch::Tensor to_tensor(torch::ScalarType dtype, const std::vector<int64_t>& shape) const;

private:
    void* host_ptr_{nullptr};
    size_t bytes_{0};
};

/**
    High-Throughput Asynchronous Memory Transfer Manager for Host-to-Device (H2D)
        and Device-to-Host (D2H) streaming using dedicated CUDA Streams.
 */
class AsyncTransferManager {
public:
    explicit AsyncTransferManager(int device_id = 0);
    ~AsyncTransferManager() = default;


    void copy_host_to_device_async(const torch::Tensor& src, torch::Tensor& dst);


    void copy_device_to_host_async(const torch::Tensor& src, torch::Tensor& dst);


    void synchronize_stream();

    c10::cuda::CUDAStream get_cuda_stream() const { return stream_; }

private:
    int device_id_;
    c10::cuda::CUDAStream stream_;
};

} // namespace nexuscache

#endif // NEXUSCACHE_PINNED_MEMORY_HPP