#include "nexuscache/pinned_memory.hpp"
#include <iostream>

namespace nexuscache {


// PinnedBuffer Implementation
PinnedBuffer::PinnedBuffer(size_t bytes, unsigned int flags)
    : bytes_(bytes) {
    if (bytes_ == 0) {
        throw std::invalid_argument("PinnedBuffer Error: Allocation size must be greater than 0 bytes.");
    }

    cudaError_t err = cudaHostAlloc(&host_ptr_, bytes_, flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("PinnedBuffer Error: cudaHostAlloc failed to allocate ") +
            std::to_string(bytes_) + " bytes: " + cudaGetErrorString(err)
        );
    }
}

PinnedBuffer::~PinnedBuffer() {
    if (host_ptr_ != nullptr) {
        cudaFreeHost(host_ptr_);
        host_ptr_ = nullptr;
    }
}

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : host_ptr_(other.host_ptr_), bytes_(other.bytes_) {
    other.host_ptr_ = nullptr;
    other.bytes_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
        if (host_ptr_ != nullptr) {
            cudaFreeHost(host_ptr_);
        }
        host_ptr_ = other.host_ptr_;
        bytes_ = other.bytes_;
        other.host_ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

torch::Tensor PinnedBuffer::to_tensor(torch::ScalarType dtype, const std::vector<int64_t>& shape) const {
    auto options = torch::TensorOptions()
                      .dtype(dtype)
                      .device(torch::kCPU)
                      .pinned_memory(true);

    // Create PyTorch tensor wrapping the raw cudaHostAlloc pointer without copying
    return torch::from_blob(host_ptr_, shape, options);
}


// AsyncTransferManager Implementation

AsyncTransferManager::AsyncTransferManager(int device_id)
    : device_id_(device_id),
      stream_(c10::cuda::getStreamFromPool(/*isHighPriority=*/true, static_cast<int16_t>(device_id))) {}

void AsyncTransferManager::copy_host_to_device_async(const torch::Tensor& src, torch::Tensor& dst) {
    c10::cuda::CUDAGuard device_guard(static_cast<int16_t>(device_id_));

    if (!src.is_cpu()) {
        throw std::invalid_argument("AsyncTransferManager Error: Source tensor must reside on CPU.");
    }
    if (!dst.is_cuda()) {
        throw std::invalid_argument("AsyncTransferManager Error: Destination tensor must reside on CUDA Device.");
    }
    if (src.nbytes() != dst.nbytes()) {
        throw std::invalid_argument("AsyncTransferManager Error: Source and Destination byte sizes must match.");
    }

    // Set target stream for asynchronous execution
    c10::cuda::CUDAStreamGuard stream_guard(stream_);

    cudaError_t err = cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        cudaMemcpyHostToDevice,
        stream_.stream()
    );

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("AsyncTransferManager Error: H2D cudaMemcpyAsync failed: ") + cudaGetErrorString(err)
        );
    }
}

void AsyncTransferManager::copy_device_to_host_async(const torch::Tensor& src, torch::Tensor& dst) {
    c10::cuda::CUDAGuard device_guard(static_cast<int16_t>(device_id_));

    if (!src.is_cuda()) {
        throw std::invalid_argument("AsyncTransferManager Error: Source tensor must reside on CUDA Device.");
    }
    if (!dst.is_cpu()) {
        throw std::invalid_argument("AsyncTransferManager Error: Destination tensor must reside on CPU.");
    }
    if (src.nbytes() != dst.nbytes()) {
        throw std::invalid_argument("AsyncTransferManager Error: Source and Destination byte sizes must match.");
    }

    c10::cuda::CUDAStreamGuard stream_guard(stream_);

    cudaError_t err = cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        cudaMemcpyDeviceToHost,
        stream_.stream()
    );

    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("AsyncTransferManager Error: D2H cudaMemcpyAsync failed: ") + cudaGetErrorString(err)
        );
    }
}

void AsyncTransferManager::synchronize_stream() {
    c10::cuda::CUDAGuard device_guard(static_cast<int16_t>(device_id_));
    stream_.synchronize();
}

} // namespace nexuscache