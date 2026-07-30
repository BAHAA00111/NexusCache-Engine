#ifndef NEXUSCACHE_MEMORY_UTILS_HPP
#define NEXUSCACHE_MEMORY_UTILS_HPP

#include <torch/extension.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace nexuscache {

/**
 * Helper utility to extract raw CUDA memory device pointer addresses (uintptr_t)
 * from PyTorch CUDA Tensors for direct kernel invocation.
 */
inline uintptr_t get_tensor_device_ptr(const torch::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        throw std::invalid_argument("get_tensor_device_ptr Error: Tensor must reside on CUDA device.");
    }
    return reinterpret_cast<uintptr_t>(tensor.data_ptr());
}

/**
 * Constructs a PyTorch CUDA Tensor wrapping an arbitrary raw CUDA device pointer (uintptr_t).
 */
inline torch::Tensor make_tensor_from_device_ptr(uintptr_t ptr,
                                                 const std::vector<int64_t>& shape,
                                                 torch::ScalarType dtype,
                                                 int device_id = 0) {
    auto options = torch::TensorOptions()
                      .dtype(dtype)
                      .device(torch::kCUDA, static_cast<int16_t>(device_id))
                      .requires_grad(false);

    return torch::from_blob(reinterpret_cast<void*>(ptr), shape, options);
}

} // namespace nexuscache

#endif // NEXUSCACHE_MEMORY_UTILS_HPP