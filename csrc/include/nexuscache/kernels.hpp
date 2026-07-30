#ifndef NEXUSCACHE_KERNELS_HPP_
#define NEXUSCACHE_KERNELS_HPP_

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexuscache {

// Standard CUDA runtime safety check
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                         \
                std::string("CUDA Error [") + cudaGetErrorString(err) +       \
                "] at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                     \
    } while (0)

struct BlockMapping {
    int32_t logical_block_idx;
    int32_t physical_block_idx;
};

extern "C" {

/**
 * Copies non-contiguous physical KV-Cache blocks into contiguous target buffers.
 */
void launch_copy_blocks_kernel(
    const void* key_cache_ptr,
    const void* value_cache_ptr,
    void* out_key_buf,
    void* out_value_buf,
    const BlockMapping* block_mappings,
    int num_mappings,
    int num_heads,
    int head_dim,
    int block_size,
    int elem_size_bytes,
    cudaStream_t stream = nullptr
);

/**
 * Zeroes out specific physical cache block indices in VRAM asynchronously.
 */
void launch_memset_blocks_kernel(
    void* cache_ptr,
    const int32_t* block_indices,
    int num_blocks,
    int block_bytes,
    cudaStream_t stream = nullptr
);

/**
 * Executes online Softmax Paged Attention decoding over non-contiguous KV-Cache blocks.
 */
void launch_paged_attention_kernel(
    void* out_ptr,
    const void* query_ptr,
    const void* key_cache_ptr,
    const void* value_cache_ptr,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int max_num_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int head_dim,
    int block_size,
    float scale,
    cudaStream_t stream = nullptr
);

} // extern "C"
} // namespace nexuscache

#endif // NEXUSCACHE_KERNELS_HPP_