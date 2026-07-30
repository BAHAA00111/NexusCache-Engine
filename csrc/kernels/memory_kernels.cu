#include "nexuscache/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace nexuscache {

// -----------------------------------------------------------------------------
// CUDA Kernel: Vectorized 128-bit Block Copy
// -----------------------------------------------------------------------------
__global__ void copy_blocks_kernel(
    const int4* __restrict__ key_cache,
    const int4* __restrict__ value_cache,
    int4* __restrict__ out_key,
    int4* __restrict__ out_value,
    const BlockMapping* __restrict__ block_mappings,
    const int block_bytes_vec
) {
    const int mapping_idx = blockIdx.y;
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx >= block_bytes_vec) return;

    BlockMapping mapping = block_mappings[mapping_idx];
    int32_t src_block = mapping.physical_block_idx;
    int32_t dst_block = mapping.logical_block_idx;

    size_t src_offset = static_cast<size_t>(src_block) * block_bytes_vec + vec_idx;
    size_t dst_offset = static_cast<size_t>(dst_block) * block_bytes_vec + vec_idx;

    // Vectorized 16-byte transfer
    out_key[dst_offset] = key_cache[src_offset];
    out_value[dst_offset] = value_cache[src_offset];
}

// -----------------------------------------------------------------------------
// CUDA Kernel: Vectorized Block Zeroing (Memset)
// -----------------------------------------------------------------------------
__global__ void memset_blocks_kernel(
    int4* __restrict__ cache_ptr,
    const int32_t* __restrict__ block_indices,
    const int block_bytes_vec
) {
    const int block_idx_pos = blockIdx.y;
    const int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx >= block_bytes_vec) return;

    int32_t target_block = block_indices[block_idx_pos];
    size_t offset = static_cast<size_t>(target_block) * block_bytes_vec + vec_idx;

    const int4 zero_vec = make_int4(0, 0, 0, 0);
    cache_ptr[offset] = zero_vec;
}

// -----------------------------------------------------------------------------
// C-Linkage Host Launch Wrappers (With Device Memory Allocation)
// -----------------------------------------------------------------------------
extern "C" {

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
    cudaStream_t stream
) {
    if (num_mappings <= 0) return;

    // 1. Allocate device memory for block mappings array
    BlockMapping* d_mappings = nullptr;
    size_t mappings_bytes = num_mappings * sizeof(BlockMapping);
    CUDA_CHECK(cudaMallocAsync(&d_mappings, mappings_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_mappings, block_mappings, mappings_bytes, cudaMemcpyHostToDevice, stream));

    // 2. Calculate vector dimensions
    size_t total_bytes_per_block = static_cast<size_t>(num_heads) * head_dim * block_size * elem_size_bytes;
    int block_bytes_vec = static_cast<int>(total_bytes_per_block / sizeof(int4));

    constexpr int threads_per_block = 256;
    dim3 grid_dim((block_bytes_vec + threads_per_block - 1) / threads_per_block, num_mappings);

    // 3. Launch kernel with device pointer
    copy_blocks_kernel<<<grid_dim, threads_per_block, 0, stream>>>(
        reinterpret_cast<const int4*>(key_cache_ptr),
        reinterpret_cast<const int4*>(value_cache_ptr),
        reinterpret_cast<int4*>(out_key_buf),
        reinterpret_cast<int4*>(out_value_buf),
        d_mappings,
        block_bytes_vec
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Async free mapping buffer on stream
    CUDA_CHECK(cudaFreeAsync(d_mappings, stream));
}

void launch_memset_blocks_kernel(
    void* cache_ptr,
    const int32_t* block_indices,
    int num_blocks,
    int block_bytes,
    cudaStream_t stream
) {
    if (num_blocks <= 0) return;

    // 1. Allocate device memory for block indices array
    int32_t* d_block_indices = nullptr;
    size_t indices_bytes = num_blocks * sizeof(int32_t);
    CUDA_CHECK(cudaMallocAsync(&d_block_indices, indices_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_block_indices, block_indices, indices_bytes, cudaMemcpyHostToDevice, stream));

    // 2. Calculate vector dimensions
    int block_bytes_vec = block_bytes / sizeof(int4);
    constexpr int threads_per_block = 256;
    dim3 grid_dim((block_bytes_vec + threads_per_block - 1) / threads_per_block, num_blocks);

    // 3. Launch kernel with device pointer
    memset_blocks_kernel<<<grid_dim, threads_per_block, 0, stream>>>(
        reinterpret_cast<int4*>(cache_ptr),
        d_block_indices,
        block_bytes_vec
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Async free indices buffer on stream
    CUDA_CHECK(cudaFreeAsync(d_block_indices, stream));
}

} // extern "C"
} // namespace nexuscache