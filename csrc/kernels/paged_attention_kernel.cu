#include "nexuscache/kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

namespace nexuscache {

__global__ void paged_attention_decoding_kernel(
    __half* __restrict__ out,                     // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ query,             // [num_seqs, num_heads, head_dim]
    const __half* __restrict__ key_cache,         // [num_blocks, num_heads, block_size, head_dim]
    const __half* __restrict__ value_cache,       // [num_blocks, num_heads, block_size, head_dim]
    const int32_t* __restrict__ block_tables,     // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,         // [num_seqs]
    const int max_blocks_per_seq,
    const int num_heads,
    const int head_dim,
    const int block_size,
    const float scale
) {
    const int seq_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len <= 0) return;

    // Linear query offset: [seq_idx, head_idx, :]
    size_t q_offset = (static_cast<size_t>(seq_idx) * num_heads + head_idx) * head_dim;
    const __half* q_vec = query + q_offset;
    __half* out_vec = out + q_offset;

    extern __shared__ char shared_mem[];
    float* s_query = reinterpret_cast<float*>(shared_mem);
    float* s_accum = s_query + head_dim;

    // Load query into shared memory and initialize accumulators
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_query[d] = __half2float(q_vec[d]);
        s_accum[d] = 0.0f;
    }
    __syncthreads();

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    const int num_blocks_seq = (seq_len + block_size - 1) / block_size;
    const int32_t* seq_block_table = block_tables + (static_cast<size_t>(seq_idx) * max_blocks_per_seq);

    for (int b_idx = 0; b_idx < num_blocks_seq; ++b_idx) {
        int32_t physical_block_id = seq_block_table[b_idx];
        if (physical_block_id < 0) break;

        int tok_start = b_idx * block_size;
        int tok_end = min(tok_start + block_size, seq_len);

        for (int tok = tok_start; tok < tok_end; ++tok) {
            int block_offset = tok % block_size;
            
            // Layout: [num_blocks, num_heads, block_size, head_dim]
            size_t kv_offset = ((((static_cast<size_t>(physical_block_id) * num_heads + head_idx) 
                                  * block_size + block_offset) * head_dim));
            
            const __half* k_tok = key_cache + kv_offset;
            const __half* v_tok = value_cache + kv_offset;

            // 1. Q * K^T Dot Product
            float qk_score = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                qk_score += s_query[d] * __half2float(k_tok[d]);
            }

            // Parallel Warp Reduction across 32 threads
            #pragma unroll
            for (int mask = 16; mask > 0; mask /= 2) {
                qk_score += __shfl_xor_sync(0xffffffff, qk_score, mask);
            }

            qk_score *= scale;

            // 2. Online Softmax Rescale Across All Threads
            float new_max = fmaxf(max_score, qk_score);
            float alpha = expf(max_score - new_max);
            float exp_score = expf(qk_score - new_max);

            max_score = new_max;
            sum_exp = sum_exp * alpha + exp_score;

            // 3. Parallel Rescale & Accumulation of V vector
            for (int d = tid; d < head_dim; d += blockDim.x) {
                s_accum[d] = s_accum[d] * alpha + exp_score * __half2float(v_tok[d]);
            }
            __syncthreads();
        }
    }

    // Write final normalized result
    if (sum_exp > 0.0f) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out_vec[d] = __float2half(s_accum[d] / sum_exp);
        }
    }
}

extern "C" {

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
    cudaStream_t stream
) {
    if (num_seqs <= 0) return;

    int32_t* d_block_tables = nullptr;
    int32_t* d_seq_lens = nullptr;

    size_t bt_bytes = static_cast<size_t>(num_seqs) * max_num_blocks_per_seq * sizeof(int32_t);
    size_t sl_bytes = static_cast<size_t>(num_seqs) * sizeof(int32_t);

    CUDA_CHECK(cudaMallocAsync(&d_block_tables, bt_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&d_seq_lens, sl_bytes, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_block_tables, block_tables, bt_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens, seq_lens, sl_bytes, cudaMemcpyHostToDevice, stream));

    dim3 grid(num_seqs * num_heads);
    dim3 block(32); // 1 warp per head

    size_t shared_mem_bytes = (2 * head_dim) * sizeof(float);

    paged_attention_decoding_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        reinterpret_cast<__half*>(out_ptr),
        reinterpret_cast<const __half*>(query_ptr),
        reinterpret_cast<const __half*>(key_cache_ptr),
        reinterpret_cast<const __half*>(value_cache_ptr),
        d_block_tables,
        d_seq_lens,
        max_num_blocks_per_seq,
        num_heads,
        head_dim,
        block_size,
        scale
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_block_tables, stream));
    CUDA_CHECK(cudaFreeAsync(d_seq_lens, stream));
}

} // extern "C"
} // namespace nexuscache