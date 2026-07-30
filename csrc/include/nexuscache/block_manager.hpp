#ifndef NEXUSCACHE_BLOCK_MANAGER_HPP
#define NEXUSCACHE_BLOCK_MANAGER_HPP

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stack>
#include <mutex>
#include <memory>
#include <cstdint>
#include <stdexcept>

namespace nexuscache {


// Configuration parameters for the Physical Block Allocator.
struct BlockAllocatorConfig {
    int64_t num_blocks;       
    int64_t block_size;     
    int64_t num_layers;       
    int64_t num_heads;        
    int64_t head_dim;         
    torch::ScalarType dtype;  
    int device_id;            
};

/**
    Lock-Free / Low-Latency Physical VRAM Block Manager for Paged KV-Cache.
    Manages physical VRAM page allocations using a pre-allocated pool to eliminate
    CUDA driver overhead and VRAM external fragmentation during continuous generation.
 */
class BlockManager {
public:
    explicit BlockManager(const BlockAllocatorConfig& config);
    ~BlockManager();

    // Prevent copying to ensure singular ownership of CUDA VRAM resources
    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;


    int64_t allocate_block();


    std::vector<int64_t> allocate_blocks(size_t num_requested);

    void free_block(int64_t block_id);


    void free_blocks(const std::vector<int64_t>& block_ids);

    // Getters
    int64_t get_num_free_blocks() const;
    int64_t get_num_allocated_blocks() const;
    int64_t get_total_blocks() const;
    

    std::pair<torch::Tensor, torch::Tensor> get_physical_kv_tensors() const;

private:
    BlockAllocatorConfig config_;
    
    // Core physical VRAM storage allocations held in PyTorch C++ Tensors
    // Key Cache Shape:   [num_blocks, num_layers, num_heads, block_size, head_dim]
    // Value Cache Shape: [num_blocks, num_layers, num_heads, block_size, head_dim]
    torch::Tensor key_cache_pool_;
    torch::Tensor value_cache_pool_;

    // Fast O(1) allocation stack tracking free physical block indices
    std::vector<int64_t> free_block_stack_;
    mutable std::mutex stack_mutex_;
};

} // namespace nexuscache

#endif // NEXUSCACHE_BLOCK_MANAGER_HPP