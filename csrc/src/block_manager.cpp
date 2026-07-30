#include "nexuscache/block_manager.hpp"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

namespace nexuscache {

BlockManager::BlockManager(const BlockAllocatorConfig& config)
    : config_(config) {
    
    // Ensure execution happens on the targeted CUDA device
    c10::cuda::CUDAGuard device_guard(static_cast<int16_t>(config_.device_id));

    if (config_.num_blocks <= 0 || config_.block_size <= 0) {
        throw std::invalid_argument("BlockManager: num_blocks and block_size must be positive.");
    }

    // Configure Torch Tensor Options for direct CUDA Allocation
    auto tensor_options = torch::TensorOptions()
                              .dtype(config_.dtype)
                              .device(torch::kCUDA, static_cast<int16_t>(config_.device_id))
                              .requires_grad(false);

    // Compute Physical Pool Shape
    // Shape: [num_blocks, num_layers, num_heads, block_size, head_dim]
    std::vector<int64_t> pool_shape = {
        config_.num_blocks,
        config_.num_layers,
        config_.num_heads,
        config_.block_size,
        config_.head_dim
    };

    try {
        // Allocate contiguous global physical memory blocks directly on GPU VRAM
        key_cache_pool_ = torch::empty(pool_shape, tensor_options);
        value_cache_pool_ = torch::empty(pool_shape, tensor_options);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            std::string("BlockManager: Out of Memory during physical VRAM pool pre-allocation: ") + e.what()
        );
    }

    // Initialize the free physical block stack with all available indices [0, num_blocks - 1]
    free_block_stack_.reserve(config_.num_blocks);
    for (int64_t i = config_.num_blocks - 1; i >= 0; --i) {
        free_block_stack_.push_back(i);
    }

    std::cout << "[NexusCache] Physical Block Allocator Initialized: "
              << config_.num_blocks << " blocks pre-allocated ("
              << (key_cache_pool_.nbytes() + value_cache_pool_.nbytes()) / (1024 * 1024)
              << " MB VRAM Pool)." << std::endl;
}

BlockManager::~BlockManager() {
    // Torch Tensor RAII automatically frees physical VRAM on destruction
    free_block_stack_.clear();
}

int64_t BlockManager::allocate_block() {
    std::lock_guard<std::mutex> lock(stack_mutex_);
    
    if (free_block_stack_.empty()) {
        throw std::runtime_error("BlockManager Error: Physical VRAM Pool Exhausted! No free blocks available.");
    }

    int64_t block_id = free_block_stack_.back();
    free_block_stack_.pop_back();
    return block_id;
}

std::vector<int64_t> BlockManager::allocate_blocks(size_t num_requested) {
    std::lock_guard<std::mutex> lock(stack_mutex_);

    if (free_block_stack_.size() < num_requested) {
        throw std::runtime_error(
            "BlockManager Error: Cannot allocate " + std::to_string(num_requested) + 
            " blocks. Only " + std::to_string(free_block_stack_.size()) + " free blocks remaining."
        );
    }

    std::vector<int64_t> allocated_ids;
    allocated_ids.reserve(num_requested);

    for (size_t i = 0; i < num_requested; ++i) {
        allocated_ids.push_back(free_block_stack_.back());
        free_block_stack_.pop_back();
    }

    return allocated_ids;
}

void BlockManager::free_block(int64_t block_id) {
    std::lock_guard<std::mutex> lock(stack_mutex_);

    if (block_id < 0 || block_id >= config_.num_blocks) {
        throw std::out_of_range("BlockManager Error: Attempted to free invalid block_id: " + std::to_string(block_id));
    }

    free_block_stack_.push_back(block_id);
}

void BlockManager::free_blocks(const std::vector<int64_t>& block_ids) {
    std::lock_guard<std::mutex> lock(stack_mutex_);

    for (int64_t block_id : block_ids) {
        if (block_id < 0 || block_id >= config_.num_blocks) {
            throw std::out_of_range("BlockManager Error: Attempted to free invalid block_id: " + std::to_string(block_id));
        }
        free_block_stack_.push_back(block_id);
    }
}

int64_t BlockManager::get_num_free_blocks() const {
    std::lock_guard<std::mutex> lock(stack_mutex_);
    return static_cast<int64_t>(free_block_stack_.size());
}

int64_t BlockManager::get_num_allocated_blocks() const {
    std::lock_guard<std::mutex> lock(stack_mutex_);
    return config_.num_blocks - static_cast<int64_t>(free_block_stack_.size());
}

int64_t BlockManager::get_total_blocks() const {
    return config_.num_blocks;
}

std::pair<torch::Tensor, torch::Tensor> BlockManager::get_physical_kv_tensors() const {
    return {key_cache_pool_, value_cache_pool_};
}

} // namespace nexuscache