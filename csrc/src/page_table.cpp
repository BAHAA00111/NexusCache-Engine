#include "nexuscache/page_table.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace nexuscache {

PageTable::PageTable(std::shared_ptr<BlockManager> block_manager, int64_t block_size)
    : block_manager_(std::move(block_manager)), block_size_(block_size) {
    if (!block_manager_) {
        throw std::invalid_argument("PageTable initialized with null BlockManager pointer.");
    }
    if (block_size_ <= 0) {
        throw std::invalid_argument("PageTable block_size must be positive.");
    }
}

void PageTable::register_sequence(int64_t sequence_id) {
    std::lock_guard<std::mutex> lock(table_mutex_);
    if (table_.find(sequence_id) != table_.end()) {
        throw std::runtime_error("Sequence ID " + std::to_string(sequence_id) + " already registered.");
    }
    
    SequenceContext ctx;
    ctx.sequence_id = sequence_id;
    ctx.current_num_tokens = 0;
    table_[sequence_id] = std::move(ctx);
}

void PageTable::free_sequence(int64_t sequence_id) {
    std::vector<int64_t> blocks_to_free;
    {
        std::lock_guard<std::mutex> lock(table_mutex_);
        auto it = table_.find(sequence_id);
        if (it == table_.end()) {
            return; // Idempotent cleanup
        }
        blocks_to_free = std::move(it->second.physical_block_ids);
        table_.erase(it);
    }

    // Free physical VRAM blocks back to the allocator
    if (!blocks_to_free.empty()) {
        block_manager_->free_blocks(blocks_to_free);
    }
}

std::vector<int64_t> PageTable::append_tokens(int64_t sequence_id, int64_t num_tokens) {
    if (num_tokens <= 0) {
        throw std::invalid_argument("num_tokens must be greater than zero.");
    }

    std::vector<int64_t> newly_allocated_blocks;
    {
        std::lock_guard<std::mutex> lock(table_mutex_);
        auto it = table_.find(sequence_id);
        if (it == table_.end()) {
            throw std::runtime_error("Sequence ID " + std::to_string(sequence_id) + " not registered.");
        }

        SequenceContext& ctx = it->second;
        int64_t prev_tokens = ctx.current_num_tokens;
        int64_t new_tokens = prev_tokens + num_tokens;

        int64_t prev_blocks = (prev_tokens + block_size_ - 1) / block_size_;
        int64_t required_blocks = (new_tokens + block_size_ - 1) / block_size_;
        int64_t blocks_needed = required_blocks - prev_blocks;

        if (blocks_needed > 0) {
            newly_allocated_blocks = block_manager_->allocate_blocks(blocks_needed);
            ctx.physical_block_ids.insert(
                ctx.physical_block_ids.end(),
                newly_allocated_blocks.begin(),
                newly_allocated_blocks.end()
            );
        }

        ctx.current_num_tokens = new_tokens;
    }

    return newly_allocated_blocks;
}

std::vector<int64_t> PageTable::get_block_table(int64_t sequence_id) const {
    std::lock_guard<std::mutex> lock(table_mutex_);
    auto it = table_.find(sequence_id);
    if (it == table_.end()) {
        throw std::runtime_error("Sequence ID " + std::to_string(sequence_id) + " not found.");
    }
    return it->second.physical_block_ids;
}

torch::Tensor PageTable::get_block_table_tensor(
    const std::vector<int64_t>& sequence_ids,
    torch::Device device
) const {
    std::lock_guard<std::mutex> lock(table_mutex_);

    int64_t batch_size = static_cast<int64_t>(sequence_ids.size());
    int64_t max_block_cols = 0;

    // First pass: Find max allocated block count in batch for padding
    for (int64_t seq_id : sequence_ids) {
        auto it = table_.find(seq_id);
        if (it != table_.end()) {
            max_block_cols = std::max(max_block_cols, static_cast<int64_t>(it->second.physical_block_ids.size()));
        }
    }

    // Prepare CPU host matrix with -1 padding
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor block_table_tensor = torch::full({batch_size, max_block_cols}, -1, options);
    auto accessor = block_table_tensor.accessor<int32_t, 2>();

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t seq_id = sequence_ids[b];
        auto it = table_.find(seq_id);
        if (it != table_.end()) {
            const auto& blocks = it->second.physical_block_ids;
            for (size_t i = 0; i < blocks.size(); ++i) {
                accessor[b][i] = static_cast<int32_t>(blocks[i]);
            }
        }
    }

    return block_table_tensor.to(device, /*non_blocking=*/true);
}

torch::Tensor PageTable::get_slot_mapping_tensor(
    const std::vector<int64_t>& sequence_ids,
    const std::vector<int64_t>& query_lens,
    torch::Device device
) const {
    if (sequence_ids.size() != query_lens.size()) {
        throw std::invalid_argument("sequence_ids and query_lens must have matching lengths.");
    }

    std::lock_guard<std::mutex> lock(table_mutex_);

    int64_t total_queries = 0;
    for (int64_t len : query_lens) {
        total_queries += len;
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor slot_mapping = torch::empty({total_queries}, options);
    int64_t* ptr = slot_mapping.data_ptr<int64_t>();

    int64_t flat_idx = 0;
    for (size_t b = 0; b < sequence_ids.size(); ++b) {
        int64_t seq_id = sequence_ids[b];
        int64_t q_len = query_lens[b];

        auto it = table_.find(seq_id);
        if (it == table_.end()) {
            throw std::runtime_error("Sequence ID " + std::to_string(seq_id) + " not found.");
        }

        const SequenceContext& ctx = it->second;
        int64_t cur_tokens = ctx.current_num_tokens;
        int64_t start_pos = cur_tokens - q_len;

        for (int64_t i = 0; i < q_len; ++i) {
            int64_t token_pos = start_pos + i;
            int64_t block_idx = token_pos / block_size_;
            int64_t block_offset = token_pos % block_size_;

            int64_t physical_block_id = ctx.physical_block_ids.at(block_idx);
            int64_t slot = (physical_block_id * block_size_) + block_offset;
            ptr[flat_idx++] = slot;
        }
    }

    return slot_mapping.to(device, /*non_blocking=*/true);
}

bool PageTable::has_sequence(int64_t sequence_id) const {
    std::lock_guard<std::mutex> lock(table_mutex_);
    return table_.find(sequence_id) != table_.end();
}

int64_t PageTable::get_sequence_length(int64_t sequence_id) const {
    std::lock_guard<std::mutex> lock(table_mutex_);
    auto it = table_.find(sequence_id);
    if (it == table_.end()) {
        return 0;
    }
    return it->second.current_num_tokens;
}

size_t PageTable::get_num_active_sequences() const {
    std::lock_guard<std::mutex> lock(table_mutex_);
    return table_.size();
}

} // namespace nexuscache