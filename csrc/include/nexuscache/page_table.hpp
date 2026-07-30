#ifndef NEXUSCACHE_PAGE_TABLE_HPP
#define NEXUSCACHE_PAGE_TABLE_HPP

#include "nexuscache/block_manager.hpp"
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdint>
#include <stdexcept>


namespace nexuscache {

//Metadata for a single request sequence inside the engine.
struct SequenceContext {
    int64_t sequence_id;
    int64_t current_num_tokens{0};
    std::vector<int64_t> physical_block_ids;
};


 //Dynamic Logical-to-Physical Page Table for Paged KV-Cache Management.
 //Tracks logical sequence growth and maps token logical positions to physical VRAM block offsets.

class PageTable {
public:
    explicit PageTable(std::shared_ptr<BlockManager> block_manager, int64_t block_size);
    ~PageTable() = default;

    // Prevent copy/move assignment with raw state
    PageTable(const PageTable&) = delete;
    PageTable& operator=(const PageTable&) = delete;


    void register_sequence(int64_t sequence_id);


    void free_sequence(int64_t sequence_id);


    std::vector<int64_t> append_tokens(int64_t sequence_id, int64_t num_tokens = 1);

    std::vector<int64_t> get_block_table(int64_t sequence_id) const;

    torch::Tensor get_block_table_tensor(const std::vector<int64_t>& sequence_ids, 
                                         torch::Device device = torch::kCUDA) const;

    torch::Tensor get_slot_mapping_tensor(const std::vector<int64_t>& sequence_ids,
                                          const std::vector<int64_t>& query_lens,
                                          torch::Device device = torch::kCUDA) const;
    // Queries
    bool has_sequence(int64_t sequence_id) const;
    int64_t get_sequence_length(int64_t sequence_id) const;
    size_t get_num_active_sequences() const;

private:
    std::shared_ptr<BlockManager> block_manager_;
    int64_t block_size_;

    // Thread-safe map tracking active requests
    std::unordered_map<int64_t, SequenceContext> table_;
    mutable std::mutex table_mutex_;
};

} // namespace nexuscache

#endif // NEXUSCACHE_PAGE_TABLE_HPP