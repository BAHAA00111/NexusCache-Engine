#include <gtest/gtest.h>
#include "nexuscache/block_manager.hpp"
#include "nexuscache/page_table.hpp"

TEST(PageTableTest, DynamicPageAllocationAndReclaim) {
    nexuscache::BlockAllocatorConfig config;
    config.num_blocks = 10;
    config.block_size = 16;
    config.num_layers = 16;
    config.num_heads = 4;
    config.head_dim = 64;
    config.dtype = torch::kFloat16;
    config.device_id = 0;

    auto block_manager = std::make_shared<nexuscache::BlockManager>(config);
    nexuscache::PageTable page_table(block_manager, config.block_size);

    int64_t seq_id = 1001;
    page_table.register_sequence(seq_id);
    EXPECT_TRUE(page_table.has_sequence(seq_id));

    // Append 32 tokens -> Should trigger exactly 2 physical block allocations (32 / 16)
    auto slots = page_table.append_tokens(seq_id, 32);
    EXPECT_EQ(slots.size(), 32);
    EXPECT_EQ(page_table.get_block_table(seq_id).size(), 2);
    EXPECT_EQ(block_manager->get_num_free_blocks(), 8);

    // Append 1 more token -> Crosses boundary into 3rd physical block
    page_table.append_tokens(seq_id, 1);
    EXPECT_EQ(page_table.get_block_table(seq_id).size(), 3);
    EXPECT_EQ(block_manager->get_num_free_blocks(), 7);

    // Terminate sequence and reclaim physical VRAM blocks
    page_table.free_sequence(seq_id);
    EXPECT_FALSE(page_table.has_sequence(seq_id));
    EXPECT_EQ(block_manager->get_num_free_blocks(), 10);
}