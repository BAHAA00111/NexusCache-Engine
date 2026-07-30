#include <gtest/gtest.h>
#include "nexuscache/block_manager.hpp"

TEST(BlockManagerTest, AllocationAndFreeCycle) {
    nexuscache::BlockAllocatorConfig config;
    config.num_blocks = 100;
    config.block_size = 16;
    config.num_layers = 32;
    config.num_heads = 8;
    config.head_dim = 128;
    config.dtype = torch::kFloat16;
    config.device_id = 0;

    nexuscache::BlockManager manager(config);

    EXPECT_EQ(manager.get_total_blocks(), 100);
    EXPECT_EQ(manager.get_num_free_blocks(), 100);

    // Test Single Allocation
    int64_t b1 = manager.allocate_block();
    EXPECT_EQ(manager.get_num_free_blocks(), 99);

    // Test Batch Allocation
    auto blocks = manager.allocate_blocks(10);
    EXPECT_EQ(blocks.size(), 10);
    EXPECT_EQ(manager.get_num_free_blocks(), 89);

    // Test Deallocation
    manager.free_block(b1);
    EXPECT_EQ(manager.get_num_free_blocks(), 90);

    manager.free_blocks(blocks);
    EXPECT_EQ(manager.get_num_free_blocks(), 100);
}