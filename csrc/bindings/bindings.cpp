#include <torch/extension.h>
#include <cuda_runtime.h>
#include "nexuscache/block_manager.hpp"
#include "nexuscache/page_table.hpp"
#include "nexuscache/pinned_memory.hpp"
#include "nexuscache/memory_utils.hpp"
#include "nexuscache/kernels.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "NexusCache C++/CUDA Dynamic Paged KV-Cache Native Subsystem Engine";

    // 1. Memory Utilities & Pointer Interfacing Functions
    m.def("get_tensor_device_ptr", &nexuscache::get_tensor_device_ptr,
          py::arg("tensor"),
          "Extracts raw CUDA memory device pointer address (uintptr_t) from PyTorch Tensor.");

    m.def("make_tensor_from_device_ptr", &nexuscache::make_tensor_from_device_ptr,
          py::arg("ptr"), py::arg("shape"), py::arg("dtype"), py::arg("device_id") = 0,
          "Creates a PyTorch CUDA Tensor wrapping a raw device memory address (uintptr_t).");


    // 2. BlockAllocatorConfig & BlockManager Bindings
    py::class_<nexuscache::BlockAllocatorConfig>(m, "BlockAllocatorConfig")
        .def(py::init<>())
        .def_readwrite("num_blocks", &nexuscache::BlockAllocatorConfig::num_blocks)
        .def_readwrite("block_size", &nexuscache::BlockAllocatorConfig::block_size)
        .def_readwrite("num_layers", &nexuscache::BlockAllocatorConfig::num_layers)
        .def_readwrite("num_heads", &nexuscache::BlockAllocatorConfig::num_heads)
        .def_readwrite("head_dim", &nexuscache::BlockAllocatorConfig::head_dim)
        .def_readwrite("dtype", &nexuscache::BlockAllocatorConfig::dtype)
        .def_readwrite("device_id", &nexuscache::BlockAllocatorConfig::device_id);

    py::class_<nexuscache::BlockManager, std::shared_ptr<nexuscache::BlockManager>>(m, "BlockManager")
        .def(py::init<const nexuscache::BlockAllocatorConfig&>(), py::arg("config"))
        .def("allocate_block", &nexuscache::BlockManager::allocate_block,
             "Allocates a single physical VRAM block index in O(1) time.")
        .def("allocate_blocks", &nexuscache::BlockManager::allocate_blocks, py::arg("num_requested"),
             "Allocates multiple physical VRAM block indices in O(1) time.")
        .def("free_block", &nexuscache::BlockManager::free_block, py::arg("block_id"),
             "Reclaims a physical VRAM block index back to the free pool.")
        .def("free_blocks", &nexuscache::BlockManager::free_blocks, py::arg("block_ids"),
             "Reclaims a list of physical VRAM block indices.")
        .def("get_num_free_blocks", &nexuscache::BlockManager::get_num_free_blocks)
        .def("get_num_allocated_blocks", &nexuscache::BlockManager::get_num_allocated_blocks)
        .def("get_total_blocks", &nexuscache::BlockManager::get_total_blocks)
        .def("get_physical_kv_tensors", &nexuscache::BlockManager::get_physical_kv_tensors,
             "Returns tuple of physical (Key, Value) cache pool PyTorch Tensors.")
        .def("get_key_cache_ptr", [](const nexuscache::BlockManager& self) -> uintptr_t {
            auto [key_cache, val_cache] = self.get_physical_kv_tensors();
            return nexuscache::get_tensor_device_ptr(key_cache);
        }, "Returns raw CUDA uintptr_t address for key physical cache pool.")
        .def("get_value_cache_ptr", [](const nexuscache::BlockManager& self) -> uintptr_t {
            auto [key_cache, val_cache] = self.get_physical_kv_tensors();
            return nexuscache::get_tensor_device_ptr(val_cache);
        }, "Returns raw CUDA uintptr_t address for value physical cache pool.");

    // 3. PageTable Bindings (STRING DEVICE CONVERSION PREVENTS SEGFAULT)
    py::class_<nexuscache::PageTable>(m, "PageTable")
        .def(py::init<std::shared_ptr<nexuscache::BlockManager>, int64_t>(),
             py::arg("block_manager"), py::arg("block_size"))
        .def("register_sequence", &nexuscache::PageTable::register_sequence, py::arg("sequence_id"))
        .def("free_sequence", &nexuscache::PageTable::free_sequence, py::arg("sequence_id"))
        .def("append_tokens", &nexuscache::PageTable::append_tokens,
             py::arg("sequence_id"), py::arg("num_tokens") = 1)
        .def("get_block_table", &nexuscache::PageTable::get_block_table, py::arg("sequence_id"))
        .def("get_block_table_tensor", [](nexuscache::PageTable& self, const std::vector<int64_t>& sequence_ids, const std::string& device_str) {
            c10::Device device(device_str);
            return self.get_block_table_tensor(sequence_ids, device);
        }, py::arg("sequence_ids"), py::arg("device") = "cuda")
        .def("get_slot_mapping_tensor", [](nexuscache::PageTable& self, const std::vector<int64_t>& sequence_ids, const std::vector<int64_t>& query_lens, const std::string& device_str) {
            c10::Device device(device_str);
            return self.get_slot_mapping_tensor(sequence_ids, query_lens, device);
        }, py::arg("sequence_ids"), py::arg("query_lens"), py::arg("device") = "cuda")
        .def("has_sequence", &nexuscache::PageTable::has_sequence, py::arg("sequence_id"))
        .def("get_sequence_length", &nexuscache::PageTable::get_sequence_length, py::arg("sequence_id"))
        .def("get_num_active_sequences", &nexuscache::PageTable::get_num_active_sequences);


    // 4. Zero-Copy Pinned Host Memory & Async Stream Bindings
    py::class_<nexuscache::PinnedBuffer>(m, "PinnedBuffer")
        .def(py::init<size_t, unsigned int>(), py::arg("bytes"), py::arg("flags") = 0)
        .def("to_tensor", &nexuscache::PinnedBuffer::to_tensor, py::arg("dtype"), py::arg("shape"))
        .def("size_bytes", &nexuscache::PinnedBuffer::size_bytes);

    py::class_<nexuscache::AsyncTransferManager>(m, "AsyncTransferManager")
        .def(py::init<int>(), py::arg("device_id") = 0)
        .def("copy_host_to_device_async", &nexuscache::AsyncTransferManager::copy_host_to_device_async)
        .def("copy_device_to_host_async", &nexuscache::AsyncTransferManager::copy_device_to_host_async)
        .def("synchronize_stream", &nexuscache::AsyncTransferManager::synchronize_stream);

    
    // 5. Explicit CUDA Kernel Direct Invocation Bindings
    m.def("launch_copy_blocks", [](
        uintptr_t key_cache_ptr,
        uintptr_t value_cache_ptr,
        uintptr_t out_key_ptr,
        uintptr_t out_value_ptr,
        const std::vector<std::pair<int32_t, int32_t>>& raw_mappings,
        int num_heads,
        int head_dim,
        int block_size,
        int elem_size_bytes,
        uintptr_t stream_ptr
    ) {
        std::vector<nexuscache::BlockMapping> mappings;
        mappings.reserve(raw_mappings.size());
        for (const auto& p : raw_mappings) {
            mappings.push_back({p.first, p.second});
        }

        nexuscache::launch_copy_blocks_kernel(
            reinterpret_cast<const void*>(key_cache_ptr),
            reinterpret_cast<const void*>(value_cache_ptr),
            reinterpret_cast<void*>(out_key_ptr),
            reinterpret_cast<void*>(out_value_ptr),
            mappings.data(),
            static_cast<int>(mappings.size()),
            num_heads,
            head_dim,
            block_size,
            elem_size_bytes,
            reinterpret_cast<cudaStream_t>(stream_ptr)
        );
    }, py::arg("key_cache_ptr"), py::arg("value_cache_ptr"),
       py::arg("out_key_ptr"), py::arg("out_value_ptr"),
       py::arg("mappings"), py::arg("num_heads"), py::arg("head_dim"),
       py::arg("block_size"), py::arg("elem_size_bytes"), py::arg("stream_ptr") = 0);

    m.def("launch_memset_blocks", [](
        uintptr_t cache_ptr,
        const std::vector<int32_t>& block_indices,
        int block_bytes,
        uintptr_t stream_ptr
    ) {
        nexuscache::launch_memset_blocks_kernel(
            reinterpret_cast<void*>(cache_ptr),
            block_indices.data(),
            static_cast<int>(block_indices.size()),
            block_bytes,
            reinterpret_cast<cudaStream_t>(stream_ptr)
        );
    }, py::arg("cache_ptr"), py::arg("block_indices"),
       py::arg("block_bytes"), py::arg("stream_ptr") = 0);

    // 6. Paged Attention Decoding Kernel Binding
    m.def("launch_paged_attention", [](
        uintptr_t out_ptr,
        uintptr_t query_ptr,
        uintptr_t key_cache_ptr,
        uintptr_t value_cache_ptr,
        const std::vector<int32_t>& block_tables,
        const std::vector<int32_t>& seq_lens,
        int max_num_blocks_per_seq,
        int num_seqs,
        int num_heads,
        int head_dim,
        int block_size,
        float scale,
        uintptr_t stream_ptr
    ) {
        nexuscache::launch_paged_attention_kernel(
            reinterpret_cast<void*>(out_ptr),
            reinterpret_cast<const void*>(query_ptr),
            reinterpret_cast<const void*>(key_cache_ptr),
            reinterpret_cast<const void*>(value_cache_ptr),
            block_tables.data(),
            seq_lens.data(),
            max_num_blocks_per_seq,
            num_seqs,
            num_heads,
            head_dim,
            block_size,
            scale,
            reinterpret_cast<cudaStream_t>(stream_ptr)
        );
    }, py::arg("out_ptr"), py::arg("query_ptr"), py::arg("key_cache_ptr"),
       py::arg("value_cache_ptr"), py::arg("block_tables"), py::arg("seq_lens"),
       py::arg("max_num_blocks_per_seq"), py::arg("num_seqs"), py::arg("num_heads"),
       py::arg("head_dim"), py::arg("block_size"), py::arg("scale"), py::arg("stream_ptr") = 0);

}