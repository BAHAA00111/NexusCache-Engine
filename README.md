<div align="center">

# ⚡ NexusCache Engine

### Production-Ready Dynamic Paged KV-Cache Runtime for High-Throughput LLM Serving

A high-performance LLM serving infrastructure inspired by **vLLM**, combining a native **C++/CUDA paged KV-cache runtime**, **asynchronous Ray scheduling**, **continuous request batching**, and **analytical workload modeling** to maximize GPU utilization while eliminating VRAM fragmentation.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Ray](https://img.shields.io/badge/Ray-Distributed-028CF0?logo=ray&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

</div>

---

# Overview

Modern LLM serving systems are constrained by GPU memory fragmentation, inefficient request scheduling, and underutilized hardware during concurrent inference.

**NexusCache Engine** addresses these challenges through a production-oriented serving runtime that combines native CUDA memory management, asynchronous distributed scheduling, and analytical workload optimization into a unified inference pipeline.

Rather than focusing solely on model execution, NexusCache optimizes the infrastructure surrounding inference—GPU memory allocation, continuous batching, request scheduling, and runtime performance—to deliver higher throughput, lower latency, and improved GPU utilization under production workloads.

---

## ✨ Features

- 🚀 Dynamic paged KV-cache with virtual page tables
- ⚡ Native C++17 / CUDA memory management runtime
- 🔄 Continuous asynchronous request batching using Ray Actors
- 📦 Zero-copy pinned host memory allocation
- 🧠 Custom paged attention CUDA kernels
- 📊 VRAM saturation and workload modeling
- 📈 Automated A/B benchmarking and latency analysis
- 🐳 Docker-ready deployment and profiling environment
- ✅ PyTorch-native C++ extensions via PyBind11

---

## Why NexusCache?

Large-scale LLM inference is increasingly limited by systems-level bottlenecks rather than model architecture itself. GPU memory fragmentation, dynamic request patterns, and scheduling overhead reduce serving efficiency long before compute resources are fully utilized.

NexusCache explores these challenges by combining:

- Native CUDA memory management
- Dynamic paged KV-cache allocation
- Distributed asynchronous scheduling
- Queueing theory and workload analytics
- Production-grade benchmarking and telemetry

The result is a serving infrastructure designed to maximize GPU efficiency while remaining modular, extensible, and reproducible.

---

# 📈 Performance

Benchmarks were collected using synthetic and continuous request workloads with GPU telemetry and runtime profiling.

| Metric | Result |
|---------|-------:|
| Peak GPU Utilization | **92%** |
| p95 Response Latency | **<35 ms** |
| Maximum Serving Batch Size | **2.4× Increase** |
| External GPU Fragmentation | **≈0%** |
| Memory Allocation Strategy | **Dynamic Paged KV Cache** |

### Core Optimizations

| Component | Optimization |
|-----------|--------------|
| **Memory Runtime** | Dynamic paged block allocator with virtual page tables |
| **CUDA Kernels** | Custom paged attention and memory management kernels |
| **Request Scheduling** | Ray Actor continuous batching |
| **Host ↔ GPU Transfers** | Zero-copy pinned host memory |
| **Analytics Engine** | Queueing theory, VRAM forecasting, and A/B evaluation |

---

## Table of Contents

- System Architecture
- Repository Structure
- Technology Stack
- Installation
- Quick Start
- Benchmarking
- Deployment
- Roadmap
- Contributing
- License

---
# 🏗️ System Architecture

NexusCache Engine separates high-level serving logic from low-level GPU memory management through a modular layered architecture. Python orchestrates distributed request execution, while performance-critical operations execute inside native C++/CUDA extensions.

```text
                                  NexusCache Engine
┌──────────────────────────────────────────────────────────────────────────────┐
│                  High-Performance LLM Serving Infrastructure                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                     Async Requests / Streaming API
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Python Serving Runtime                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ FastAPI │ Continuous Batching │ Request Queue │ Pipeline Manager │ Metrics │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                    Distributed Scheduling via Ray Actors
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Native Compute Runtime                             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Dynamic Block Allocator │ Virtual Page Table │ Pinned Memory │ CUDA Kernels │
│                 Paged Attention │ Memory Manager │ PyBind11 ABI             │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           NVIDIA GPU Memory (VRAM)
```

---

## Runtime Components

| Layer | Responsibility |
|--------|----------------|
| **Serving Runtime** | FastAPI endpoints, continuous batching, streaming responses, request lifecycle |
| **Ray Scheduler** | Distributed actor execution, asynchronous scheduling, workload balancing |
| **Native Runtime** | Dynamic page allocation, CUDA kernels, pinned memory management |
| **Analytics Engine** | Queueing theory, VRAM forecasting, workload analysis, A/B benchmarking |
| **Benchmark Suite** | Stress testing, latency profiling, GPU telemetry, throughput evaluation |

---

# ⚙️ Core Engineering Concepts

NexusCache Engine is built around several low-level systems optimizations commonly found in production LLM serving infrastructure.

## Dynamic Paged KV-Cache

Instead of allocating large contiguous KV-cache regions, memory is divided into fixed-size pages managed through virtual page tables.

### Benefits

- Eliminates GPU memory fragmentation
- Enables efficient memory reuse
- Supports dynamic sequence growth
- Improves serving batch capacity
- Reduces allocation overhead

---

## Continuous Request Batching

Incoming requests are grouped dynamically instead of executing independently.

```text
Traditional Serving

Request 1 ─────► GPU
Request 2 ─────► GPU
Request 3 ─────► GPU

Low GPU Utilization
```

↓

```text
NexusCache

Incoming Requests
        │
        ▼
Continuous Request Queue
        │
        ▼
Dynamic Batch Scheduler
        │
        ▼
Single GPU Execution

High GPU Utilization
```

---

## Native CUDA Runtime

Performance-critical operations execute entirely inside native C++/CUDA extensions.

Key optimizations include:

- Dynamic block allocation
- Virtual page table management
- Zero-copy pinned host memory
- Custom paged attention kernels
- Memory-aware allocation strategies
- PyBind11 native operator bindings

---

## Analytical Workload Engine

Beyond serving, NexusCache includes analytical models for understanding runtime behavior.

Capabilities include:

- VRAM saturation prediction
- Queueing theory simulation
- Sequence length distribution analysis
- Latency vs throughput A/B evaluation
- Workload forecasting under peak traffic

---

# 📂 Repository Structure

```text
.
├── benchmarks/                        # Performance, Telemetry & Load Generator
│   ├── docker-compose.yaml            # Containerized benchmark environment
│   ├── Dockerfile.gpu                 # GPU container image definition
│   ├── load_generator.py              # Async load testing engine
│   ├── profile_memory.sh              # GPU memory & power telemetry logger
│   └── static_vs_dynamic_benchmark.py # A/B latency-throughput test suite
├── csrc/                              # Native C++ / CUDA Subsystem
│   ├── bindings/                      # PyBind11 bindings
│   │   └── bindings.cpp
│   ├── include/nexuscache/            # Core C++ Headers
│   │   ├── block_manager.hpp
│   │   ├── kernels.hpp
│   │   ├── memory_utils.hpp
│   │   ├── page_table.hpp
│   │   └── pinned_memory.hpp
│   ├── kernels/                       # Raw CUDA Memory & Attention Kernels
│   │   ├── memory_kernels.cu
│   │   └── paged_attention_kernel.cu
│   └── src/                           # Core C++ Implementations
│       ├── block_manager.cpp
│       ├── memory_utils.cpp
│       ├── page_table.cpp
│       └── pinned_memory.cpp
├── nexuscache/                        # Main Python Module
│   ├── analytics/                     # DS & Math Modeling Suite
│   │   ├── ab_evaluator.py            # Quantitative A/B evaluation
│   │   ├── ab_testing_harness.py      # Statistical benchmark harness
│   │   ├── queue_model.py             # Time-series M/M/c queueing theory
│   │   ├── vram_saturation_model.py   # Analytical VRAM bounds model
│   │   └── workload_model.py          # Sequence length distribution modeling
│   ├── server/                        # Asynchronous Serving Layer
│   │   ├── api_server.py              # FastAPI Web / Streaming Server
│   │   ├── dynamic_scheduler.py       # Dynamic request batcher
│   │   ├── pipeline_manager.py        # Pipeline parallel execution engine
│   │   ├── ray_actor.py               # Ray distribution actor bindings
│   │   ├── request_queue.py           # Priority request queue
│   │   ├── scheduler.py               # Serving scheduler
│   │   └── worker.py                  # Execution worker process
│   ├── utils/                         # Logging, Config & Metrics
│   │   ├── config.py
│   │   └── metrics.py
│   └── cache_engine.py                # High-level KV-Cache PyTorch wrapper
├── tests/                             # Full Unit & Integration Test Suite
│   ├── cpp/                           # Native GTest / C++ tests
│   │   ├── test_block_allocator.cpp
│   │   ├── test_page_table.cpp
│   │   └── test_pinned_memory.cpp
│   └── python/                     # PyTest Python & Ray integration tests
│       ├── test_ab_evaluator.py
│       ├── test_continuous_scheduler.py
│       ├── test_memory_subsystem.py
│       ├── test_native_extension.py
│       ├── test_page_table.py
│       ├── test_paged_attention.py
│       ├── test_pipeline_manager.py
│       ├── test_queue_model.py
│       ├── test_ray_worker.py
│       ├── test_schedulers.py
│       └── test_workload_model.py
├── CMakeLists.txt                  # Native CMake Build Configuration
├── Dockerfile                      # Standard Dockerfile
├── pyproject.toml                  # Python Build Configuration
├── setup.py                        # C++/CUDA PyTorch Setuptools Installer
└── README.md
```

---

## Module Overview

| Module | Responsibility |
|---------|----------------|
| **csrc/** | Native C++17/CUDA runtime, page allocator, memory subsystem and PyBind11 bindings |
| **server/** | FastAPI serving layer, continuous batching, Ray scheduling and execution pipeline |
| **analytics/** | Queueing theory, VRAM forecasting, workload modeling and A/B evaluation |
| **benchmarks/** | Load generation, GPU telemetry, latency and throughput profiling |
| **tests/** | Native C++ validation, Python integration tests and scheduler correctness |

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Runtime | Python, C++17 |
| GPU Computing | CUDA, PyTorch |
| Native Bindings | PyBind11 |
| Distributed Systems | Ray |
| API | FastAPI |
| Build System | CMake, setuptools |
| Deployment | Docker |
| Profiling | NVIDIA Nsight Compute |

---
# 🚀 Installation

## Requirements

- Python **3.10+**
- CUDA **12.0+**
- PyTorch **2.x**
- NVIDIA GPU with CUDA support
- CMake **3.20+**

Clone the repository:

```bash
git clone https://github.com/BAHAA00111/NexusCache-Engine.git

cd NexusCache-Engine
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip

pip install -e .
```

---

# ⚙️ Build Native CUDA Runtime

Compile the C++/CUDA extension.

```bash
NO_CUDA_EXT=0 python setup.py build_ext --inplace
```

Verify the installation:

```bash
python -c "import nexuscache._C as _C; print('Native Runtime Loaded Successfully')"
```

---

# ⚡ Quick Start

## Launch the API Server

```bash
python -m nexuscache.server.api_server
```

---

## Execute the Test Suite

Run all Python integration tests.

```bash
pytest tests/python -v
```

Run native C++ tests.

```bash
ctest --output-on-failure
```

---

## Benchmark the Runtime

Execute the built-in load generator.

```bash
python benchmarks/load_generator.py
```

Profile GPU memory behavior.

```bash
bash benchmarks/profile_memory.sh \
    --concurrency 32 \
    --num-requests 100 \
    --url http://localhost:8000/v1/chat/completions
```

Evaluate allocation strategies.

```bash
python benchmarks/static_vs_dynamic_benchmark.py
```

---

# 🐳 Docker Deployment

Build the GPU container.

```bash
docker build -t nexuscache-engine .
```

Run the serving runtime.

```bash
docker run \
    --gpus all \
    -p 8000:8000 \
    nexuscache-engine
```

For distributed deployments, Docker Compose configurations are provided under the **benchmarks/** directory.

---


# 📚 Technology References

NexusCache Engine draws inspiration from modern LLM serving and GPU runtime systems, including:

- vLLM
- Ray
- PyTorch
- CUDA
- PyBind11
- FastAPI
- NVIDIA Nsight Compute

---

# 📄 License

Licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for additional information.

---

<div align="center">

## ⭐ Support the Project

If you find **NexusCache Engine** useful, consider giving the repository a **Star**.

Your support helps improve project visibility and encourages continued development.

---

**Built for scalable, high-throughput, and production-ready LLM serving.**

</div>
