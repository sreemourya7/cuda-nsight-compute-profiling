# Kernel 03 — Shared Memory Tiling | Nsight Compute

Raw NVIDIA Nsight Compute profiling measurements for the FP32 SGEMM shared-memory tiling kernel.

**[Profiling Analysis & Explanation](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/03_shared_memory_tiling/profiling.md)**

## 1. Profiling Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| Architecture | Turing (SM 7.5) |
| SM Count | 14 |
| CUDA Toolkit | 13.1 |
| Nsight Compute | 2025.4.1 |
| Kernel | `sgemm_smem` |
| Matrix | M=N=K=512 |
| Thread Block | 32 × 32 (1024 threads) |
| Grid | 16 × 16 (256 blocks) |
| Tile Size | 32 × 32 |
| Registers / Thread | 44 |
| Shared Memory / Block | 8.19 KB |

## 2. Profiling Commands

Run from the Kernel 03 directory using an Administrator x64 Native Tools Command Prompt.

**GPU Speed of Light & Occupancy**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 smem_gemm.exe 512 1 0
```

**Scheduler, Warp State & Memory Workload**

```cmd
ncu --kernel-name sgemm_smem --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis smem_gemm.exe 512 1 0
```

| Configuration | Value |
|---|---|
| Matrix Size | 512 × 512 |
| Timed Runs | 1 |
| CPU Verification | Disabled |
| Detailed Profiling Replay Passes | 11 |

## 3. GPU Speed of Light

| Metric | Value |
|---|---:|
| Kernel Duration | 892 µs |
| Compute (SM) Throughput | 69.13% |
| Memory Throughput | 69.13% |
| L1/TEX Cache Throughput | 81.97% |
| L2 Cache Throughput | 12.22% |
| DRAM Throughput | 10.71% |
| SM Frequency | 1.34 GHz |
| DRAM Frequency | 4.99 GHz |
| Elapsed Cycles | 1,200,618 |
| SM Active Cycles | 1,152,616.93 |

## 4. Launch Statistics & Occupancy

| Metric | Value |
|---|---:|
| Grid Size | 256 blocks |
| Block Size | 1024 threads |
| Total Threads | 262,144 |
| Registers / Thread | 44 |
| Static Shared Memory / Block | 8.19 KB |
| Dynamic Shared Memory / Block | 0 |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 99.78% |
| Theoretical Active Warps / SM | 32 |
| Achieved Active Warps / SM | 31.93 |
| Block Limit — Registers | 1 |
| Block Limit — Shared Memory | 4 |
| Block Limit — Warps | 1 |
| Waves / SM | 18.29 |

## 5. Scheduler Statistics

| Metric | Value |
|---|---:|
| Active Warps / Scheduler | 7.93 |
| Eligible Warps / Scheduler | 1.00 |
| Issued Warps / Scheduler | 0.21 |
| One or More Eligible | 20.63% |
| No Eligible Warp | 79.37% |
| Average Cycles / Warp Issue | ~4.8 |

## 6. Warp State Statistics

| Metric | Value |
|---|---:|
| Warp Cycles / Issued Instruction | 38.44 |
| Warp Cycles / Executed Instruction | 38.45 |
| MIO Throttle | ~23.6 cycles |
| MIO Throttle / Issue Interval | ~61.4% |
| Average Active Threads / Warp | 32 |
| Average Not Predicated Off Threads / Warp | 31.94 |

## 7. Memory Workload Analysis

| Metric | Value |
|---|---:|
| L1/TEX Hit Rate | 2.94% |
| L2 Hit Rate | 62.36% |
| Reported Memory Throughput | 17.07 GB/s |
| Mem Busy | 41.07% |
| Max Bandwidth | 69.28% |
| Mem Pipes Busy | 69.28% |
| L2 Sector Promotion Misses | 0% |
| Local Memory Spilling Requests | 0 |
| Local Memory Spilling Request Overhead | 0% |
| Shared Memory Spilling Requests | 0 |
| Shared Memory Spilling Request Overhead | 0% |

## 8. Additional Profiler Findings

| Finding | Reported Value |
|---|---|
| Dominant Reported Stall | MIO Throttle |
| MIO Stall Contribution | ~23.6 cycles |
| Estimated MIO Stall Speedup | 61.37% |
| Scheduler Estimated Local Speedup | 79.37% |
| L2 Slice Imbalance Estimated Speedup | 7.228% |

*Speedup estimates are Nsight Compute optimization suggestions, not measured speedups.*

## 9. Profiling Status

| GPU | Status |
|---|---|
| GTX 1650 (Turing) | Profiled |
| RTX 5090 (Blackwell) | Pending |

---

**Note:** Measurements were collected across two Nsight Compute profiling runs. Metrics are grouped by their original profiling sections.

For interpretation, bottleneck investigation, and optimization hypotheses, see the [Kernel 03 Profiling Analysis](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/03_shared_memory_tiling/profiling.md).