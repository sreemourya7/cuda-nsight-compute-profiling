# Kernel 02 — Global Memory Coalescing | Nsight Compute

Raw NVIDIA Nsight Compute profiling measurements for the FP32 SGEMM global-memory-coalescing kernel.

**[Profiling Analysis & Explanation](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/02_global_memory_coalescing/profiling.md)**

## 1. Profiling Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| Architecture | Turing (SM 7.5) |
| SM Count | 14 |
| CUDA Toolkit | 13.1 |
| Nsight Compute | 2025.4.1 |
| Kernel | `sgemm_coalescing` |
| Matrix | M=N=K=512 |
| Thread Block | 1024 threads |
| Grid | 16 × 16 (256 blocks) |
| Registers / Thread | 50 |

## 2. Profiling Commands

Run from the Kernel 02 directory in an Administrator x64 Native Tools Command Prompt.

**GPU Speed of Light, Launch Statistics & Occupancy**

```cmd
ncu --kernel-name sgemm_coalescing --launch-count 1 gmem_gemm.exe 512 1 0
```

**Scheduler, Warp State & Memory Workload**

```cmd
ncu --kernel-name sgemm_coalescing --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis gmem_gemm.exe 512 1 0
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
| Kernel Duration | 1.05 ms |
| SM Throughput | 86.47% |
| Memory Throughput | 86.47% |
| L1/TEX Cache Throughput | 89.58% |
| L2 Cache Throughput | 10.47% |
| DRAM Throughput | 9.06% |
| SM Frequency | 1.33 GHz |
| DRAM Frequency | 4.98 GHz |
| Elapsed Cycles | 1,403,407 |
| SM Active Cycles | 1,341,738.36 |

## 4. Launch Statistics & Occupancy

| Metric | Value |
|---|---:|
| Grid Size | 256 blocks |
| Block Size | 1024 threads |
| Total Threads | 262,144 |
| Registers / Thread | 50 |
| Static Shared Memory / Block | 0 |
| Dynamic Shared Memory / Block | 0 |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 98.52% |
| Theoretical Active Warps / SM | 32 |
| Achieved Active Warps / SM | 31.53 |
| Block Limit — Registers | 1 |
| Block Limit — Warps | 1 |
| Waves / SM | 18.29 |

## 5. Scheduler Statistics

| Metric | Value |
|---|---:|
| Active Warps / Scheduler | 7.91 |
| Eligible Warps / Scheduler | 1.13 |
| Issued Warps / Scheduler | 0.31 |
| One or More Eligible | 31.46% |
| No Eligible Warp | 68.54% |
| Average Cycles / Warp Issue | ~3.2 |

## 6. Warp State Statistics

| Metric | Value |
|---|---:|
| Warp Cycles / Issued Instruction | 25.15 |
| Warp Cycles / Executed Instruction | 25.16 |
| LG Throttle | ~18.9 cycles |
| LG Throttle / Issue Interval | ~75.3% |
| Average Active Threads / Warp | 32 |
| Average Not Predicated Off Threads / Warp | 31.92 |

## 7. Memory Workload Analysis

| Metric | Value |
|---|---:|
| L1/TEX Hit Rate | 94.86% |
| L2 Hit Rate | 77.90% |
| Reported Memory Throughput | 14.40 GB/s |
| Mem Busy | 43.03% |
| Max Bandwidth | 86.02% |
| Mem Pipes Busy | 86.02% |
| L2 Sector Promotion Misses | 39.39% |
| Local Memory Spilling Requests | 0 |
| Local Memory Spilling Request Overhead | 0% |
| Shared Memory Spilling Requests | 0 |
| Shared Memory Spilling Request Overhead | 0% |

## 8. Profiling Status

| GPU | Status |
|---|---|
| GTX 1650 (Turing) | Profiled |
| RTX 5090 (Blackwell) | Pending |

---

**Note:** Measurements were collected using two Nsight Compute profiling runs. The reported metrics belong to their respective profiling sections.

For performance interpretation, bottleneck investigation, and optimization hypotheses, see the [Kernel 02 Profiling Analysis](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/02_global_memory_coalescing/profiling.md).