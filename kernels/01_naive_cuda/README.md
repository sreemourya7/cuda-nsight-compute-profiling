# Kernel 01 — Naive CUDA SGEMM | Nsight Compute

Raw NVIDIA Nsight Compute profiling measurements for the naive FP32 SGEMM kernel.

**[Profiling Analysis & Explanation](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/01_naive_cuda/profiling.md)**

## 1. Profiling Setup

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| Architecture | Turing (SM 7.5) |
| SM Count | 14 |
| CUDA Toolkit | 13.1 |
| Nsight Compute | 2025.4.1 |
| Kernel | `sgemm_naive` |
| Matrix Dimensions | M=N=K=512 |
| Thread Block | 32 × 32 (1024 threads) |
| Grid | 16 × 16 (256 blocks) |
| Registers / Thread | 50 |

## 2. Profiling Commands

**GPU Speed of Light & Occupancy**

```bash
ncu --kernel-name sgemm_naive --launch-count 1 naive_gemm.exe 512 1 0
```

**Scheduler, Warp State & Memory Analysis**

```bash
ncu --kernel-name sgemm_naive --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis naive_gemm.exe 512 1 0
```

| Profiling Run | Replay Passes |
|---|---:|
| GPU Speed of Light / Occupancy | 9 |
| Scheduler / Warp State / Memory | 11 |

## 3. GPU Speed of Light

| Metric | Value |
|---|---:|
| Kernel Duration | 15.22 ms |
| SM Throughput | 5.86% |
| Memory Throughput | 48.32% |
| L1/TEX Cache Throughput | 96.63% |
| L2 Cache Throughput | 1.17% |
| DRAM Throughput | 0.76% |
| SM Frequency | ~1.35 GHz |
| DRAM Frequency | ~5.99 GHz |

## 4. Occupancy

| Metric | Value |
|---|---:|
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 98.28% |
| Theoretical Active Warps / SM | 32 |
| Achieved Active Warps / SM | 31.45 |
| Registers / Thread | 50 |
| Block Limit — Registers | 1 |
| Block Limit — Warps | 1 |
| Waves / SM | 18.29 |

## 5. Scheduler Statistics

| Metric | Value |
|---|---:|
| Active Warps / Scheduler | 7.86 |
| Eligible Warps / Scheduler | 0.08 |
| Issued Warps / Scheduler | 0.02 |
| One or More Eligible | 2.13% |
| No Eligible Warp | 97.87% |
| Average Cycles / Warp Issue | ~46.9 |

## 6. Warp State Statistics

| Metric | Value |
|---|---:|
| Warp Cycles / Issued Instruction | 369.19 |
| Warp Cycles / Executed Instruction | 369.21 |
| LG Throttle | ~353.8 cycles |
| Average Active Threads / Warp | 32 |
| Average Non-Predicated-Off Threads / Warp | ~31.92 |

## 7. Memory Workload Analysis

| Metric | Value |
|---|---:|
| L1/TEX Hit Rate | 99.21% |
| L2 Hit Rate | 73.02% |
| Reported Memory Throughput | 1.52 GB/s |
| Mem Busy | 48.21% |
| Max Bandwidth | 13.13% |
| Mem Pipes Busy | 5.84% |
| Local Memory Spilling Requests | 0 |
| Shared Memory Spilling | 0 |

---

## Profiling Status

| GPU | Status |
|---|---|
| GTX 1650 (Turing) | Profiled |
| RTX 5090 (Blackwell) | Pending |

**Note:** Metrics are transcribed from two Nsight Compute profiling runs. Values are grouped by their original profiling sections; measurements from different runs should not be treated as a single simultaneous capture.

For interpretations, bottleneck diagnosis, and optimization hypotheses, see the [Kernel 01 Profiling Analysis](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/01_naive_cuda/profiling.md).
