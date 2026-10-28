# Kernel 04 - 1D Block Tiling: Raw Nsight Compute Metrics

This file records **commands and measured profiler outputs**, without performance diagnosis. Interpretation belongs in [`profiling.md` of the SGEMM implementation repository](https://github.com/sreemourya7/SGEMM-kernel-optimization-FP32/blob/main/experiments/04_1d_block_tiling/profiling.md).

## Environment and kernel

| Parameter | Value |
|---|---|
| GPU | NVIDIA GTX 1650 (Turing, `sm_75`), 14 SMs |
| CUDA toolkit | 13.1 |
| Nsight Compute CLI | 2025.4.1 |
| Kernel symbol | `sgemm_block1d` |
| Source/binary | `block1d_gemm.cu` / `block1d_gemm.exe` |
| Tile | `BM=64, BN=64, BK=8, TM=8` |
| Launch at 512 | Grid `(8,8)`, 512 threads/block |
| Launch at 4096 | Grid `(64,64)`, 512 threads/block |
| Shared memory/block | 4,096 bytes |
| Build | `nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo block1d_gemm.cu -o block1d_gemm.exe` |

Run the following commands from the `experiments/04_1d_block_tiling` directory. `512 1 0` and `4096 1 0` mean `matrix_size runs verification_flag`. Nsight may replay the selected kernel internally.

## 1. Basic GPU / launch / occupancy metrics

**512 x 512:**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 block1d_gemm.exe 512 1 0
```

**4096 x 4096:**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 block1d_gemm.exe 4096 1 0
```

| Metric | 512 x 512 | 4096 x 4096 |
|---|---:|---:|
| NCU profiling passes | Not recorded in notes | 9 |
| NCU kernel duration | 460.10 us | 196.62 ms |
| SM frequency | 1.34 GHz | 1.39 GHz |
| DRAM frequency | 5.01 GHz | 5.99 GHz |
| Compute (SM) Throughput | 42.73% | 49.02% |
| Memory Throughput (SOL) | 42.73% | 49.02% |
| L1/TEX Cache Throughput | 72.06% | 82.91% |
| L2 Cache Throughput | 12.55% | 13.17% |
| DRAM Throughput | 12.31% | 13.37% |
| Registers/thread | 66 | 66 |
| Theoretical occupancy | 50.00% | 50.00% |
| Achieved occupancy | 50.20% | 49.99% |
| Active warps/SM | 16.06 | 16.00 |
| Theoretical active warps/SM | 16 | 16 |
| Grid size (blocks) | 64 | 4,096 |
| Block size (threads) | 512 | 512 |
| Waves per SM | ~4.57 | 292.57 |
| Block limit: registers | 1 | 1 |
| Block limit: shared memory | 8 | 8 |
| Block limit: warps | 2 | 2 |

**Note:** Nsight's `Duration` is the time during this collection, not the five-run standalone benchmark median. At `4096`, the standalone median was **156.4366 ms**.

## 2. Scheduler, warp state, and memory workload

**512 x 512 (main sections):**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis block1d_gemm.exe 512 1 0
```

**512 x 512 (full stall breakdown):**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section WarpStateStats --print-details all block1d_gemm.exe 512 1 0
```

**4096 x 4096 (full details):**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section SchedulerStats --section WarpStateStats --section MemoryWorkloadAnalysis --print-details all block1d_gemm.exe 4096 1 0
```

### Scheduler statistics

| Metric | 512 x 512 | 4096 x 4096 |
|---|---:|---:|
| One or More Eligible | 24.53% | 25.47% |
| No Eligible | 75.47% | 74.53% |
| Issued Warp Per Scheduler | 0.25 | 0.25 |
| Active Warps Per Scheduler | 3.99 | 4.00 |
| Eligible Warps Per Scheduler | 0.60 | 0.64 |
| GPU Maximum Warps Per Scheduler | 8 | 8 |

### Warp state statistics

The 512 values below are from the `--print-details all` collection, in which the issue interval was 16.23 cycles (a separate section collection reported 16.28).

| Metric | 512 x 512 | 4096 x 4096 |
|---|---:|---:|
| Warp Cycles Per Issued Instruction | 16.23 | 15.72 |
| Warp Cycles Per Executed Instruction | 16.23 | 15.72 |
| Avg. Active Threads Per Warp | 32 | 32 |
| Avg. Not Predicated Off Threads Per Warp | 31.99 | 32.00 |
| Stall MIO Throttle | 4.62 | 4.03 |
| Stall Long Scoreboard | 4.22 | 4.09 |
| Stall Wait | 1.81 | 1.80 |
| Stall Barrier | 1.58 | 1.72 |
| Stall Not Selected | 1.48 | 1.52 |
| Stall Math Pipe Throttle | 0.99 | 1.06 |
| Stall Short Scoreboard | 0.30 | 0.32 |
| Stall Dispatch Stall | 0.06 | 0.06 |
| Stall No Instruction | 0.05 | 0.05 |
| Stall Branch Resolving | 0.03 | 0.03 |
| Stall LG Throttle | 0.00 | 0.00 |
| Selected | 1.00 | 1.00 |

**Units for stall rows:** average warp cycles per issued instruction (Nsight's warp-state breakdown), not overall execution cycles.

### Memory Workload Analysis

| Metric | 512 x 512 | 4096 x 4096 |
|---|---:|---:|
| Memory Throughput | 19.59 GB/s | 25.73 GB/s |
| Mem Busy | 36.29% | 41.48% |
| Max Bandwidth | 43.04% | 49.05% |
| Mem Pipes Busy | 43.04% | 49.05% |
| L1/TEX Hit Rate | 5.56% | 0.77% |
| L2 Hit Rate | 62.97% | 45.45% |
| L2 Sector Promotion Misses | 9.35% | 6.05% |
| Local Memory Spilling Requests | 0 | 0 |
| Shared Memory Spilling Requests | 0 | 0 |

A profiler warning indicated that the optional metric `smsp__pcsamp_sample_count` was unavailable in this collection. The requested sections completed.

## 3. Executed memory-instruction counters

**512 x 512:**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum block1d_gemm.exe 512 1 0
```

**4096 x 4096:**

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum block1d_gemm.exe 4096 1 0
```

| Raw metric | 512 x 512 | 4096 x 4096 |
|---|---:|---:|
| `smsp__sass_inst_executed_op_global_ld.sum` | 139,264 | 67,633,152 |
| `smsp__sass_inst_executed_op_global_st.sum` | 8,192 | 524,288 |
| `smsp__sass_inst_executed_op_shared_ld.sum` | 1,572,864 | 805,306,368 |
| `smsp__sass_inst_executed_op_shared_st.sum` | 131,072 | 67,108,864 |

**Unit:** executed warp-level SASS instructions, not bytes or memory transactions. The `4096` profiling run issued a warning about workload launch time but completed the single counter-collection pass. The timed run displayed during that pass was 196.7157 ms and should not replace the standalone five-run median.

## 4. SourceCounters (512 x 512)

Collect the report:

```cmd
ncu --kernel-name sgemm_block1d --launch-count 1 --section SourceCounters --import-source yes -o block1d_source_512 block1d_gemm.exe 512 1 0
```

Export to text:

```cmd
ncu --import block1d_source_512.ncu-rep --page source --print-source cuda,sass > block1d_source_512.txt
```

The collection completed in **5 passes** and produced `block1d_source_512.ncu-rep`. The full CUDA/SASS table is retained in [reports/source_counters_512.txt](reports/source_counters_512.txt).

### Selected source-level sample counts (as printed)

| CUDA line | Source region | SASS instruction type | Stall type | Samples |
|---|---|---|---|---:|
| 97 | `Bs[tid] = ...` | `STS [R0.X4+0x800], R15` | Long Scoreboard | 1,121 |
| 118 | A operand in inner K loop | multiple `LDS.U.128` | MIO Throttle | 1,041 |
| 112 | B operand in inner K loop | multiple `LDS.U` | MIO Throttle | 493 |

Source-level sample counts are separate from the normalized warp stall cycle values in Section 2. The `4096 x 4096` SourceCounters collection was interrupted due to excessive runtime; no 4096 source-line report is claimed.

## 5. Companion Kernel 03 reference values

The earlier Kernel 03 profiler records were collected separately. Selected comparison values, `512 / 4096`:

| Metric | Kernel 03 at 512 | Kernel 03 at 4096 |
|---|---:|---:|
| MIO Throttle | 23.60 | 23.80 |
| Warp Cycles Per Issued Instruction | 38.44 | 38.99 |
| No Eligible | 79.37% | 79.49% |
| Achieved occupancy | 99.78% | 99.98% |
| Registers/thread | 44 | 44 |
| Shared-memory loads (512 only) | 5,242,880 | Not collected here |
| Shared-memory stores (512 only) | 262,144 | Not collected here |
| Global-memory loads (512 only) | 270,336 | Not collected here |
| Global-memory stores (512 only) | 8,192 | Not collected here |

See [Kernel 03 raw profiling records](../03_smem_cuda/README.md) for original collection details.
