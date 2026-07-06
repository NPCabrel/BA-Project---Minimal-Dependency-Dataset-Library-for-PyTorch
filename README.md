# GPU Benchmarking & Storage Format Evaluation

Bachelor Project: *Minimal Dependency Dataset Library for PyTorch* — DFKI 2026.

## Overview

This repository contains tools to:
1. Benchmark GPU compute delays across models and GPUs
2. Build a dummy model for GPU-free data loading tests
3. Evaluate storage formats for efficient data access

## Files

### GPU Benchmarking
| File                    | Purpose                                                                       |
| ----------------------- | ----------------------------------------------------------------------------- |
| `benchmark_gpu.py`      | Quick GPU delay measurement (~200 batches). Mean/std per model/GPU/optimizer. |
| `calibrate_dummy.py`    | Deep GPU stability analysis (1000 batches). Anomaly detection.                |
| `dummy_model.py`        | Simulates GPU compute delay without a GPU. Reads from `gpu_delays.json`.      |
| `build_delay_config.py` | Generates `gpu_delays.json` from benchmark summary CSVs.                      |
| `launch_benchmarks.py`  | SLURM job submission for all models × GPUs (original).                        |
| `launch_arrays.py`      | SLURM job arrays (better priority) for missing benchmarks.                    |
| `find_missing.py`       | Finds which GPU/model/optimizer/batch_size combos are missing.                |
| `gpu_delays.json`       | Delay config for dummy model (auto-generated).                                |

### Training Baselines
| File                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `train_resnet50.py`      | ResNet-50 single GPU baseline. 75.69% val accuracy. |
| `train_resnet50_ddp.py`  | DDP training on 2 GPUs. 95% scaling efficiency.     |
| `analyze_dataloading.py` | Profiles data loading vs. compute time.             |

### Storage Format Benchmarks
| File                        | Purpose                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| `benchmark_plain.py`        | Baseline: plain JPEG files. 267 seq, 256 random samples/s.         |
| `benchmark_zip.py`          | Zip (no compression). 342 seq, 233 random s/s. NOT thread-safe.    |
| `benchmark_tar.py`          | Tar. 424 seq s/s. No true random access.                           |
| `benchmark_lmdb.py`         | LMDB key-value store. 416 seq, 305 random s/s. Thread-safe.        |
| `benchmark_parquet.py`      | Parquet columnar format. **473 seq, 474 random s/s.** Thread-safe. |
| `benchmark_msgpack.py`      | Sharded binary blob + MessagePack index. 431 seq, 313 random s/s.  |
| `write_lmdb.py` / `.sbatch` | Writer for LMDB format.                                            |
| `file_io.py`                | BinaryReader/BinaryWriter for MessagePack format (tutor's code).   |

## Storage Format Results Summary

| Format      | Sequential (s/s) | Random (s/s) | Overhead | Thread-Safe |
| ----------- | ---------------- | ------------ | -------- | ----------- |
| Parquet     | **472.7**        | **473.7**    | -0.3%    | Yes         |
| MessagePack | 431.4            | 312.9        | 0.0%     | Not Really  |
| Tar         | 424.1            | N/A          | 1.5%     | Not Realy   |
| LMDB        | 415.8            | 304.7        | 1.8%     | Yes         |
| Zip         | 341.5            | 233.2        | 0.1%     | Not safe    |
| Plain       | 266.6            | 255.9        | 0%       | Yes         |

*100,000 images from tiny-imagenet on fscratch (SSD). Single-threaded.*

## GPU Benchmark Results

10 GPU types, 16 model architectures, 3,540 runs total.

| GPU       | Models | Avg Delay (BS=256, SGD) |
| --------- | ------ | ----------------------- |
| H200      | 15     | 522 ms                  |
| RTXB6000  | 15     | 537 ms                  |
| RTX3090   | 5      | 569 ms                  |
| H100      | 15     | 597 ms                  |
| L40S      | 11     | 663 ms                  |
| A100-40GB | 9      | 702 ms                  |
| B200      | 16     | 711 ms                  |
| A100-80GB | 13     | 829 ms                  |
| RTXA6000  | 9      | 1,042 ms                |
| A40       | 10     | 1,060 ms                |

## Usage

```bash
# GPU benchmark
python3 benchmark_gpu.py --model resnet50 --batch-size 256 --optimizer sgd

# Calibrate for dummy model
python3 calibrate_dummy.py --model resnet50 --batch-size 256 --optimizer adam

# Dummy model
python3 dummy_model.py

# Storage benchmark
python3 benchmark_parquet.py
