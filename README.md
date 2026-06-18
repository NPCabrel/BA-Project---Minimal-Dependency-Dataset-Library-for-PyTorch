# GPU Benchmarking & Dummy Model Framework

Bachelor Project: *Minimal Dependency Dataset Library for PyTorch* — DFKI 2026.

## Overview

This repository contains tools to benchmark GPU compute delays and build a dummy model for data loading pipeline testing without requiring a real GPU.

## Files

| File | Purpose |
|------|---------|
| `benchmark_gpu.py` | Quick GPU delay measurement (~200 batches). Measures mean/std per model/GPU/optimizer. |
| `calibrate_dummy.py` | Deep GPU stability analysis (1000 batches). Anomaly detection. |
| `dummy_model.py` | Simulates GPU compute delay without a GPU. Variable delay with Gaussian noise. |
| `launch_benchmarks.py` | Automates SLURM job submission for all models across all GPU types. |
| `train_resnet50.py` | ResNet-50 single GPU baseline. 75.69% val accuracy. |
| `train_resnet50_ddp.py` | DDP training on 2 GPUs. 95% scaling efficiency. |
| `analyze_dataloading.py` | Profiles data loading vs. compute time. |

## Usage

```bash
python3 benchmark_gpu.py --model resnet50 --batch-size 256 --optimizer sgd
python3 calibrate_dummy.py --model resnet50 --batch-size 256 --optimizer adam
python3 dummy_model.py
