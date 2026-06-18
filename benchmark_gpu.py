#!/usr/bin/env python3
"""
GPU Model Benchmark Script
Measures per-batch compute time for torchvision models with random data.
Usage: python benchmark_gpu.py --model resnet50 --batch-size 256 --optimizer adam
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import time
import csv
import os
from datetime import datetime

# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser(description="Benchmark torchvision model on GPU")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., resnet50, vit_b_16)")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer type")
parser.add_argument("--num-batches", type=int, default=200, help="Number of batches to benchmark")
parser.add_argument("--warmup", type=int, default=50, help="Warmup batches (not recorded)")
parser.add_argument("--output-dir", type=str, default="/netscratch/nague/gpu_benchmarks", help="Output directory for CSV")
args = parser.parse_args()

# ============================================================
# SETUP
# ============================================================
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"Model: {args.model}")
print(f"Batch size: {args.batch_size}")
print(f"Optimizer: {args.optimizer}")
print(f"Batches: {args.num_batches} (+ {args.warmup} warmup)")

# Load model
try:
    model = getattr(models, args.model)(weights=None)
except AttributeError:
    print(f"Error: model '{args.model}' not found in torchvision.models")
    print("Available models:", [m for m in dir(models) if m.islower() and not m.startswith('_')])
    exit(1)

model = model.cuda()
model.train()

# Optimizer
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters())
else:
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # no momentum

criterion = nn.CrossEntropyLoss()

# ============================================================
# WARMUP
# ============================================================
print(f"\nWarmup ({args.warmup} batches)...")
for i in range(args.warmup):
    images = torch.randn(args.batch_size, 3, 224, 224).cuda()
    labels = torch.randint(0, 1000, (args.batch_size,)).cuda()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# ============================================================
# BENCHMARK
# ============================================================
print(f"Benchmarking ({args.num_batches} batches)...")
results = []

for i in range(args.num_batches):
    images = torch.randn(args.batch_size, 3, 224, 224).cuda()
    labels = torch.randint(0, 1000, (args.batch_size,)).cuda()

    torch.cuda.synchronize()
    start = time.perf_counter()

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    results.append(elapsed)

    if (i + 1) % 100 == 0:
        avg_so_far = sum(results) / len(results)
        print(f"  Batch {i+1}/{args.num_batches}: avg={avg_so_far*1000:.2f}ms, last={elapsed*1000:.2f}ms")

# ============================================================
# STATISTICS
# ============================================================
import statistics
mean_time = statistics.mean(results)
std_time = statistics.stdev(results)
min_time = min(results)
max_time = max(results)
median_time = statistics.median(results)

print(f"\nResults for {args.model} (batch_size={args.batch_size}, opt={args.optimizer}):")
print(f"  Mean:   {mean_time*1000:.2f} ms")
print(f"  Std:    {std_time*1000:.2f} ms")
print(f"  Median: {median_time*1000:.2f} ms")
print(f"  Min:    {min_time*1000:.2f} ms")
print(f"  Max:    {max_time*1000:.2f} ms")
print(f"  Throughput: {args.batch_size/mean_time:.1f} img/s")

# ============================================================
# SAVE RESULTS
# ============================================================
os.makedirs(args.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = os.path.join(
    args.output_dir,
    f"benchmark_{args.model}_bs{args.batch_size}_{args.optimizer}_{timestamp}.csv"
)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "batch_size", "optimizer", "gpu", "gpu_memory_gb",
        "batch_index", "time_seconds", "time_ms"
    ])
    for i, t in enumerate(results):
        writer.writerow([
            args.model, args.batch_size, args.optimizer, gpu_name, f"{gpu_mem:.1f}",
            i + 1, f"{t:.6f}", f"{t*1000:.2f}"
        ])

# Also save summary
summary_file = os.path.join(
    args.output_dir,
    f"summary_{args.model}_bs{args.batch_size}_{args.optimizer}_{timestamp}.csv"
)
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "batch_size", "optimizer", "gpu", "gpu_memory_gb",
        "num_batches", "mean_ms", "std_ms", "median_ms", "min_ms", "max_ms",
        "throughput_img_s"
    ])
    writer.writerow([
        args.model, args.batch_size, args.optimizer, gpu_name, f"{gpu_mem:.1f}",
        args.num_batches, f"{mean_time*1000:.2f}", f"{std_time*1000:.2f}",
        f"{median_time*1000:.2f}", f"{min_time*1000:.2f}", f"{max_time*1000:.2f}",
        f"{args.batch_size/mean_time:.1f}"
    ])

print(f"\nRaw data saved: {csv_file}")
print(f"Summary saved: {summary_file}")
