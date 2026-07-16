#!/usr/bin/env python3
"""Benchmark DataLoader across batch sizes and worker counts."""
import sys
import json
import os
import time
sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import BaseDataset, DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-workers", type=int, required=True)
parser.add_argument("--num-samples", type=int, default=500)
args = parser.parse_args()

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
METRICS_DIR = "/netscratch/nague/dataloader_benchmarks"

dataset = BaseDataset(DATA_DIR, max_samples=args.num_samples)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    metrics_dir=METRICS_DIR
)

batch_count = 0
start = time.time()
for batch in loader:
    batch_count += 1
    if batch_count % 100 == 0:
        print(f"  [{args.batch_size}/{args.num_workers}] batch {batch_count}, {batch_count * args.batch_size} samples", flush=True)
elapsed = time.time() - start

summary = loader._cleanup()
samples = batch_count * args.batch_size

# Print single-line CSV for easy aggregation
print(f"{args.batch_size},{args.num_workers},{batch_count},{elapsed:.2f},"
      f"{samples/elapsed:.1f},"
      f"{summary['aggregate']['avg_utilization_pct']:.1f},"
      f"{summary['staging_queue']['empty_events']},"
      f"{summary['staging_queue']['full_events']},"
      f"{summary['batch_queue']['total_get_wait_s']:.2f}")
