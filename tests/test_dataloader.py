#!/usr/bin/env python3
"""Benchmark minimal_dataset DataLoader with different worker counts."""
import sys
import time
sys.path.insert(0, '/home/nague/bachelor-project')
from minimal_dataset import BaseDataset, DataLoader

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
NUM_SAMPLES = 1000  # Assez pour voir une différence entre workers
BATCH_SIZE = 32

print(f"Dataset: {NUM_SAMPLES} samples, batch_size={BATCH_SIZE}")
print(f"{'Workers':<10} {'Batches':<10} {'Time(s)':<10} {'Samples/s':<12}")
print("-" * 45)

for num_workers in [1, 2]:
    dataset = BaseDataset(DATA_DIR, max_samples=NUM_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers)

    batch_count = 0
    start = time.time()
    for batch in loader:
        batch_count += 1
    elapsed = time.time() - start

    samples = batch_count * BATCH_SIZE
    print(f"{num_workers:<10} {batch_count:<10} {elapsed:<10.2f} {samples/elapsed:<12.1f}")

print("\nDone!")

