#!/usr/bin/env python3
"""Benchmark: LMDB format."""
import os
import time
import lmdb
import pickle
from PIL import Image
from torch.utils.data import Dataset

LMDB_PATH = "/fscratch/nague/storage_benchmarks/lmdb"

class LMDBDataset(Dataset):
    """Reads images from LMDB."""
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = str(idx).encode('utf-8')
            value = txn.get(key)
            img_bytes, label = pickle.loads(value)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return img, label

# Need io for BytesIO
import io

# Benchmark
dataset = LMDBDataset(LMDB_PATH)
print(f"Dataset: {len(dataset)} samples")

# Warmup
for i in range(100):
    _ = dataset[i]

# Sequential read
start = time.perf_counter()
for i in range(len(dataset)):
    _ = dataset[i]
elapsed = time.perf_counter() - start
print(f"Sequential: {len(dataset)} samples in {elapsed:.2f}s -> {len(dataset)/elapsed:.1f} samples/s")

# Random access
import random
indices = [random.randint(0, len(dataset)-1) for _ in range(10000)]
start = time.perf_counter()
for i in indices:
    _ = dataset[i]
elapsed = time.perf_counter() - start
print(f"Random: 10000 samples in {elapsed:.2f}s -> {10000/elapsed:.1f} samples/s")
