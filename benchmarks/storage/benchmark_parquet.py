#!/usr/bin/env python3
"""Benchmark: Parquet format."""
import os
import io
import time
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
PARQUET_PATH = "/fscratch/nague/storage_benchmarks/images.parquet"

# ============================================================
# WRITE
# ============================================================
print("Writing Parquet...")
os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)

samples = []
for class_dir in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.JPEG', '.jpg', '.png')):
                with open(os.path.join(class_path, fname), 'rb') as f:
                    img_bytes = f.read()
                samples.append({'image': img_bytes, 'label': class_dir, 'filename': fname})

table = pa.Table.from_pylist(samples)
pq.write_table(table, PARQUET_PATH)

parquet_size = os.path.getsize(PARQUET_PATH)
raw_size = sum(os.path.getsize(os.path.join(DATA_DIR, s['label'], s['filename'])) for s in samples)
print(f"Parquet size: {parquet_size/1024/1024:.1f} MB")
print(f"Raw size: {raw_size/1024/1024:.1f} MB")
print(f"Overhead: {(parquet_size/raw_size - 1)*100:.1f}%")

# ============================================================
# READ DATASET
# ============================================================
class ParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.table = pq.read_table(parquet_path)
    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pylist()[0]
        img = Image.open(io.BytesIO(row['image'])).convert('RGB')
        return img, row['label']

# ============================================================
# BENCHMARK
# ============================================================
dataset = ParquetDataset(PARQUET_PATH)
print(f"Dataset: {len(dataset)} samples")

# Warmup
for i in range(100):
    _ = dataset[i]

# Sequential
start = time.perf_counter()
for i in range(len(dataset)):
    _ = dataset[i]
elapsed = time.perf_counter() - start
print(f"Sequential: {len(dataset)} in {elapsed:.2f}s -> {len(dataset)/elapsed:.1f} samples/s")

# Random
import random
indices = [random.randint(0, len(dataset)-1) for _ in range(5000)]
start = time.perf_counter()
for i in indices:
    _ = dataset[i]
elapsed = time.perf_counter() - start
print(f"Random (5000): {elapsed:.2f}s -> {5000/elapsed:.1f} samples/s")
