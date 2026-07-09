#!/usr/bin/env python3
"""Benchmark: Zip format (no compression)."""
import os
import io
import time
import zipfile
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
ZIP_PATH = "/fscratch/nague/storage_benchmarks/images.zip"

# ============================================================
# WRITE
# ============================================================
print("Writing zip (no compression)...")
os.makedirs(os.path.dirname(ZIP_PATH), exist_ok=True)

samples = []
for class_dir in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.JPEG', '.jpg', '.png')):
                samples.append((os.path.join(class_path, fname), class_dir, fname))
with zipfile.ZipFile(ZIP_PATH, 'w', compression=zipfile.ZIP_STORED) as zf:
    for img_path, label, fname in samples:
        zf.write(img_path, arcname=f"{label}/{fname}")

zip_size = os.path.getsize(ZIP_PATH)
raw_size = sum(os.path.getsize(s[0]) for s in samples)
print(f"Zip size: {zip_size/1024/1024:.1f} MB")
print(f"Raw size: {raw_size/1024/1024:.1f} MB")
print(f"Overhead: {(zip_size/raw_size - 1)*100:.1f}%")

# ============================================================
# READ DATASET
# ============================================================
class ZipDataset(Dataset):
    def __init__(self, zip_path):
        self.zf = zipfile.ZipFile(zip_path, 'r')
        self.names = [n for n in self.zf.namelist() if n.endswith(('.JPEG', '.jpg', '.png'))]
        self.names.sort()
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        img_bytes = self.zf.read(self.names[idx])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.names[idx].split('/')[0]
        return img, label

# ============================================================
# BENCHMARK
# ============================================================
dataset = ZipDataset(ZIP_PATH)
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
