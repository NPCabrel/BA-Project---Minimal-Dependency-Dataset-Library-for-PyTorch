#!/usr/bin/env python3
"""Benchmark: Tar format. NOTE: tar does NOT support true random access."""
import os
import io
import time
import random
import tarfile
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
TAR_PATH = "/fscratch/nague/storage_benchmarks/images.tar"

# ============================================================
# WRITE
# ============================================================
print("Writing tar...")
os.makedirs(os.path.dirname(TAR_PATH), exist_ok=True)

samples = []
for class_dir in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.JPEG', '.jpg', '.png')):
                samples.append((os.path.join(class_path, fname), class_dir, fname))

with tarfile.open(TAR_PATH, 'w') as tar:
    for img_path, label, fname in samples:
        tar.add(img_path, arcname=f"{label}/{fname}")

tar_size = os.path.getsize(TAR_PATH)
raw_size = sum(os.path.getsize(s[0]) for s in samples)
print(f"Tar size: {tar_size/1024/1024:.1f} MB")
print(f"Raw size: {raw_size/1024/1024:.1f} MB")
print(f"Overhead: {(tar_size/raw_size - 1)*100:.1f}%")

# ============================================================
# READ DATASET
# ============================================================
class TarDataset(Dataset):
    def __init__(self, tar_path):
        self.tar = tarfile.open(tar_path, 'r')
        self.members = [m for m in self.tar.getmembers() if m.isfile() and m.name.endswith(('.JPEG', '.jpg', '.png'))]
        self.members.sort(key=lambda m: m.name)
    def __len__(self):
        return len(self.members)
    def __getitem__(self, idx):
        member = self.members[idx]
        f = self.tar.extractfile(member)
        return Image.open(io.BytesIO(f.read())).convert('RGB'), member.name.split('/')[0]

# ============================================================
# BENCHMARK
# ============================================================
dataset = TarDataset(TAR_PATH)
print(f"Dataset: {len(dataset)} samples")

# Warmup
for i in range(100):
    _ = dataset[i]

# Sequential
start = time.perf_counter()
for i in range(len(dataset)):
    _ = dataset[i]
seq_elapsed = time.perf_counter() - start
print(f"Sequential: {len(dataset)} in {seq_elapsed:.2f}s -> {len(dataset)/seq_elapsed:.1f} samples/s")

# NOTE: Tar does NOT support true random access.
# The following test accesses random indices but tar scans sequentially internally.
# Results are included for completeness but are misleading.
print("NOTE: Tar random access is simulated (sequential scan under the hood).")
indices = [random.randint(0, len(dataset)-1) for _ in range(5000)]
start = time.perf_counter()
for i in indices:
    _ = dataset[i]
rand_elapsed = time.perf_counter() - start
print(f"Random (5000, simulated): {rand_elapsed:.2f}s -> {5000/rand_elapsed:.1f} samples/s")
print("WARNING: This is NOT true random access. Tar scans sequentially for each lookup.")
