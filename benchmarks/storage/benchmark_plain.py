#!/usr/bin/env python3
"""Benchmark: Plain files (baseline)."""
import os
import time
import random
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"

class PlainFileDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for class_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for fname in sorted(os.listdir(class_path)):
                    if fname.endswith(('.JPEG', '.jpg', '.png')):
                        self.samples.append(os.path.join(class_path, fname))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return Image.open(self.samples[idx]).convert('RGB')

dataset = PlainFileDataset(DATA_DIR)
print(f"Dataset: {len(dataset)} images")

# Warmup
for i in range(100):
    _ = dataset[i]

# Sequential
start = time.perf_counter()
for i in range(len(dataset)):
    _ = dataset[i]
seq_elapsed = time.perf_counter() - start
print(f"Sequential: {len(dataset)} in {seq_elapsed:.2f}s -> {len(dataset)/seq_elapsed:.1f} samples/s")

# Random
indices = [random.randint(0, len(dataset)-1) for _ in range(10000)]
start = time.perf_counter()
for i in indices:
    _ = dataset[i]
rand_elapsed = time.perf_counter() - start
print(f"Random (10000): {rand_elapsed:.2f}s -> {10000/rand_elapsed:.1f} samples/s")
