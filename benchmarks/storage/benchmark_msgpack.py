#!/usr/bin/env python3
"""Benchmark: Sharded binary blob + MessagePack index (tutor's format)."""
import os
import sys
import time
import random

# Add project dir to path for imports
sys.path.insert(0, os.path.expanduser("~/bachelor-project"))

from benchmarks.storage.file_io import BinaryReader, BinaryWriter
# Import the tutor's classes from dataset_tutor.py (we'll save it separately)
# For now, we'll define them here since they're in dataset.txt
import msgpack
import numpy as np
import io
from PIL import Image
from torch.utils.data import Dataset

# ============================================================
# TUTOR'S CLASSES (from dataset.txt)
# ============================================================
class DatasetWriter:
    def __init__(self, output_dir: str, max_shard_size: int):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_shard_size = max_shard_size
        self.current_shard_index = 0
        self.current_shard_count = 0
        self.current_sample_count = 0
        self.writer = None
        self._open_new_shard()

    def _open_new_shard(self):
        if self.writer is not None:
            self.writer.close()
        shard_path = os.path.join(self.output_dir, f"{self.current_shard_index}")
        self.writer = BinaryWriter(shard_path)
        self.current_shard_count = 0
        self.current_shard_index += 1

    def write(self, element: dict):
        if self.current_shard_count == self.max_shard_size:
            self._open_new_shard()
        self.writer.write(msgpack.packb(element, use_bin_type=True))
        self.current_shard_count += 1
        self.current_sample_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
        self.writer = None

class ImageDataset(Dataset):
    def __init__(self, dataset_dir: str, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.shard_paths = []
        self.shard_readers = {}
        self.shard_lengths = []
        self.total_length = 0
        self._discover_shards()
        self._calculate_offsets()

    def _discover_shards(self):
        shard_indices = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                shard_indices.append(int(item))
        shard_indices.sort()
        for shard_idx in shard_indices:
            shard_path = os.path.join(self.dataset_dir, str(shard_idx))
            self.shard_paths.append(shard_path)
            reader = BinaryReader(shard_path)
            self.shard_readers[shard_idx] = reader
            self.shard_lengths.append(len(reader))
            self.total_length += len(reader)

    def _calculate_offsets(self):
        self.shard_offsets = [0]
        for length in self.shard_lengths:
            self.shard_offsets.append(self.shard_offsets[-1] + length)

    def _find_shard_and_index(self, global_index):
        for i, offset in enumerate(self.shard_offsets[1:]):
            if global_index < offset:
                return i, global_index - self.shard_offsets[i]
        raise IndexError(f"Index {global_index} out of bounds")

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        shard_idx, local_idx = self._find_shard_and_index(index)
        reader = self.shard_readers[shard_idx]
        raw_bytes = reader[local_idx]
        data_dict = msgpack.unpackb(raw_bytes, raw=False)
        return self._process_sample(data_dict)

    def _process_sample(self, data_dict):
        label = data_dict['label']  # Keep as string
        img_bytes = data_dict['image']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
# ============================================================
# WRITE
# ============================================================
DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
OUTPUT_DIR = "/fscratch/nague/storage_benchmarks/msgpack"
SHARD_SIZE = 10000  # samples per shard

print("Writing MessagePack shards...")
writer = DatasetWriter(OUTPUT_DIR, max_shard_size=SHARD_SIZE)

samples = []
for class_dir in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.JPEG', '.jpg', '.png')):
                with open(os.path.join(class_path, fname), 'rb') as f:
                    img_bytes = f.read()
                writer.write({
                    'image': img_bytes,
                    'label': class_dir
                })
writer.close()

# Measure overhead
total_size = 0
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))
raw_size = sum(os.path.getsize(os.path.join(DATA_DIR, d, f))
               for d in sorted(os.listdir(DATA_DIR))
               if os.path.isdir(os.path.join(DATA_DIR, d))
               for f in sorted(os.listdir(os.path.join(DATA_DIR, d)))
               if f.endswith(('.JPEG', '.jpg', '.png')))
print(f"MsgPack size: {total_size/1024/1024:.1f} MB")
print(f"Raw size: {raw_size/1024/1024:.1f} MB")
print(f"Overhead: {(total_size/raw_size - 1)*100:.1f}%")

# ============================================================
# BENCHMARK
# ============================================================
dataset = ImageDataset(OUTPUT_DIR)
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

# Random
indices = [random.randint(0, len(dataset)-1) for _ in range(10000)]
start = time.perf_counter()
for i in indices:
    _ = dataset[i]
rand_elapsed = time.perf_counter() - start
print(f"Random (10000): {rand_elapsed:.2f}s -> {10000/rand_elapsed:.1f} samples/s")
