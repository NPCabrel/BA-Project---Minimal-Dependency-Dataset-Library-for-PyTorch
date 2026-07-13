#!/usr/bin/env python3
"""Write tiny-imagenet to LMDB format."""
import os
import lmdb
import pickle
from PIL import Image

DATA_DIR = "/fscratch/nague/tiny-imagenet/train"
OUTPUT_DIR = "/fscratch/nague/storage_benchmarks/lmdb"
MAP_SIZE = 50 * 1024 * 1024 * 1024  # 50 GB max

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all samples
samples = []
for class_dir in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for fname in sorted(os.listdir(class_path)):
            if fname.endswith(('.JPEG', '.jpg', '.png')):
                # Extract label from class name (like n01443537)
                label = class_dir
                samples.append((os.path.join(class_path, fname), label))

print(f"Writing {len(samples)} samples to LMDB...")

env = lmdb.open(OUTPUT_DIR, map_size=MAP_SIZE)

with env.begin(write=True) as txn:
    for idx, (img_path, label) in enumerate(samples):
        # Read image bytes
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        
        # Serialize: (image_bytes, label)
        value = pickle.dumps((img_bytes, label))
        key = str(idx).encode('utf-8')
        txn.put(key, value)
        
        if (idx + 1) % 10000 == 0:
            print(f"  {idx+1}/{len(samples)}")

env.close()

# Measure size
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
                 for f in os.listdir(OUTPUT_DIR))
print(f"LMDB size: {total_size / 1024 / 1024:.1f} MB")
print("Done!")
