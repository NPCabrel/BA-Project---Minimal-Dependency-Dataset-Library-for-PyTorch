#!/usr/bin/env python3
"""
Build delay config JSON from benchmark summary CSVs.
Averages delay across all models for each GPU at batch_size=256 with SGD.
Maps real GPU names to clean partition names.
"""
import os
import csv
import json
from collections import defaultdict

SUMMARY_DIR = "/netscratch/nague/gpu_benchmarks"
OUTPUT_FILE = "/home/nague/bachelor-project/gpu_delays.json"

# Map real GPU names (from torch.cuda) to clean partition names (what user types)
GPU_NAME_MAP = {
    "NVIDIA RTX A6000": "RTXA6000",
    "NVIDIA GeForce RTX 3090": "RTX3090",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTXB6000",
    "NVIDIA L40S": "L40S",
    "NVIDIA A40": "A40",
    "NVIDIA A100 80GB PCIe": "A100-80GB",
    "NVIDIA A100-PCIE-40GB": "A100-40GB",
    "NVIDIA A100-SXM4-40GB": "A100-40GB",
    "NVIDIA A100-PCI": "A100-PCI",
    "NVIDIA H100": "H100",
    "NVIDIA H100 NVL": "H100",
    "NVIDIA H100 PCIe": "H100-PCI",
    "NVIDIA H200": "H200",
    "NVIDIA H200 NVL": "H200",
    "NVIDIA H200 PCIe": "H200-PCI",
    "NVIDIA B200": "B200",
}

def clean_gpu_name(name):
    """Convert real GPU name to clean partition name."""
    if name in GPU_NAME_MAP:
        return GPU_NAME_MAP[name]
    # Try partial match
    for real, clean in GPU_NAME_MAP.items():
        if real.lower().replace(" ", "") in name.lower().replace(" ", ""):
            return clean
    # Fallback: keep original but remove spaces
    return name.replace(" ", "_")

# Collect all summary data
gpu_data = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(SUMMARY_DIR):
    if not fname.startswith("summary_") or not fname.endswith(".csv"):
        continue
    
    filepath = os.path.join(SUMMARY_DIR, fname)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            raw_gpu = row["gpu"]
            gpu = clean_gpu_name(raw_gpu)
            bs = int(row["batch_size"])
            opt = row["optimizer"]
            mean_ms = float(row["mean_ms"])
            std_ms = float(row["std_ms"])
            
            # Only use batch_size=256 and SGD for default config
            if bs == 256 and opt == "sgd":
                gpu_data[gpu][model].append({
                    "mean_ms": mean_ms,
                    "std_ms": std_ms
                })

# Average across models for each GPU
config = {}
for gpu, models in sorted(gpu_data.items()):
    all_means = []
    all_stds = []
    model_entries = {}
    
    for model, entries in models.items():
        avg_mean = sum(e["mean_ms"] for e in entries) / len(entries)
        avg_std = sum(e["std_ms"] for e in entries) / len(entries)
        model_entries[model] = {
            "mean_ms": round(avg_mean, 2),
            "std_ms": round(avg_std, 2)
        }
        all_means.append(avg_mean)
        all_stds.append(avg_std)
    
    config[gpu] = {
        "models": model_entries,
        "default_mean_ms": round(sum(all_means) / len(all_means), 2) if all_means else None,
        "default_std_ms": round(sum(all_stds) / len(all_stds), 2) if all_stds else None,
        "num_models_benchmarked": len(model_entries)
    }

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Config saved to {OUTPUT_FILE}")
print(f"GPUs: {len(config)}")
for gpu, data in config.items():
    print(f"  {gpu}: {data['num_models_benchmarked']} models, default={data['default_mean_ms']}ms")
