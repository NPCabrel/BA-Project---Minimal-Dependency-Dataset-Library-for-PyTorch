#!/usr/bin/env python3
"""Submit missing benchmarks as SLURM job arrays (one array per GPU partition)."""
import subprocess
import os

GPU_CONFIGS = {
    "A100-40GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
    "A100-80GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "A100-PCI":  [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
    "B200":      [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072],
    "H100":      [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "H100-PCI":  [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
    "H200":      [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
    "H200-PCI":  [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
    "L40S":      [32, 64, 128, 192, 256, 384, 512, 640, 768],
    "RTX3090":   [32, 64, 128, 192, 256, 320],
    "RTXA6000":  [32, 64, 128, 192, 256, 384, 512, 640, 768],
    "RTXB6000":  [32, 64, 128, 192, 256, 320],
}

MODELS = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "swin_t", "swin_s", "swin_b",
    "swin_v2_t", "swin_v2_s", "swin_v2_b",
]

OPTIMIZERS = ["adam", "sgd"]
NUM_BATCHES = 200
OUTPUT_DIR = "/netscratch/nague/gpu_benchmarks"

for partition, batch_sizes in GPU_CONFIGS.items():
    # Build task list for this GPU
    tasks = []
    for model in MODELS:
        for bs in batch_sizes:
            for opt in OPTIMIZERS:
                tasks.append(f"{model} {bs} {opt}")
    
    if len(tasks) == 0:
        continue
    
    # Create array script
    script = f"""#!/bin/bash
#SBATCH --job-name=bench_{partition}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-{len(tasks)-1}
#SBATCH --output={OUTPUT_DIR}/slurm_%A_%a.out
#SBATCH --error={OUTPUT_DIR}/slurm_%A_%a.err

# Parse task list
TASKS=(
"""
    for t in tasks:
        script += f'"{t}"\n'
    script += f""")

# Get this array task's parameters
TASK=${{TASKS[$SLURM_ARRAY_TASK_ID]}}
MODEL=$(echo $TASK | cut -d' ' -f1)
BS=$(echo $TASK | cut -d' ' -f2)
OPT=$(echo $TASK | cut -d' ' -f3)

source ~/venv/torch_env/bin/activate
python3 ~/bachelor-project/benchmark_gpu.py --model $MODEL --batch-size $BS --optimizer $OPT --num-batches {NUM_BATCHES}
"""
    
    script_path = f"/tmp/array_{partition}.sbatch"
    with open(script_path, 'w') as f:
        f.write(script)
    
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"OK: {partition} ({len(tasks)} tasks) -> {job_id}")
    else:
        print(f"FAIL: {partition} -> {result.stderr.strip()}")
    os.remove(script_path)
