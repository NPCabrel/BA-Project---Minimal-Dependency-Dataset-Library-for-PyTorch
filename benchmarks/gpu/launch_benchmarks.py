#!/usr/bin/env python3
"""
Generate and submit SLURM jobs for all GPU benchmarks.
"""


import subprocess
import os



MODELS = [
	"resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vit_b_16",
	"vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14", "swin_t", "swin_s", "swin_b",
	"swin_v2_t", "swin_v2_s", "swin_v2_b",
]

OPTIMIZERS = ["adam", "sgd"]
NUM_BATCHES = 200
OUTPUT_DIR = "/netscratch/nague/gpu_benchmarks"


GPU_CONFIGS = {
	"A100-40GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
	"A100-80GB": [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
	"A100-PCI": [32, 64, 128, 256, 384, 512, 768, 1024, 1280],
	"B200":     [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072],
	"H100":     [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
	"H100-PCI": [32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536],
	"H200":     [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
	"H200-PCI": [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560],
	"L40S":     [32, 64, 128, 192, 256, 384, 512, 640, 768],
	"RTX3090":  [32, 64, 128, 192, 256, 320],
	"RTXA6000": [32, 64, 128, 192, 256, 384, 512, 640, 768],
	"RTXB6000": [32, 64, 128, 192, 256, 320],
}



os.makedirs(OUTPUT_DIR, exist_ok=True)
total_jobs = 0

for partition, batch_sizes in GPU_CONFIGS.items():
	for model in MODELS:
		for bs in batch_sizes:
			for opt in OPTIMIZERS:
				job_name = f"b_{model[:6]}_bs{bs}_{opt[:3]}"
				script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output={OUTPUT_DIR}/slurm_%J.out
#SBATCH --error={OUTPUT_DIR}/slurm_%j.err

source ~/venv/torch_env/bin/activate
python3 ~/bachelor-project/benchmark_gpu.py --model {model} --batch-size {bs} --optimizer {opt} --num-batches {NUM_BATCHES}
"""


				script_path = f"/tmp/bench_{model}_{bs}_{opt}_{partition}.sbatch"
				with open(script_path, 'w') as f:
					f.write(script)
				result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
				if result.returncode == 0:
					print(f"OK: {job_name} -> {result.stdout.strip().split()[-1]}({partition})")
					total_jobs += 1
				else:
					print(f"FAIL: {job_name} -> {result.stderr.strip()}")
				os.remove(script_path)

print(f"\nTotal jobs submitted: {total_jobs}")
