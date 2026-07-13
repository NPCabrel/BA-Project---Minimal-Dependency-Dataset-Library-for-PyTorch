#!/usr/bin/env python3

"""
GPU Dummy Model Calibration Script
-Measures per-batch compute delay over 1000 batches to characterize GPU stability.
-Records mean, std, coefficient of variation, and anomaly counts (batches exceeding 1.5x mean delay).

Unlike benchmark_gpu.py, this captures instability patterns and slowdowns over time, not just average performance.

Outputs: per-batch CSV + Summary CSV with stability metrics.
usage: python calibrate_dummy.py --model resnet50, --batch-size 256 --optimizer sgd
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import time
import csv
import os
from datetime import datetime
import statistics


#----Arguments

parser = argparse.ArgumentParser(description="Calibrate dummy model measuring GPU stability over 1000 batches")
parser.add_argument("--model", type=str, required=True, help="model name")
parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="Optimizer type")
parser.add_argument("--warmup", type=int, default=50, help="Warmup batches (not recorded)")
parser.add_argument("--anomaly-threshold", type=float, default=1.5, help="Multiplier of mean above which a batch is considered anomalous")
parser.add_argument("--output-dir", type=str, default="/netscratch/nague/gpu_calibration", help="Output directory for CSV")

args = parser.parse_args()


#-----Setups

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
print(f"Model: {args.model}")
print(f"Batch Size: {args.batch_size}")
print(f"Optimizer: {args.optimizer}")
print(f"Batches: 1000 (+{args.warmup} warmup)")


#----Loading our model

try:
	model = getattr(models, args.model)(weights=None)
except AttributeError:
	print(f"Error: model '{args.model}' not found in torchvision.models")
	exit(1)

model = model.cuda()
model.train()



#-----Optimizer
if args.optimizer == "adam":
	optimizer = optim.Adam(model.parameters())
else:
	optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()



#----Warmup

print(f"\nWarmup ({args.warmup} batches)...")
for _ in range(args.warmup):
	images = torch.randn(args.batch_size, 3, 224, 224).cuda()
	labels = torch.randint(0, 1000, (args.batch_size,)).cuda()
	optimizer.zero_grad()
	outputs = model(images)
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()


#----main part, calibration over 1000 batches
NUM_BATCHES = 1000
print(f"\nCalibration ({NUM_BATCHES} batches..).")
results = []

for i in range(NUM_BATCHES):
	images = torch.randn(args.batch_size, 3, 224, 224).cuda()
	labels = torch.randint(0, 1000, (args.batch_size,)).cuda()

	torch.cuda.synchronize()
	start = time.perf_counter()

	optimizer.zero_grad()
	outputs = model(images)
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()

	torch.cuda.synchronize()
	elapsed = time.perf_counter() - start


	results.append(elapsed)

	if(i+1)%100==0:
		avg_so_far = sum(results) / len(results)
		print(f" Batch {i+1}/{NUM_BATCHES}: avg={avg_so_far*1000:.2f}ms, last={elapsed*1000:.2f}ms")




#---statistics
mean_time = statistics.mean(results)
std_time = statistics.stdev(results)
min_time = min(results)
max_time = max(results)
median_time = statistics.median(results)
coeff_var = (std_time/mean_time) * 100

#anomaly detection
threshold = mean_time * args.anomaly_threshold
anomaly_count = sum(1 for t in results if t > threshold)
anomaly_pct = (anomaly_count / NUM_BATCHES) * 100

print(f"\nCalibration results for {args.model} (batch_size={args.batch_size}, opt={args.optimizer}):")
print(f" Mean: {mean_time*1000:.2f} ms")
print(f" Std: {std_time*1000:.2f} ms")
print(f" Coeff Var: {coeff_var:.2f}%")
print(f" Median: {median_time*1000:.2f} ms")
print(f" Min: {min_time*1000:.2f} ms")
print(f" Max: {max_time*1000:.2f} ms")
print(f" Throughput: {args.batch_size/mean_time:.1f} img/s")
print(f" Anomalies: {anomaly_count}/{NUM_BATCHES} ({anomaly_pct:.1f}%) [threshold: {args.anomaly_threshold}x mean = {threshold*1000:.1f}ms]")













# ------------ save results


#Per-Batch CSV
os.makedirs(args.output_dir, exist_ok = True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
gpu_name_clean = gpu_name.replace(" ", "_")
csv_file = os.path.join(
	args.output_dir,
	f"calib_{args.model}_bs{args.batch_size}_nb{NUM_BATCHES}_{args.optimizer}_{gpu_name_clean}_{timestamp}.csv"
)



with open(csv_file, 'w', newline = '') as f:
	writer = csv.writer(f)
	writer.writerow(["model", "batch_size", "optimizer", "gpu", "gpu_mem_gb", "batch_index", "time_s", "time_ms"])
	for i, t in enumerate(results):
		writer.writerow([args.model, args.batch_size, args.optimizer, gpu_name, f"{gpu_mem:.1f}", i+1, f"{t:.6f}", f"{t*1000:.2f}"])
print(f"Per-batch data saved: {csv_file}")


#Summary CSV

summary_file = os.path.join(
	args.output_dir,
	f"calib_summary_{args.model}_bs{args.batch_size}_nb{NUM_BATCHES}_{args.optimizer}_{gpu_name_clean}_{timestamp}.csv"
)

with open(summary_file, 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow(["model", "batch_size", "optimizer", "gpu", "gpu_mem_gb", "num_batches", "mean_ms", "std_ms", "coeff_var_pct", "median_ms", "min_ms", "max_ms", "throughput_img_s", "anomaly_count", "anomaly_pct", "anomaly_threshold"])
	writer.writerow([args.model, args.batch_size, args.optimizer, gpu_name, f"{gpu_mem:.1f}", NUM_BATCHES, f"{mean_time*1000:.2f}", f"{std_time*1000:.2f}", f"{coeff_var:.2f}", f"{median_time*1000:.2f}", f"{min_time*1000:.2f}", f"{max_time*1000:.2f}", f"{args.batch_size/mean_time:.1f}", anomaly_count, f"{anomaly_pct:.1f}", f"{args.anomaly_threshold}x"])
print(f"Summary saved: {summary_file}")
