#!/bin/bash
#SBATCH --job-name=bench_parquet
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/netscratch/%u/parquet_%j.out
#SBATCH --error=/netscratch/%u/parquet_%j.err

source ~/venv/torch_env/bin/activate
python3 ~/bachelor-project/benchmark_parquet.py
