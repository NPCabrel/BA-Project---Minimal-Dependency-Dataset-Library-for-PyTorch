#!/bin/bash
#SBATCH --job-name=write_lmdb
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/netscratch/%u/write_lmdb.out
#SBATCH --error=/netscratch/%u/write_lmdb.err

source ~/venv/torch_env/bin/activate
python3 ~/bachelor-project/write_lmdb.py
