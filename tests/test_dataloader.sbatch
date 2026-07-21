#!/bin/bash
#SBATCH --job-name=test_dataloader
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/netscratch/%u/test_dataloader_%j.out
#SBATCH --error=/netscratch/%u/test_dataloader_%j.err

source ~/venv/torch_env/bin/activate
python3 -u ~/bachelor-project/tests/test_dataloader.py
