#!/bin/bash
#SBATCH --job-name=resnet50_1pgu
#SBATCH --partition=RTXA6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00
#SBATCH --output=/netscratch/%u/resnet50_%j.out
#SBATCH --error=/netscratch/%u/resnet50_%j.err

echo "Job started on $(date)"
echo "Node: $(hostname)"

source ~/venv/torch_env/bin/activate
python3 ~/train_resnet50.py

echo "Job ended on $(data)"
