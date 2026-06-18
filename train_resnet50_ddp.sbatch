#!/bin/bash
#SBATCH --job-name=resnet50_ddp
#SBATCH --partition=RTXA6000
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/netscratch/%u/ddp_%j.out
#SBATCH --error=/netscratch/%u/ddp_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"

source ~/venv/torch_env/bin/activate

# Lancer avec torchrun (1 processus par GPU)
torchrun --nproc_per_node=2 --nnodes=1 ~/train_resnet50_ddp.py

echo "Job ended: $(date)"
