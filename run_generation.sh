#!/bin/bash
#SBATCH --job-name=qwen_aoc_generation
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --partition=gpushort

echo "Starting job on $(hostname)"
echo "Time: $(date)"

set -e

module load Python/3.13.5-GCCcore-14.3.0
module load uv

cd ~/scriptie/habrok

source .env
uv run generation/main.py

echo "Finished at $(date)"
