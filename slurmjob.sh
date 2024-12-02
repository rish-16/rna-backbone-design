#!/bin/bash

#SBATCH --job-name=rnaff_evalsuite_rebuttals
#SBATCH --time 7:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --mem=40G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd ~/project/rna-backbone-design/
module load miniconda
conda activate rnasp
python eval_se3_flows.py
# salloc --gpus=2 --time=2:00:00 --partition gpu_devel
