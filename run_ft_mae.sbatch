#!/bin/bash
#SBATCH --job-name=FT_MAE
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --partition=day
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=node-gpu-03
#SBATCH --account=member
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32

PROJ_HOME="/home/rawangq1/git_wq/SSL_veronika"

CUDA_VISIBLE_DEVICES=1
source $PROJ_HOME/.venv/bin/activate

srun python $PROJ_HOME/notebooks/finetune_ViT_mae.py
