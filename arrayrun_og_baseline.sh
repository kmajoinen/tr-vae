#!/bin/bash

#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --partition=[gpu-v100-16g,gpu-v100-32g,gpu-a100-80g]
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --job-name=spin_sacVae
#SBATCH --exclude=gpu[40,43]
#SBATCH --error=./outfiles/error_%A_%a.err
#SBATCH --output=./outfiles/sac_ae_%A_%a.out
#SBATCH --array=0-4

mkdir -p outputs

module load mamba
source activate ../turst_region_autoencoder/env/

SEEDS=(
  1
  2
  3
  4
  5
)

DOMAIN="finger"
TASK="spin"
MAX_STEP=1100000
SEED=$((SLURM_ARRAY_TASK_ID))

srun /usr/bin/time -f "MaxRSS: %M KB" python3 -u train.py \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat 2 \
    --work_dir ./log \
    --seed $SEED \
    --num_train_steps $MAX_STEP \
    --vae True \
    --wandb_sync \
    > outputs/$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID-beta-$BETA-$BETAPOS.txt
