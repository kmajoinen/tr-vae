#!/bin/bash

#SBATCH --mem=70G
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-v100-32g,gpu-v100-16g
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --job-name=spin_TrVae
#SBATCH --exclude=gpu[40,43]
#SBATCH --error=./outfiles/error_%A_%a.err
#SBATCH --output=./outfiles/sac_trvae_%A_%a.out
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
BETAPOS=0.0005
BETA=1e-7
REPEAT=2
echo $DOMAIN $TASK $SEED
srun python3 -u train.py \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --encoder_type pixel \
    --decoder_type pixel \
    --action_repeat $REPEAT \
    --work_dir ./log \
    --seed $SEED \
    --num_train_steps $MAX_STEP \
    --vae True \
    --wandb_sync \
    --beta2 $BETAPOS \
    --proj_name "SAC" \
    > outputs/$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID-beta-$BETA-$BETAPOS.txt
