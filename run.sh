#!/bin/bash

# Array of seeds
SEEDS=(
  0
#  1
#  2
#  3
#  4
)

# Configuration variables
DOMAIN="cheetah"
TASK="run"
MAX_STEP=1000000
BETAPOS=0.0001
BETA=1e-7
REPEAT=4
K=1
M=1
L=0
GPU=0

# Loop through the seeds array
for SEED in "${SEEDS[@]}"; do
  echo "Running training for seed $SEED..."
  
  # Execute the Python script with the current seed
  python3 train.py \
      --domain_name $DOMAIN \
      --task_name $TASK \
      --encoder_type pixel \
      --decoder_type pixel \
      --action_repeat $REPEAT \
      --work_dir ./log \
      --seed $SEED \
      --num_train_steps $MAX_STEP \
      --beta $BETA \
      --beta2 $BETAPOS \
      --K $K \
      --M $M \
      --L $L \
      --reward_pred False \
      --proj_name "reward pred" \
      --gpu_num $GPU \
      --vae True \
      --wandb_sync True \
      
      
      
      
      
      
      
  echo "Finished training for seed $SEED."
done

echo "All training runs completed."