#!/bin/bash
set -x
# Clear compiled Torch extension cache.
# Qwen3-MoE models compile custom Triton kernels. Stale artifacts in
# ~/.cache/torch_extensions can cause hangs during vLLM initialization.
# Removing this cache before launching training forces a clean build【98320054399522†L207-L327】.
rm -rf ~/.cache/torch_extensions
export PATH="$(dirname "$0")/.venv/bin:$PATH"
export NINJA="$(dirname "$0")/.venv/bin/ninja"

# NCCL stuff
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=900

# OpenRLHF stuff
export OPENRLHF_ASYNC_NUM_TASKS=128
export OPENRLHF_ASYNC_QUEUE_SIZE=2

# Read config from JSON file
CONFIG_FILE="$(dirname "$0")/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Parse JSON config using Python
CONFIG_VALUES=$(python3 -c "
import json
import sys
import os

config_file = '$CONFIG_FILE'
with open(config_file, 'r') as f:
    config = json.load(f)

# Extract hyperparameters
hp = config['hyperparameters']
model_name = config['model']['name']

# Extract model identifier (e.g., 'Qwen/Qwen2.5-Coder-7B-Instruct' -> 'qwen2.5-coder-7b-instruct')
model_id = model_name.split('/')[-1].lower().replace('-', '-')

# Construct dynamic path
path_suffix = f\"{model_id}-obsidian-{hp['actor_learning_rate']}-{hp['critic_learning_rate']}-{hp['max_epochs']}epochs-{hp['num_episodes']}episodes\"

# Output all values as shell variables
print(f'MODEL_NAME=\"{model_name}\"')
print(f'INIT_KL_COEF={hp[\"init_kl_coef\"]}')
print(f'KL_TARGET={hp[\"kl_target\"]}')
print(f'KL_HORIZON={hp[\"kl_horizon\"]}')
print(f'MAX_EPOCHS={hp[\"max_epochs\"]}')
print(f'ACTOR_LR={hp[\"actor_learning_rate\"]}')
print(f'CRITIC_LR={hp[\"critic_learning_rate\"]}')
print(f'NUM_EPISODES={hp[\"num_episodes\"]}')
print(f'ADVANTAGE_ESTIMATOR=\"{hp[\"advantage_estimator\"]}\"')
print(f'SAVE_PATH=\"training/ckpt/{path_suffix}\"')
print(f'CKPT_PATH=\"training/ckpt/{path_suffix}\"')
")

# Evaluate the config values as shell variables
eval "$CONFIG_VALUES"

echo "Loaded configuration:"
echo "  Model: $MODEL_NAME"
echo "  Save/Checkpoint Path: $SAVE_PATH"
echo "  Hyperparameters: init_kl_coef=$INIT_KL_COEF, kl_target=$KL_TARGET, max_epochs=$MAX_EPOCHS, etc."

# Launch PPO training with Ray.
.venv/bin/python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 8 \
   --vllm_gpu_memory_utilization 0.25 \
   --colocate_all_models \
   --init_kl_coef $INIT_KL_COEF \
   --kl_target $KL_TARGET \
   --kl_horizon $KL_HORIZON \
   --gamma 0.99 \
   --kl_estimator k3 \
   --pretrain $MODEL_NAME \
   --agent_func_path training/agent_func.py \
   --save_path $SAVE_PATH \
   --ckpt_path $CKPT_PATH \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 4 \
   --n_samples_per_prompt 2 \
   --max_epochs $MAX_EPOCHS \
   --prompt_max_len 2048 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --ds_tensor_parallel_size 8 \
   --bf16 \
   --actor_learning_rate $ACTOR_LR \
   --critic_learning_rate $CRITIC_LR \
   --prompt_data json@data/openrlhf \
   --input_key context_messages \
   --label_key label \
   --apply_chat_template \
   --use_kl_loss \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --deepspeed_enable_sleep \
   --use_wandb True \
   --num_episodes $NUM_EPISODES \
   --save_steps 1 \
   --packing_samples --flash_attn \
   --use_liger_kernel \
   --wandb_project obsidian-retrieval-openrlhf \
   --eps_clip 0.1 \
   --ptx_coef 0.13 \
   --advantage_estimator $ADVANTAGE_ESTIMATOR \
   --async_train \
   --policy_loss_type gspo
#  --aux_loss_coef 0.001 \