#!/bin/bash
set -x

# Ensure the virtual environment's binaries (such as ninja) are on PATH. This
# avoids issues where PyTorch cannot find the `ninja` executable when building
# extensions.
export PATH="$(dirname "$0")/.venv/bin:$PATH"
export NINJA="$(dirname "$0")/.venv/bin/ninja"

.venv/bin/python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.4 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --pretrain Qwen/Qwen3-4B \
   --agent_func_path training/agent_func.py \
   --save_path training/ckpt/qwen3-4b-obsidian \
   --ckpt_path training/ckpt/qwen3-4b-obsidian \
   --advantage_estimator reinforce \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 8 \
   --max_epochs 50 \
   --prompt_max_len 8192 \
   --max_samples 100000 \
   --generate_max_len 8192 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --prompt_data json@data/openrlhf \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb True \
   --wandb_project obsidian-retrieval-openrlhf
