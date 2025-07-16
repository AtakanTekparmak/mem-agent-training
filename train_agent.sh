#!/bin/bash
set -x
export PATH="$(dirname "$0")/.venv/bin:$PATH"
export NINJA="$(dirname "$0")/.venv/bin/ninja"

.venv/bin/python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 4 \
   --vllm_gpu_memory_utilization 0.20 \
   --colocate_all_models \
   --init_kl_coef 0.04 \
   --kl_target 0.07 \
   --kl_horizon 2000 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --pretrain Qwen/Qwen3-8B \
   --agent_func_path training/agent_func.py \
   --save_path training/ckpt/qwen3-8b-obsidian-1e-6-5e-6-30epochs-4episodes \
   --ckpt_path training/ckpt/qwen3-8b-obsidian-1e-6-5e-6-30epochs-4episodes \
   --advantage_estimator reinforce \
   --save_hf_ckpt \
   --micro_train_batch_size 2 \
   --train_batch_size 16 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 16 \
   --n_samples_per_prompt 4 \
   --max_epochs 30 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-6 \
   --critic_learning_rate 5e-6 \
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
   --num_episodes 4 \
   --save_steps 1 \
   --packing_samples --flash_attn \
   --wandb_project obsidian-retrieval-openrlhf