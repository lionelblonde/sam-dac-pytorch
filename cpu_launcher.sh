#!/usr/bin/env bash

python main.py train --cfg="tasks/train.yml" --env_id="Walker2d-v4" --seed=0 --num_demos=4 --expert_path="/Users/lionelblonde/Datasets/mujoco-experts/Walker2d-v4"

# python main.py \
#     --wandb_project ava \
#     --env_id Walker2d-v4 \
#     --seed 0 \
#     --no-cuda \
#     --no-fp16 \
#     --no-mps \
#     --no-render \
#     --no-record \
#     --task "train" \
#     --num_timesteps 10000000 \
#     --training_steps_per_iter 4 \
#     --eval_steps_per_iter 10 \
#     --eval_every 10 \
#     --save_every 10 \
#     --layer_norm \
#     --actor_lr 1e-4 \
#     --critic_lr 1e-4 \
#     --clip_norm 40. \
#     --wd_scale 0.001 \
#     --acc_grad_steps 8 \
#     --rollout_len 2 \
#     --batch_size 128 \
#     --gamma 0.99 \
#     --mem_size 100000 \
#     --noise_type "adaptive-param_0.2" \
#     --pn_adapt_frequency 50. \
#     --polyak 0.005 \
#     --targ_up_freq 100 \
#     --n_step_returns \
#     --lookahead 5 \
#     --no-ret_norm \
#     --no-popart \
#     --expert_path "/Users/lionelblonde/Datasets/mujoco-experts/Walker2d-v4" \
#     --num_demos 4 \
#     --g_steps 3 \
#     --d_steps 1 \
#     --d_lr 0.00001 \
#     --no-state_only \
#     --minimax_only \
#     --ent_reg_scale 0.001 \
#     --spectral_norm \
#     --grad_pen \
#     --grad_pen_targ 1. \
#     --grad_pen_scale 10. \
#     --one_sided_pen \
#     --historical_patching \
#     --wrap_absorb \
#     --no-d_batch_norm
