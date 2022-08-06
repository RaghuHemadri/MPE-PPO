#!/bin/bash

# Slurm sbatch options
#SBATCH -a 0-2
#SBATCH -o mqmix_%a.out # name the output file
#SBATCH --job-name mqmix
#SBATCH -c 20 # cpus per task

# Loading the required module
source /etc/profile
module load anaconda/2020a

logs_folder="betas"
mkdir -p $logs_folder

env="MPE"
scenario="simple_spread"  # simple_speaker_listener # simple_reference
num_landmarks=3
num_agents=3
algo="rmappo"
exp_names=("beta_4" "beta_3" "beta_2")
seed_max=1
betas=(8e-4 8e-3 8e-2)
mu_coeffs=1
var_coeffs=1
use_reparametrization="True"


python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} \
--experiment_name ${exp_names[$SLURM_ARRAY_TASK_ID]} \
--scenario_name ${scenario} \
--num_agents ${num_agents} --num_landmarks ${num_landmarks} \
--seed ${seed} \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 --episode_length 25 \
--num_env_steps 20000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--wandb_name "raghu-ops" --user_name "raghu-ops" \
--use_reparametrization ${use_reparametrization} \
--beta ${betas[$SLURM_ARRAY_TASK_ID]} \
--mu_coef ${mu_coeffs} --var_coef ${var_coeffs} \
&> $logs_folder/out_${exp_names[$SLURM_ARRAY_TASK_ID]}