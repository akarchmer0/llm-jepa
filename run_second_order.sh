#!/bin/bash
# Experiment pipeline for second-order STP (parabolic degeneracy fix).
#
# Three model variants are run per (seed, lbd1, lbd2):
#   1. Regular SFT baseline
#   2. First-order STP (random_span, no trajectory_reg)
#   3. STP + trajectory regularization (both first- and second-order penalties)
#
# After each model is trained, trajectory analysis is run to compare
# hidden-state geometry before and after the fix.

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

run_regular() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}
  model_folder=${9}

  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate} --grad_accum=8
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
}

run_jepa() {
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}
  model_folder=${9}

  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span --grad_accum=8
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
}

run_traj_reg() {
  # STP (random_span) + first-order velocity + second-order acceleration regularization
  base_model_name=${1}
  learning_rate=${2}
  epoch=${3}
  last_token=${4}
  predictors=${5}
  seed=${6}
  lbd=${7}
  dataset=${8}
  model_folder=${9}
  lbd1=${10}
  lbd2=${11}

  echo "Success Rate: traj_reg ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} lbd1=${lbd1} lbd2=${lbd2} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span --trajectory_reg --lbd1=${lbd1} --lbd2=${lbd2} --grad_accum=8
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 | tee -a output.txt
}

run_analysis() {
  model_folder=${1}
  base_model_name=${2}
  dataset=${3}
  label=${4}

  echo "--- Trajectory Analysis: ${label} ---" >> output.txt
  python3 analyze_trajectory.py \
    --model_name=${model_folder} \
    --original_model_name=${base_model_name} \
    --input_file=datasets/${dataset}_test.jsonl \
    --max_examples=200 | tee -a output.txt
}

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

model_name=Qwen/Qwen3-1.7B
dataset=synth
lbd=0.02        # STP (random_span) loss weight — same as run_stp.sh
predictors=0
learning_rate=2e-5
epochs=4
last_token=-3

# Hyperparameter grid for trajectory regularization (3 x 3 = 9 combinations)
lbd1_values=(0.001 0.01 0.1)   # first-order: velocity consistency weight
lbd2_values=(0.001 0.01 0.1)   # second-order: acceleration weight

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

for seed in 82 23 37 84 4
do
  # ----- Baseline 1: Regular SFT ------------------------------------------
  model_folder=ft-r-${learning_rate}-${seed}
  run_regular ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
              ${seed} ${lbd} ${dataset} ${model_folder}
  run_analysis ${model_folder} ${model_name} ${dataset} "regular-seed${seed}"

  # ----- Baseline 2: First-order STP (no trajectory_reg) ------------------
  model_folder=ft-j-${learning_rate}-${lbd}-${predictors}-${seed}
  run_jepa ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
           ${seed} ${lbd} ${dataset} ${model_folder}
  run_analysis ${model_folder} ${model_name} ${dataset} "stp-seed${seed}"

  # ----- Grid: STP + trajectory regularization ----------------------------
  for lbd1 in "${lbd1_values[@]}"
  do
    for lbd2 in "${lbd2_values[@]}"
    do
      model_folder=ft-tr-${learning_rate}-${lbd}-${lbd1}-${lbd2}-${seed}
      run_traj_reg ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
                   ${seed} ${lbd} ${dataset} ${model_folder} ${lbd1} ${lbd2}
      run_analysis ${model_folder} ${model_name} ${dataset} \
                   "traj_reg-lbd1${lbd1}-lbd2${lbd2}-seed${seed}"
    done
  done
done
