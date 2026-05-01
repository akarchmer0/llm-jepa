#!/bin/bash
set -e  # abort immediately if any command fails
# Experiment pipeline for second-order STP (parabolic degeneracy fix).
#
# Usage:
#   bash run_second_order.sh          # full run
#   bash run_second_order.sh --debug  # 1/4 epoch training, 100 eval examples
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
  max_steps_arg=${10}
  eval_max=${11}
  analyze_max=${12}

  if [ -d "${model_folder}" ]; then
    echo "SKIP regular ${model_folder} — already exists" | tee -a output.txt
    return 0
  fi
  echo "Success Rate: regular ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} --regular \
    --model_name=${base_model_name} --learning_rate=${learning_rate} --grad_accum=8 \
    --max_steps=${max_steps_arg}
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 \
    --max_examples=${eval_max} | tee -a output.txt
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
  max_steps_arg=${10}
  eval_max=${11}
  analyze_max=${12}

  if [ -d "${model_folder}" ]; then
    echo "SKIP jepa ${model_folder} — already exists" | tee -a output.txt
    return 0
  fi
  echo "Success Rate: jepa ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span --grad_accum=8 --max_steps=${max_steps_arg}
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 \
    --max_examples=${eval_max} | tee -a output.txt
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
  max_steps_arg=${12}
  eval_max=${13}
  analyze_max=${14}

  if [ -d "${model_folder}" ]; then
    echo "SKIP traj_reg ${model_folder} — already exists" | tee -a output.txt
    return 0
  fi
  echo "Success Rate: traj_reg ${base_model_name} lr=${learning_rate} e=${epoch} lt=${last_token} p=${predictors} s=${seed} lbd=${lbd} lbd1=${lbd1} lbd2=${lbd2} dataset=${dataset}" >> output.txt
  torchrun --nproc_per_node=4 stp.py \
    --train_file datasets/${dataset}_train.jsonl \
    --output_dir=${model_folder} --num_epochs=${epoch} --finetune_seed=${seed} \
    --last_token=${last_token} --lbd=${lbd} --predictors=${predictors} \
    --model_name=${base_model_name} --learning_rate=${learning_rate} \
    --linear=random_span --trajectory_reg --lbd1=${lbd1} --lbd2=${lbd2} --grad_accum=8 \
    --max_steps=${max_steps_arg}
  python3 evaluate.py --model_name=${model_folder} \
    --input_file=datasets/${dataset}_test.jsonl --output_file=eval.jsonl --split_tune_untune \
    --original_model_name=${base_model_name} --nosplit_data \
    --spider_path=spider_data/database --max_new_tokens=-1 \
    --max_examples=${eval_max} | tee -a output.txt
}

run_analysis() {
  model_folder=${1}
  base_model_name=${2}
  dataset=${3}
  label=${4}
  analyze_max=${5}

  echo "--- Trajectory Analysis: ${label} ---" >> output.txt
  python3 analyze_trajectory.py \
    --model_name=${model_folder} \
    --original_model_name=${base_model_name} \
    --input_file=datasets/${dataset}_test.jsonl \
    --max_examples=${analyze_max} | tee -a output.txt
}

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

DEBUG=0
for arg in "$@"; do
  if [ "$arg" = "--debug" ]; then
    DEBUG=1
  fi
done

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

model_name=meta-llama/Llama-3.2-1B-Instruct
dataset=synth
lbd=0.02        # STP (random_span) loss weight — same as run_stp.sh
predictors=0
learning_rate=2e-5
last_token=-2

if [ "$DEBUG" = "1" ]; then
  epochs=1          # 1 epoch but with max_steps to cap at 1/4 epoch
  max_steps=63      # 8000 examples / (4 GPUs * 4 batch * 8 grad_accum) = 63 steps/epoch → 1/4 epoch
  eval_max_examples=100
  analyze_max_examples=50
  echo "DEBUG MODE: ~1/4 epoch training, 100 eval examples, 50 trajectory examples"
else
  epochs=4
  max_steps=-1      # no cap
  eval_max_examples=-1
  analyze_max_examples=200
fi

# Hyperparameter grid for trajectory regularization (3 x 3 = 9 combinations)
lbd1_values=(0.0001 0.003 0.007)   # first-order: velocity consistency weight
lbd2_values=(0.0001 0.003 0.007)   # second-order: acceleration weight

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

for seed in 82 23 37 84 4
do
  # ----- Baseline 1: Regular SFT ------------------------------------------
  model_folder=ft-r-${learning_rate}-${seed}
  run_regular ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
              ${seed} ${lbd} ${dataset} ${model_folder} \
              ${max_steps} ${eval_max_examples} ${analyze_max_examples}
  run_analysis ${model_folder} ${model_name} ${dataset} "regular-seed${seed}" \
               ${analyze_max_examples}

  # ----- Baseline 2: First-order STP (no trajectory_reg) ------------------
  model_folder=ft-j-${learning_rate}-${lbd}-${predictors}-${seed}
  run_jepa ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
           ${seed} ${lbd} ${dataset} ${model_folder} \
           ${max_steps} ${eval_max_examples} ${analyze_max_examples}
  run_analysis ${model_folder} ${model_name} ${dataset} "stp-seed${seed}" \
               ${analyze_max_examples}

  # ----- Grid: STP + trajectory regularization ----------------------------
  for lbd1 in "${lbd1_values[@]}"
  do
    for lbd2 in "${lbd2_values[@]}"
    do
      model_folder=ft-tr-${learning_rate}-${lbd}-${lbd1}-${lbd2}-${seed}
      run_traj_reg ${model_name} ${learning_rate} ${epochs} ${last_token} ${predictors} \
                   ${seed} ${lbd} ${dataset} ${model_folder} ${lbd1} ${lbd2} \
                   ${max_steps} ${eval_max_examples} ${analyze_max_examples}
      run_analysis ${model_folder} ${model_name} ${dataset} \
                   "traj_reg-lbd1${lbd1}-lbd2${lbd2}-seed${seed}" \
                   ${analyze_max_examples}
    done
  done
done
