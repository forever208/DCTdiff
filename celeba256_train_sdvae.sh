#!/usr/bin/env bash

set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate uvit
export TORCH_HOME=/home/ningmang/torch_cache

PROJECT_DIR=/home/ningmang/DCTdiff
OUTPUT_DIR="$PROJECT_DIR/exps/celeba256_f8_sdvae_80k_stdx1"

# log and error settings
LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

cd "$PROJECT_DIR"

accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 \
train_ldm.py \
--config=configs/celeba256_uvit_small_f8.py \
--workdir="$OUTPUT_DIR"

# sslaunch submit -j uvit -c zhongwei -q stream -n 1 --no-log -- bash celeba256_train_sdvae.sh
