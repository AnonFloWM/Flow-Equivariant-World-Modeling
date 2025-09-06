#!/usr/bin/env bash
set -euo pipefail

# SCRIPT FOR GENERATING VIS FOR PAPER RESULTS

# helper to clear output dir
prepare_dir () {
  local outdir="$1"
  if [[ -d "$outdir" ]]; then
    echo "Clearing existing output: $outdir"
    rm -rf -- "$outdir"
  fi
  mkdir -p -- "$outdir"
}

# 1
OUTDIR="./data/mnist_world/dynamic_validation_bigworld_biased_200_vis"
prepare_dir "$OUTDIR"
python -m scripts.generate_mnist_world_dynamic \
    --num_examples 8 \
    --output_dir "$OUTDIR" \
    --train \
    --world_size 80 --num_digits 8 --window_size 32 \
    --seq_len 50 --target_seq_len 150 \
    --straightline_biased_rollout --forward_probability 0.95 \
    --step_size 10 \
    --render_full_world --render_full_world_with_cam \
    --num_workers 1 --shard_size 8 --constant_velocity

# 2
OUTDIR="./data/mnist_world/dynamic_validation_biased_200_vis"
prepare_dir "$OUTDIR"
python -m scripts.generate_mnist_world_dynamic \
    --num_examples 8 \
    --output_dir "$OUTDIR" \
    --train \
    --world_size 50 --num_digits 5 --window_size 32 \
    --seq_len 50 --target_seq_len 150 \
    --straightline_biased_rollout --forward_probability 0.95 \
    --step_size 10 \
    --render_full_world --render_full_world_with_cam \
    --num_workers 1 --shard_size 8 --constant_velocity

# 3
OUTDIR="./data/mnist_world/dynamic_validation_smallworld_biased_200_vis"
prepare_dir "$OUTDIR"
python -m scripts.generate_mnist_world_dynamic \
    --num_examples 8 \
    --output_dir "$OUTDIR" \
    --train \
    --world_size 32 --num_digits 3 --window_size 32 \
    --seq_len 50 --target_seq_len 150 \
    --straightline_biased_rollout --forward_probability 0.95 \
    --step_size 10 \
    --render_full_world --render_full_world_with_cam \
    --num_workers 1 --shard_size 8 --constant_velocity

# 4
OUTDIR="./data/mnist_world/dynamic_validation_smallworld_no_em_biased_200_vis"
prepare_dir "$OUTDIR"
python -m scripts.generate_mnist_world_dynamic \
    --num_examples 8 \
    --output_dir "$OUTDIR" \
    --train \
    --world_size 32 --num_digits 3 --window_size 32 \
    --seq_len 50 --target_seq_len 150 \
    --step_size 0 \
    --render_full_world --render_full_world_with_cam \
    --num_workers 1 --shard_size 8

# 5
OUTDIR="./data/mnist_world/static_validation_biased_200_vis"
prepare_dir "$OUTDIR"
python -m scripts.generate_mnist_world \
    --num_examples 8 \
    --output_dir "$OUTDIR" \
    --train \
    --world_size 50 --num_digits 5 --window_size 32 \
    --seq_len 50 --target_seq_len 150 \
    --straightline_biased_rollout --forward_probability 0.95 \
    --step_size 10 \
    --render_full_world --render_full_world_with_cam \
    --num_workers 1 --shard_size 8 --constant_velocity
