#!/usr/bin/env bash
# Stage 2: Full LLM Fine-tuning (optimized for 90GB+ Blackwell GPU)
#
# Key optimizations:
#   - TileGym cuTile kernels (RoPE+RMSNorm+SwiGLU+FA4)
#   - FusedCE enabled (saves 31% GPU time on CE loss)
#   - torch.compile max-autotune-no-cudagraphs (~5% boost on top of TileGym)
#   - PackingCollator (parallel in DataLoader workers, seq_length=1024)
#   - Raw batch=64 (collator packs into ~20 sequences of 1024 tokens each)
#   - No gradient checkpointing (GPU memory allows it)
#   - 16 dataloader workers

set -euo pipefail

STAGE1_CHECKPOINT="${1:?Usage: $0 <stage1_checkpoint_path> [extra args...]}"
shift

OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export STAGE=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python scripts/train.py \
    --stage_1_checkpoint_path "$STAGE1_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --no_freeze_text_model \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --use_tilegym \
    --torch_compile \
    --use_packing \
    --seq_length 1024 \
    --no_gradient_checkpointing \
    --no_callbacks \
    --bf16 \
    --learning_rate 2e-5 \
    --dataloader_num_workers 16 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --max_eval_samples 50 \
    --seed 42 \
    "$@"
