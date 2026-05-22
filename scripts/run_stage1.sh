#!/usr/bin/env bash
# Stage 1: Projector Alignment (optimized for 90GB+ Blackwell GPU)
#
# Key optimizations:
#   - TileGym cuTile kernels (RoPE+RMSNorm+SwiGLU+FA4)
#   - FusedCE enabled (beneficial in Stage 1 where only projector trains)
#   - torch.compile max-autotune-no-cudagraphs
#   - PackingCollator (parallel in DataLoader workers, seq_length=512)
#   - Large raw batch (256) to give collator enough samples for tight packing
#   - 16 dataloader workers to feed the pipeline
#   - No gradient checkpointing (unnecessary on 90GB GPU)

set -euo pipefail

VISION_MODEL="${VISION_MODEL:-google/siglip2-base-patch16-224}"
TEXT_MODEL="${TEXT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export STAGE=1

.venv/bin/python scripts/train.py \
    --vision_model_name_or_path "$VISION_MODEL" \
    --text_model_name_or_path "$TEXT_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --use_tilegym \
    --torch_compile \
    --use_packing \
    --seq_length 512 \
    --no_gradient_checkpointing \
    --no_callbacks \
    --bf16 \
    --learning_rate 1e-3 \
    --dataloader_num_workers 16 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --max_eval_samples 50 \
    --seed 42 \
    "$@"
