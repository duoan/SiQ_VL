#!/bin/bash

# Unified launcher for SiQ_VL training (stage 1 & stage 2)
# - Handles host detection, venv setup, accelerate launch, etc.
# - Stage-specific configs (freeze_llm, LR, max_samples, etc.) are selected by $STAGE.
#
# Usage (from project root):
#   STAGE=1 bash scripts/train_launch.sh [extra args...]
#   STAGE=2 bash scripts/train_launch.sh [extra args...]
#
# The thin scripts train_stage_1.sh and train_stage_2.sh just set STAGE and forward args.

set -e

if [[ -z "$STAGE" ]]; then
    echo ">>> Error: STAGE environment variable not set. Use STAGE=1 or STAGE=2."
    exit 1
fi

if [[ "$STAGE" != "1" && "$STAGE" != "2" ]]; then
    echo ">>> Error: STAGE must be '1' or '2', got '$STAGE'."
    exit 1
fi

echo ">>> Using training stage: $STAGE"

# Detect host type
detect_host() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macbook"
    elif [[ -n "${EC2_INSTANCE_TYPE}" ]]; then
        # Check EC2_INSTANCE_TYPE environment variable
        if [[ "${EC2_INSTANCE_TYPE}" == *"p4d.24xlarge"* ]]; then
            echo "aws_p4d"
        else
            echo "aws_other"
        fi
    elif command -v curl >/dev/null 2>&1; then
        # Try to get instance type from EC2 metadata service using IMDSv2
        TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" \
                    -s --max-time 2 2>/dev/null || echo "")

        if [[ -n "$TOKEN" ]]; then
            INSTANCE_TYPE=$(curl -s --max-time 2 \
                -H "X-aws-ec2-metadata-token: $TOKEN" \
                http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "")
        else
            INSTANCE_TYPE=""
        fi

        if [[ "$INSTANCE_TYPE" == *"p4d.24xlarge"* ]]; then
            echo "aws_p4d"
        elif [[ -n "$INSTANCE_TYPE" ]]; then
            echo "aws_other"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

HOST_TYPE=$(detect_host)
echo ">>> Detected host type: $HOST_TYPE"

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# Change to project root
cd "$PROJECT_ROOT"

# Train script path (relative to project root)
TRAIN_SCRIPT="scripts/train.py"

# Setup virtual environment using uv
if [[ ! -d "$VENV_DIR" ]]; then
    echo ">>> Virtual environment not found. Creating with uv sync..."
    if ! command -v uv >/dev/null 2>&1; then
        echo ">>> Error: uv not found. Please install uv first."
        echo ">>> Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    uv sync
fi

# Use Python from virtual environment
if [[ -f "$VENV_DIR/bin/python3" ]]; then
    PYTHON_CMD="$VENV_DIR/bin/python3"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
    PYTHON_CMD="$VENV_DIR/bin/python"
else
    echo ">>> Error: Python not found in .venv. Try running 'uv sync' manually."
    exit 1
fi

echo ">>> Using Python: $PYTHON_CMD"

# Stage-specific base parameters
if [[ "$STAGE" == "1" ]]; then
    # Stage 1: freeze LLM, train projector
    BASE_ARGS=(
        "--freeze_llm"
        "--output_dir" "./checkpoints"
        # Keep default project name from train.py ("siq-vl")
    )
else
    # Stage 2: unfreeze LLM, full finetuning
    BASE_ARGS=(
        "--no_freeze_llm"
        "--output_dir" "./checkpoints"
    )
fi

# Host- and stage-specific parameters
if [[ "$HOST_TYPE" == "macbook" ]]; then
    echo ">>> Configuring for MacBook (quick test / sanity mode)"

    if [[ "$STAGE" == "1" ]]; then
        MAC_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-base-patch16-224"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-0.5B-Instruct"
            "--sub_sets" "sharegpt4v(knowledge)"  # Only use knowledge subset for quick testing
            "--max_samples" "1000"
            "--per_device_train_batch_size" "2"
            "--gradient_accumulation_steps" "2"
            "--max_steps" "20"
            "--num_proc" "2"
            "--dataloader_num_workers" "1"
            "--learning_rate" "1e-3"
            "--no_bf16"  # MacBook typically doesn't support bf16
            "--fp16"     # Use fp16 if available
            "--logging_steps" "2"
            "--save_steps" "10"
            "--no_distributed"
        )
    else
        MAC_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-base-patch16-224"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-0.5B-Instruct"
            "--sub_sets" "sharegpt4v(knowledge)"  # Quick VQA-like subset
            "--max_samples" "1000"
            "--per_device_train_batch_size" "2"
            "--gradient_accumulation_steps" "2"
            "--max_steps" "20"
            "--num_proc" "2"
            "--dataloader_num_workers" "1"
            "--learning_rate" "2e-5"
            "--no_bf16"  # MacBook typically doesn't support bf16
            "--fp16"     # Use fp16 if available
            "--logging_steps" "2"
            "--save_steps" "10"
            "--no_distributed"
        )
    fi

    echo ">>> Running training script directly (no distributed training)"
    $PYTHON_CMD "$TRAIN_SCRIPT" "${BASE_ARGS[@]}" "${MAC_ARGS[@]}" "$@"

elif [[ "$HOST_TYPE" == "aws_p4d" ]]; then
    echo ">>> Configuring for AWS p4d.24xlarge (full training mode)"

    if [[ "$STAGE" == "1" ]]; then
        AWS_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-so400m-patch16-512"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-1.5B-Instruct"
            # reduce to 256 tokens
            "--pixel_shuffle_factor" "2"
            "--per_device_train_batch_size" "4"
            "--gradient_accumulation_steps" "4"
            "--max_steps" "1000"
            "--num_proc" "96"
            "--dataloader_num_workers" "4"
            "--learning_rate" "1e-3"
            "--bf16"
            "--logging_steps" "10"
            "--save_steps" "500"
            "--gen_eval_interval" "200"
            "--push_to_hub"
        )
    else
        # Stage 2: large-scale finetuning with ~1M VQA samples, auto max_steps
        AWS_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-so400m-patch16-512"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-1.5B-Instruct"
            # reduce to 256 tokens
            "--pixel_shuffle_factor" "2" 
            "--per_device_train_batch_size" "4"
            "--gradient_accumulation_steps" "4"
            "--max_samples" "1000000"
            "--num_proc" "96"
            "--dataloader_num_workers" "4"
            "--learning_rate" "2e-5"
            "--bf16"
            "--logging_steps" "20"
            "--save_steps" "1000"
            "--gen_eval_interval" "1000"
            "--push_to_hub"
        )
    fi

    echo ">>> Running with accelerate launcher (distributed training)"
    if [[ -f "$VENV_DIR/bin/accelerate" ]]; then
        ACCELERATE_CMD="$VENV_DIR/bin/accelerate"
    elif command -v accelerate >/dev/null 2>&1; then
        ACCELERATE_CMD="accelerate"
    else
        echo ">>> Error: accelerate not found. Install with: $PYTHON_CMD -m pip install accelerate"
        exit 1
    fi

    export PATH="$VENV_DIR/bin:$PATH"
    $ACCELERATE_CMD launch \
        "$TRAIN_SCRIPT" \
        "${BASE_ARGS[@]}" \
        "${AWS_ARGS[@]}" \
        "$@"

else
    echo ">>> Warning: Unknown host type. Using default parameters."
    echo ">>> You may want to specify parameters manually."

    if [[ "$STAGE" == "1" ]]; then
        DEFAULT_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-base-patch16-224"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-0.5B-Instruct"
            "--per_device_train_batch_size" "4"
            "--gradient_accumulation_steps" "4"
            "--max_steps" "1000"
            "--num_proc" "8"
            "--dataloader_num_workers" "2"
            "--learning_rate" "1e-3"
            "--bf16"
            "--logging_steps" "10"
            "--save_steps" "500"
            "--no_distributed"
        )
    else
        DEFAULT_ARGS=(
            "--vision_model_name_or_path" "google/siglip2-base-patch16-224"
            "--llm_model_name_or_path" "Qwen/Qwen2.5-0.5B-Instruct"
            "--per_device_train_batch_size" "4"
            "--gradient_accumulation_steps" "4"
            "--num_proc" "8"
            "--dataloader_num_workers" "2"
            "--learning_rate" "2e-5"
            "--bf16"
            "--logging_steps" "10"
            "--save_steps" "500"
            "--no_distributed"
        )
    fi

    $PYTHON_CMD "$TRAIN_SCRIPT" "${BASE_ARGS[@]}" "${DEFAULT_ARGS[@]}" "$@"
fi


