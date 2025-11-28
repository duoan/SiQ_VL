#!/bin/bash

# Train Stage 1: Freeze LLM to train VLM projector
# Auto-detects host type and sets appropriate parameters

set -e  # Exit on error

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
        # Try to get instance type from EC2 metadata service
        INSTANCE_TYPE=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "")
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

# Base parameters for stage 1 (freeze LLM)
BASE_ARGS=(
    "--freeze_llm"
    "--output_dir" "./checkpoints/siq_vlm_stage1"
    "--project" "siq_vl_stage_1"
)

# Host-specific parameters
if [[ "$HOST_TYPE" == "macbook" ]]; then
    echo ">>> Configuring for MacBook (quick test mode)"
    
    # MacBook parameters: small dataset, no distributed training
    MACBOOK_ARGS=(
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
        "--fp16"  # Use fp16 if available
        "--logging_steps" "2"
        "--save_steps" "10"
        "--no_distributed"
    )
    
    # Run without accelerate (single process)
    echo ">>> Running training script directly (no distributed training)"
    $PYTHON_CMD "$TRAIN_SCRIPT" "${BASE_ARGS[@]}" "${MACBOOK_ARGS[@]}" "$@"
    
elif [[ "$HOST_TYPE" == "aws_p4d" ]]; then
    echo ">>> Configuring for AWS p4d.24xlarge (full training mode)"
    
    # AWS p4d.24xlarge parameters: full dataset, distributed training
    AWS_ARGS=(
        "--vision_model_name_or_path" "google/siglip2-so400m-patch16-512"
        "--llm_model_name_or_path" "Qwen/Qwen2.5-1.5B-Instruct"
        "--per_device_train_batch_size" "8"
        "--gradient_accumulation_steps" "4"
        "--max_steps" "1000"
        "--num_proc" "96"
        "--dataloader_num_workers" "4"
        "--learning_rate" "1e-3"
        "--bf16"
        "--logging_steps" "10"
        "--save_steps" "500"
    )
    
    # Run with accelerate launcher for distributed training
    echo ">>> Running with accelerate launcher (distributed training)"
    # Use accelerate from venv if available, otherwise use system accelerate
    if [[ -f "$VENV_DIR/bin/accelerate" ]]; then
        ACCELERATE_CMD="$VENV_DIR/bin/accelerate"
    elif command -v accelerate >/dev/null 2>&1; then
        ACCELERATE_CMD="accelerate"
    else
        echo ">>> Error: accelerate not found. Install with: $PYTHON_CMD -m pip install accelerate"
        exit 1
    fi
    
    # Ensure venv's bin is in PATH for accelerate to find the right Python
    export PATH="$VENV_DIR/bin:$PATH"
    $ACCELERATE_CMD launch \
        --dispatch_batches=false \
        --split_batches=false \
        "$TRAIN_SCRIPT" \
        "${BASE_ARGS[@]}" \
        "${AWS_ARGS[@]}" \
        "$@"
    
else
    echo ">>> Warning: Unknown host type. Using default parameters."
    echo ">>> You may want to specify parameters manually."
    
    # Default: assume single GPU, moderate settings
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
    
    $PYTHON_CMD "$TRAIN_SCRIPT" "${BASE_ARGS[@]}" "${DEFAULT_ARGS[@]}" "$@"
fi

