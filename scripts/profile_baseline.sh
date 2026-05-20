#!/bin/bash
# =============================================================================
# SiQ-VL Baseline Profiling Launcher
#
# Two modes:
#   1. torch.profiler only (default):
#        bash scripts/profile_baseline.sh
#
#   2. nsys + torch.profiler (full CUDA kernel-level trace):
#        USE_NSYS=1 bash scripts/profile_baseline.sh
#
# Output goes to docs/traces/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

# Use Python from venv
if [[ -f "$VENV_DIR/bin/python3" ]]; then
    PYTHON_CMD="$VENV_DIR/bin/python3"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
    PYTHON_CMD="$VENV_DIR/bin/python"
else
    echo ">>> Error: Python not found in .venv. Run 'uv sync' first."
    exit 1
fi

echo ">>> Using Python: $PYTHON_CMD"

# Ensure output directory exists
mkdir -p docs/traces docs/figs

# Default profiling args (override via CLI)
PROFILE_ARGS=(
    "--stage" "${STAGE:-1}"
    "--vision_model_name_or_path" "${VISION_MODEL:-google/siglip2-so400m-patch16-512}"
    "--text_model_name_or_path" "${TEXT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    "--pixel_shuffle_factor" "${PIXEL_SHUFFLE:-4}"
    "--data_path" "${DATA_PATH:-HuggingFaceM4/FineVision}"
    "--sub_sets" "${SUB_SETS:-sharegpt4v(coco)}"
    "--max_samples" "${MAX_SAMPLES:-2000}"
    "--num_proc" "${NUM_PROC:-8}"
    "--per_device_train_batch_size" "${BATCH_SIZE:-4}"
    "--gradient_accumulation_steps" "${GRAD_ACCUM:-4}"
    "--dataloader_num_workers" "${NUM_WORKERS:-4}"
    "--warmup_steps" "${WARMUP_STEPS:-5}"
    "--profile_steps" "${PROFILE_STEPS:-20}"
    "--output_dir" "docs/traces"
    "--trace_name" "${TRACE_NAME:-iter_0_baseline}"
)

if [[ "${USE_NSYS:-0}" == "1" ]]; then
    # =========================================================================
    # Mode: nsys profiling (captures CUDA kernels, memory, NVTX ranges)
    # =========================================================================
    if ! command -v nsys >/dev/null 2>&1; then
        echo ">>> Error: nsys not found. Install NVIDIA Nsight Systems."
        echo ">>> On Ubuntu: apt install nsight-systems"
        exit 1
    fi

    NSYS_OUTPUT="docs/traces/${TRACE_NAME:-iter_0_baseline}_nsys"
    echo ">>> Running with nsys profiler..."
    echo ">>> nsys output: ${NSYS_OUTPUT}.nsys-rep"

    nsys profile \
        --output "$NSYS_OUTPUT" \
        --force-overwrite true \
        --trace cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage true \
        --capture-range cudaProfilerApi \
        --capture-range-end stop \
        --stats true \
        $PYTHON_CMD scripts/profile_baseline.py "${PROFILE_ARGS[@]}" "$@"

    echo ""
    echo ">>> nsys trace saved to: ${NSYS_OUTPUT}.nsys-rep"
    echo ">>> Open with: nsys-ui ${NSYS_OUTPUT}.nsys-rep"
    echo ">>> Or export to SQLite: nsys export --type sqlite ${NSYS_OUTPUT}.nsys-rep"

else
    # =========================================================================
    # Mode: torch.profiler only (lighter weight, produces Chrome trace)
    # =========================================================================
    echo ">>> Running with torch.profiler only..."
    echo ">>> (Set USE_NSYS=1 to enable full nsys profiling)"
    echo ""

    $PYTHON_CMD scripts/profile_baseline.py "${PROFILE_ARGS[@]}" "$@"
fi

echo ""
echo ">>> Profiling complete!"
echo ">>> Results in docs/traces/"
echo ">>> View Chrome traces at: https://ui.perfetto.dev"
