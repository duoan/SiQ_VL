#!/usr/bin/env bash
# Full benchmark suite: runs all optimization configs on a single backbone.
# Usage:
#   bash scripts/run_full_benchmark.sh small   # SigLIP2-base-224 + Qwen2.5-0.5B
#   bash scripts/run_full_benchmark.sh large   # SigLIP2-so400m + Qwen2.5-1.5B

set -euo pipefail

MODEL="${1:-small}"
STEPS=20
WARMUP=5
OUT_DIR="docs/traces/benchmark_${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "  Full Benchmark Suite: model=${MODEL}"
echo "  Output: ${OUT_DIR}"
echo "========================================================================"

PYTHON="${PYTHON:-.venv/bin/python}"
CMD="${PYTHON} scripts/benchmark_real_efficiency.py --model ${MODEL} --steps ${STEPS} --warmup ${WARMUP}"

echo ""
echo ">>> [1/12] Baseline: FP32 bug, B=4, no optimizations"
$CMD --force_fp32 --batch_size 4 --output_json "${OUT_DIR}/01_baseline_fp32.json"

echo ""
echo ">>> [2/12] BF16 fix + no_grad, B=4"
$CMD --batch_size 4 --output_json "${OUT_DIR}/02_bf16_b4.json"

echo ""
echo ">>> [3/12] BF16, B=16"
$CMD --batch_size 16 --output_json "${OUT_DIR}/03_bf16_b16.json"

echo ""
echo ">>> [4/12] BF16, B=16, bucketing"
$CMD --batch_size 16 --use_bucketing --output_json "${OUT_DIR}/04_bf16_b16_bucket.json"

echo ""
echo ">>> [5/12] BF16, B=32, bucketing"
$CMD --batch_size 32 --use_bucketing --output_json "${OUT_DIR}/05_bf16_b32_bucket.json"

echo ""
echo ">>> [6/12] BF16, B=64, bucketing"
$CMD --batch_size 64 --use_bucketing --output_json "${OUT_DIR}/06_bf16_b64_bucket.json"

echo ""
echo ">>> [7/12] Liger-Kernel, B=32, bucketing"
$CMD --batch_size 32 --use_bucketing --use_liger --output_json "${OUT_DIR}/07_liger_b32_bucket.json"

echo ""
echo ">>> [8/12] torch.compile, B=32, bucketing"
$CMD --batch_size 32 --use_bucketing --use_compile --output_json "${OUT_DIR}/08_compile_b32_bucket.json"

echo ""
echo ">>> [9/12] TileGym, B=32, bucketing, pad64"
$CMD --batch_size 32 --use_bucketing --use_tilegym --pad_to_multiple_of 64 --output_json "${OUT_DIR}/09_tilegym_b32_bucket_pad64.json"

echo ""
echo ">>> [10/12] TileGym, B=64, bucketing, pad64"
$CMD --batch_size 64 --use_bucketing --use_tilegym --pad_to_multiple_of 64 --output_json "${OUT_DIR}/10_tilegym_b64_bucket_pad64.json"

echo ""
echo ">>> [11/12] Packing (N=1024), B=64"
$CMD --batch_size 64 --use_packing --pack_max_length 1024 --output_json "${OUT_DIR}/11_packing_b64.json"

echo ""
echo ">>> [12/12] Packing (N=1024) + TileGym, B=64"
$CMD --batch_size 64 --use_packing --pack_max_length 1024 --use_tilegym --output_json "${OUT_DIR}/12_packing_tilegym_b64.json"

echo ""
echo "========================================================================"
echo "  ALL DONE. Results in: ${OUT_DIR}/"
echo "========================================================================"
echo ""

# Print summary table
echo "Config                          | Real tok/s | VRAM   | Pad%  | ms/step"
echo "---                             | ---        | ---    | ---   | ---"
for f in "${OUT_DIR}"/*.json; do
    name=$(basename "$f" .json | sed 's/^[0-9]*_//')
    tok=$($PYTHON -c "import json; d=json.load(open('$f')); print(f\"{d['results']['real_tok_per_sec']:,.0f}\")")
    vram=$($PYTHON -c "import json; d=json.load(open('$f')); print(f\"{d['results']['vram_gb']:.1f}GB\")")
    pad=$($PYTHON -c "import json; d=json.load(open('$f')); print(f\"{d['results']['pad_pct']:.1f}%\")")
    ms=$($PYTHON -c "import json; d=json.load(open('$f')); print(f\"{d['results']['avg_step_ms']:.1f}\")")
    printf "%-32s| %-11s| %-7s| %-6s| %s\n" "$name" "$tok" "$vram" "$pad" "$ms"
done
