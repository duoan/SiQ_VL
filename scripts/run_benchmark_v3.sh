#!/usr/bin/env bash
# V3 Benchmark Suite: Clean re-run of ALL experiments.
# - No gradient checkpointing (unnecessary on 90GB GPU)
# - Liger FusedCE tested both ON and OFF
# - No callbacks (benchmark script is raw training loop)
# - Single backbone: SigLIP2-base-224 + Qwen2.5-0.5B
#
# Usage:
#   bash scripts/run_benchmark_v3.sh

set -uo pipefail

MODEL="small"
STEPS=20
WARMUP=5
OUT_DIR="docs/traces/benchmark_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

PYTHON=".venv/bin/python"
BASE="$PYTHON scripts/benchmark_real_efficiency.py --model ${MODEL} --steps ${STEPS} --warmup ${WARMUP}"

echo "========================================================================"
echo "  V3 Full Benchmark Suite"
echo "  Model: small (SigLIP2-base-224 + Qwen2.5-0.5B)"
echo "  Output: ${OUT_DIR}"
echo "  No grad_ckpt, no callbacks, clean environment"
echo "========================================================================"

# ============================================================
# STAGE 1 (Frozen LLM — only projector trains)
# ============================================================
echo ""
echo "====================== STAGE 1 ======================"

echo -e "\n>>> [S1-01] Baseline: FP32, B=4"
$BASE --stage 1 --force_fp32 --batch_size 4 \
  --output_json "${OUT_DIR}/s1_01_baseline_fp32_b4.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-02] BF16, B=4"
$BASE --stage 1 --batch_size 4 \
  --output_json "${OUT_DIR}/s1_02_bf16_b4.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-03] BF16, B=16 (no bucket)"
$BASE --stage 1 --batch_size 16 \
  --output_json "${OUT_DIR}/s1_03_bf16_b16.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-04] BF16, B=16, bucketing"
$BASE --stage 1 --batch_size 16 --use_bucketing \
  --output_json "${OUT_DIR}/s1_04_bf16_b16_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-05] BF16, B=32, bucketing"
$BASE --stage 1 --batch_size 32 --use_bucketing \
  --output_json "${OUT_DIR}/s1_05_bf16_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-06] BF16, B=64, bucketing"
$BASE --stage 1 --batch_size 64 --use_bucketing \
  --output_json "${OUT_DIR}/s1_06_bf16_b64_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-07] Liger (with FusedCE), B=32, bucketing"
$BASE --stage 1 --batch_size 32 --use_bucketing --use_liger \
  --output_json "${OUT_DIR}/s1_07_liger_CE_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-08] Liger (no FusedCE), B=32, bucketing"
$BASE --stage 1 --batch_size 32 --use_bucketing --use_liger --no_fused_ce \
  --output_json "${OUT_DIR}/s1_08_liger_noCE_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-09] torch.compile, B=32, bucketing"
$BASE --stage 1 --batch_size 32 --use_bucketing --use_compile \
  --output_json "${OUT_DIR}/s1_09_compile_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-10] TileGym, B=32, bucketing, pad64"
$BASE --stage 1 --batch_size 32 --use_bucketing --use_tilegym --pad_to_multiple_of 64 \
  --output_json "${OUT_DIR}/s1_10_tilegym_b32_bucket_pad64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-11] TileGym, B=64, bucketing, pad64"
$BASE --stage 1 --batch_size 64 --use_bucketing --use_tilegym --pad_to_multiple_of 64 \
  --output_json "${OUT_DIR}/s1_11_tilegym_b64_bucket_pad64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-12] Packing N=1024, B=64 (no kernel)"
$BASE --stage 1 --batch_size 64 --use_packing --pack_max_length 1024 \
  --output_json "${OUT_DIR}/s1_12_packing_b64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S1-13] Packing N=1024 + TileGym, B=64"
$BASE --stage 1 --batch_size 64 --use_packing --pack_max_length 1024 --use_tilegym \
  --output_json "${OUT_DIR}/s1_13_packing_tilegym_b64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

# ============================================================
# STAGE 2 (Unfrozen LLM, NO gradient checkpointing)
# ============================================================
echo ""
echo "====================== STAGE 2 (no grad_ckpt) ======================"

echo -e "\n>>> [S2-01] Vanilla, B=4 (baseline)"
$BASE --stage 2 --batch_size 4 \
  --output_json "${OUT_DIR}/s2_01_vanilla_b4.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-02] Vanilla, B=16, bucketing"
$BASE --stage 2 --batch_size 16 --use_bucketing \
  --output_json "${OUT_DIR}/s2_02_vanilla_b16_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-03] Vanilla, B=32, bucketing"
$BASE --stage 2 --batch_size 32 --use_bucketing \
  --output_json "${OUT_DIR}/s2_03_vanilla_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-04] Liger (no FusedCE), B=16, bucketing"
$BASE --stage 2 --batch_size 16 --use_bucketing --use_liger --no_fused_ce \
  --output_json "${OUT_DIR}/s2_04_liger_noCE_b16_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-05] Liger (with FusedCE), B=16, bucketing"
$BASE --stage 2 --batch_size 16 --use_bucketing --use_liger \
  --output_json "${OUT_DIR}/s2_05_liger_CE_b16_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-06] Liger (no FusedCE), B=32, bucketing"
$BASE --stage 2 --batch_size 32 --use_bucketing --use_liger --no_fused_ce \
  --output_json "${OUT_DIR}/s2_06_liger_noCE_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-07] torch.compile, B=16, bucketing"
$BASE --stage 2 --batch_size 16 --use_bucketing --use_compile \
  --output_json "${OUT_DIR}/s2_07_compile_b16_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-08] torch.compile, B=32, bucketing"
$BASE --stage 2 --batch_size 32 --use_bucketing --use_compile \
  --output_json "${OUT_DIR}/s2_08_compile_b32_bucket.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-09] TileGym, B=16, bucket, pad64"
$BASE --stage 2 --batch_size 16 --use_bucketing --use_tilegym --pad_to_multiple_of 64 \
  --output_json "${OUT_DIR}/s2_09_tilegym_b16_bucket_pad64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-10] TileGym, B=32, bucket, pad64"
$BASE --stage 2 --batch_size 32 --use_bucketing --use_tilegym --pad_to_multiple_of 64 \
  --output_json "${OUT_DIR}/s2_10_tilegym_b32_bucket_pad64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-11] TileGym, B=64, bucket, pad64"
$BASE --stage 2 --batch_size 64 --use_bucketing --use_tilegym --pad_to_multiple_of 64 \
  --output_json "${OUT_DIR}/s2_11_tilegym_b64_bucket_pad64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-12] Packing N=1024, B=32 (no kernel)"
$BASE --stage 2 --batch_size 32 --use_packing --pack_max_length 1024 \
  --output_json "${OUT_DIR}/s2_12_packing_b32.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

echo -e "\n>>> [S2-13] Packing N=1024 + TileGym, B=64"
$BASE --stage 2 --batch_size 64 --use_packing --pack_max_length 1024 --use_tilegym \
  --output_json "${OUT_DIR}/s2_13_packing_tilegym_b64.json" 2>&1 | grep -E "(tok/s|waste|ratio|saved)"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================================================"
echo "  ALL DONE. Results: ${OUT_DIR}/"
echo "========================================================================"
echo ""

# Generate summary
$PYTHON -c "
import json
from pathlib import Path

out_dir = Path('${OUT_DIR}')
files = sorted(out_dir.glob('*.json'))

print()
print('  V3 RESULTS SUMMARY')
print('  ' + '='*90)

for stage in ['s1', 's2']:
    stage_files = [f for f in files if f.stem.startswith(stage)]
    if not stage_files:
        continue
    label = 'Stage 1 (Frozen LLM)' if stage == 's1' else 'Stage 2 (Unfrozen, no grad_ckpt)'
    print(f'\n  {label}')
    print(f'  {\"#\":<6} {\"Config\":<45s} {\"Real tok/s\":>10} {\"VRAM\":>6} {\"Pad%\":>5}')
    print(f'  {\"-\"*75}')
    
    baseline = None
    for f in stage_files:
        data = json.loads(f.read_text())
        r = data['results']
        cfg = data['config']
        
        parts = []
        if cfg.get('force_fp32'): parts.append('FP32')
        if cfg.get('use_compile'): parts.append('compile')
        elif cfg.get('use_tilegym'): parts.append('TileGym')
        elif cfg.get('use_liger'):
            if cfg.get('no_fused_ce'): parts.append('Liger(noCE)')
            else: parts.append('Liger(+CE)')
        else: parts.append('Vanilla')
        parts.append(f\"B={cfg['batch_size']}\")
        if cfg.get('use_packing'): parts.append(f\"pack{cfg.get('pack_max_length',1024)}\")
        elif cfg.get('use_bucketing'): parts.append('bucket')
        if cfg.get('pad_to_multiple_of'): parts.append(f\"pad{cfg['pad_to_multiple_of']}\")
        name = ', '.join(parts)
        
        toks = r['real_tok_per_sec']
        if baseline is None: baseline = toks
        speedup = toks / baseline
        vram = r.get('vram_gb', 0)
        pad = r.get('pad_pct', 0)
        
        print(f'  {f.stem[:5]:<6} {name:<45s} {toks:>10,.0f} {vram:>5.1f}G {pad:>4.1f}%  {speedup:.2f}x')
"
