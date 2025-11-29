# Quick Fix for OOM Issues

## Problem
You're experiencing OOM (Out of Memory) errors when training with:
- Vision: `google/siglip2-so400m-patch16-512`
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`

## Root Cause
The vision model produces **1024 image patches** (32×32), which when combined with text tokens creates a very long sequence (~3072 tokens). The transformer attention mechanism has **O(n²) memory complexity**, making this the primary memory bottleneck.

## Solution: Use Pixel Shuffle

Add `--pixel_shuffle_factor 4` to your training command:

```bash
STAGE=1 bash scripts/train_launch.sh --pixel_shuffle_factor 4
```

Or for Stage 2:
```bash
STAGE=2 bash scripts/train_launch.sh --pixel_shuffle_factor 4
```

## Expected Results

- **Memory Reduction**: ~23% (from 10.8 GB to 8.3 GB)
- **Quality Impact**: Minimal (64 patches still provides good spatial resolution)
- **Training Speed**: Slightly faster (shorter sequences)

## Alternative Factors

- **Factor 4** (Recommended): 64 patches, ~8.3 GB memory
- **Factor 8**: 16 patches, ~8.7 GB memory  
- **Factor 2**: 256 patches, ~8.7 GB memory

## Why Not Use Auto-Calculation?

The auto-calculation chooses the **largest factor** (32), which reduces to just 1 patch - this is too aggressive and will hurt quality. Always specify the factor explicitly.

## Additional Tips if Still OOM

1. Reduce batch size: `--per_device_train_batch_size 2`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Both: Keep effective batch size = 2 × 8 = 16

See `MEMORY_ANALYSIS.md` for detailed analysis.

