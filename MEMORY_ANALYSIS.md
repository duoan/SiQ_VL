# GPU Memory Usage Analysis for SiQ-VL

## Configuration
- **Vision Model**: `google/siglip2-so400m-patch16-512`
- **LLM Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4
- **Precision**: bf16
- **Gradient Checkpointing**: Enabled

## Memory Breakdown (Without Pixel Shuffle)

### Total Memory: **~10.8 GB**

| Component | Memory (GB) | Notes |
|-----------|------------|-------|
| Model Weights | 3.54 | Vision (0.75) + LLM (2.79) + Projector (0.00) |
| **LLM Activations** | **5.26** | **This is the main memory bottleneck** |
|   - Attention Matrices | 4.00 | O(n²) - scales with sequence length² |
|   - Hidden States | 0.50 | Stored at checkpoint boundaries |
|   - FFN Activations | 0.76 | Feed-forward network |
| Other Activations | 0.05 | Vision features, embeddings |
| Training Overhead | 0.01 | Gradients, optimizer states |
| Input Data | 0.01 | Pixel values, input IDs |
| System Overhead | 2.00 | PyTorch, CUDA, etc. |

## The Problem: Sequence Length

For `siglip2-so400m-patch16-512`:
- **Image Size**: 512×512
- **Patch Size**: 16×16
- **Number of Patches**: (512/16)² = **1024 patches**

When these 1024 image patches are concatenated with text tokens (e.g., 2048 tokens), the total sequence length becomes:
- **Total Sequence Length**: 2048 + 1024 = **3072 tokens**

The attention mechanism in transformers has **O(n²) memory complexity**, so:
- Attention memory per layer ≈ `batch_size × num_heads × seq_len² × bytes`
- With 1024 patches: ~4.0 GB for attention matrices
- This is the **primary cause of OOM**

## Solution: Pixel Shuffle

Pixel shuffle reduces the number of image tokens by spatially downsampling the patch sequence.

### How It Works

1. **Before Pixel Shuffle**: 32×32 = 1024 patches
2. **After Pixel Shuffle (factor=8)**: (32/8)×(32/8) = 4×4 = **16 patches**
3. **New Total Sequence Length**: 2048 + 16 = **2064 tokens**

The attention memory scales quadratically, so:
- **Memory Reduction**: (3072² - 2064²) / 3072² ≈ **55% reduction in attention memory**

### Valid Pixel Shuffle Factors

For 32×32 patches, valid factors are: **1, 2, 4, 8, 16, 32**

| Factor | Patches | Total Seq Len | Attention Memory (GB) | Total Memory (GB) | Savings |
|--------|---------|---------------|----------------------|-------------------|---------|
| 1 (none) | 1024 | 3072 | 4.00 | **10.82** | Baseline |
| 2 | 256 | 2304 | 2.25 | **8.68** | **19.7%** ✅ |
| 4 | 64 | 2112 | 1.89 | **8.33** | **23.0%** ✅✅ |
| **8** | **16** | **2064** | **1.81** | **8.65** | **20.1%** ✅✅ |
| 16 | 4 | 2052 | 1.80 | 10.31 | 4.7% |

## Recommendation

**Use `--pixel_shuffle_factor 4` or `8`**

### Factor 4 (Recommended for Best Balance)
- **Memory**: 8.33 GB (23% reduction)
- **Patches**: 64 (good quality retention)
- **Total Savings**: ~2.5 GB

### Factor 8 (Alternative)
- **Memory**: 8.65 GB (20% reduction)  
- **Patches**: 16 (more aggressive compression)
- **Total Savings**: ~2.2 GB

## How to Use

### Option 1: Manual Specification
```bash
STAGE=1 bash scripts/train_launch.sh --pixel_shuffle_factor 4
```

### Option 2: Auto-Calculation
⚠️ **WARNING**: The model will auto-calculate the **largest valid factor** if you don't specify one. 

For `siglip2-so400m-patch16-512` (32×32 patches), this would choose **factor=32**, reducing to just **1 patch** - this is **too aggressive** and will significantly hurt model quality!

**Recommendation**: **Always explicitly set** `--pixel_shuffle_factor 4` or `8` for optimal memory/quality tradeoff.

## Additional Memory Optimization Tips

1. **Reduce Batch Size**: If still OOM, try `--per_device_train_batch_size 2`
2. **Increase Gradient Accumulation**: Compensate with `--gradient_accumulation_steps 8`
3. **Enable Gradient Checkpointing**: Already enabled by default ✅
4. **Use bf16**: Already using bf16 ✅
5. **Freeze LLM in Stage 1**: Already doing this ✅

## Expected Results

With `pixel_shuffle_factor=4`:
- **Memory Usage**: ~8.3 GB (down from 10.8 GB)
- **Training Speed**: Slightly faster (shorter sequences)
- **Model Quality**: Minimal impact (64 patches still provides good spatial resolution)

## Verification

Run the memory calculation script to verify:
```bash
python3 scripts/calculate_memory.py
```

This will show detailed memory breakdown for your specific configuration.

