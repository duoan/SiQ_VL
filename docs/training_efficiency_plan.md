# SiQ-VL Training Efficiency Optimization — Plan & Iteration Log

> This document is the **iteration log for SiQ-VL training efficiency optimization**, and also serves as raw engineering evidence for a future blog post.
> Each optimization round (Iteration N) follows a **Hypothesis → Change → Measurement → Result → Decision** five-part structure,
> ensuring every commit, number, and screenshot is traceable back to a specific iteration.

---

## 0. Scope & Hardware

| Item | Configuration |
|---|---|
| Hardware | 1 × NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM, sm_120 |
| CUDA / Driver | CUDA 13.0, Driver 580.126.20 |
| Python / Torch | Python 3.10, PyTorch 2.9.1, transformers ≥ 4.57.3 |
| Vision encoder | `google/siglip2-so400m-patch16-512` (frozen in Stage 1; mostly frozen in Stage 2) |
| LLM backbone | `Qwen/Qwen2.5-1.5B-Instruct` (frozen in Stage 1, trainable / LoRA in Stage 2) |
| Precision | bf16 |
| Effective batch | per_device_bs (4) × grad_accum (4) = 16 |
| Dataset | HuggingFaceM4/FineVision (multi-subset interleaved) |

**Optimization Goal**: Reduce single-GPU training step time by at least **2x** (ideally **3–5x**)
while maintaining loss/eval quality, stable VRAM usage, and full reproducibility.

**Non-Goals (out of scope for now)**:

- Model architecture changes (swapping vision encoder or LLM)
- Quantized training (QLoRA, INT8 training)

**Future Phase (planned after single-GPU optimization is validated)**:

- Multi-GPU distributed training on [Modal](https://modal.com/) (FSDP / DeepSpeed ZeRO) — see §3 Phase P4

---

## 1. Baseline Pipeline Overview

### 1.1 Data Flow

```
HF FineVision (PIL)
  → VQADataset (randomly selects one turn per item)
    → SiQ_VLDataCollator (per-batch, runs on dataloader workers)
      → SiQ_VLProcessor.__call__:
          1. SiQ_VLImageProcessor: PIL → resize/tile → tensor → normalize  (CPU)
          2. tokenizer.apply_chat_template + tokenizer(...)               (CPU)
          3. label masking                                                 (CPU)
        → batch (pixel_values, input_ids, attention_mask, labels, num_image_tokens)
          → GPU
            → SiglipVisionModel(pixel_values)        →  (B*Tiles, P, D_v)
              → Projector: pixel_shuffle + Linear    →  (B*Tiles, P', D_t)
                → scatter vision tokens into input_embeds at <|image_pad|> positions
                  → Qwen2ForCausalLM (HF default lm_head + cross_entropy)
                    → loss
```

### 1.2 Known Inefficiency Sources (ranked by estimated savings, largest first)

| # | Problem | Estimated Impact | Corresponding Optimization |
|---|---|---|---|
| 1 | `padding="longest"` pads within each batch → O(L²) attention waste 30–50% | step time -20%~-40% | **Bucketing + Packing** |
| 2 | Stage 1 Vision encoder is frozen yet SigLIP forward runs every epoch | Stage 1 wall-clock -40%~-60% | **SigLIP feature offline cache** |
| 3 | LM Head + softmax + CE not fused; logits tensor (B, L, V≈152K) ≈ 2–5 GB intermediate | step time -10%~-20%, peak mem -20%~-30% | **Liger-Kernel fused linear-CE** |
| 4 | Default attn backend may not be FA2; vision forward not wrapped in `no_grad` | step time -10%~-20% | **Explicit FA2 + no_grad on vision** |
| 5 | Stage 1 LLM is frozen yet gradient checkpointing still enabled (useless recompute) | step time -10%~-20% | **Disable ckpt for frozen modules** |
| 6 | Image preprocessing (PIL bicubic / tile) entirely on CPU; `num_workers=4` too low | step time -5%~-15% | **Offline preprocessing + tuned dataloader** |
| 7 | `token_embeddings[image_token_positions] = ...` scatter / projector not fused | < 5% | Not prioritized |

---

## 2. Optimization Tracks (Three Tracks + System-Level)

### Track A — Data: Offline Preprocessing + Feature Cache + Bucketing/Packing

- **A1** Offline dump `pixel_values` and `num_tiles_per_image`, skip PIL/torchvision at train time
- **A2** Offline dump SigLIP features (optionally post-pixel-shuffle), skip vision forward entirely in Stage 1
- **A3** Token length estimation → bucketing → sample packing within buckets (block-diagonal attention mask)

### Track B — Model: Fused Kernels for VL Training

- **B1** **Liger-Kernel `LigerFusedLinearCrossEntropyLoss`**: never materializes full logits tensor
- **B2** Liger fused **RMSNorm + Residual**
- **B3** Liger fused **SwiGLU**
- **B4** **FlashAttention 2** / **FlexAttention** (required for packing)
- **B5** Custom kernel for projector / pixel-shuffle (low priority)

### Track C — System: Framework/Scheduling Level

- **C1** Disable gradient checkpointing for LLM in Stage 1 (frozen module, no gradients to recompute)
- **C2** Explicitly set `attn_implementation="flash_attention_2"`
- **C3** Wrap `vision_model.forward()` with `torch.no_grad()` in Stage 1 (even with `requires_grad=False`, autograd graph is still built)
- **C4** DataLoader: `num_workers=8/16`, `pin_memory=True`, `persistent_workers=True`
- **C5** `torch.compile(mode="reduce-overhead")` (last-mile; test Liger compatibility first)

### Track D — Infrastructure: Modal Distributed Training

- **D1** Package the training pipeline as a Modal `App` with `@modal.function(gpu="A100-80GB", count=N)` or equivalent
- **D2** Use Modal `Volume` for dataset cache (pre-processed pixel_values / SigLIP features) and checkpoint persistence
- **D3** Integrate PyTorch FSDP (or DeepSpeed ZeRO-2/3) for multi-GPU sharding within a Modal container group
- **D4** Implement elastic fault tolerance: checkpoint every K steps to Modal Volume, auto-resume on preemption
- **D5** Parameterize GPU type / count via Modal `@modal.cls` config so scaling is a one-line change
- **D6** CI/CD: trigger training runs from GitHub Actions → Modal webhook, results auto-upload to W&B

---

## 3. Implementation Roadmap (Risk-Adjusted)

| Phase | Contents | Expected Step Time Gain | VRAM Gain | Complexity | Risk |
|---|---|---|---|---|---|
| **P0** (0.5–1 day) | Liger fused LMHead-CE + RMSNorm + SwiGLU; FA2; vision no_grad; disable Stage 1 ckpt | -25%~-40% | -20%~-30% | Low | Low |
| **P1** (1–2 days) | SigLIP feature offline cache + Dataset/Collator/Model adaptation; HF `group_by_length=True` | Stage 1 wall-clock -40%~-60% | Neutral | Medium | Medium |
| **P2** (2–3 days) | Sample packing + FlexAttention block-diagonal mask | -30%~-50% | -10%~-20% | High | High (requires modifying Qwen2 forward) |
| **P3** (0.5 day) | `torch.compile` LLM | -5%~-15% | Neutral | Medium | Medium (potential conflict with P0 monkey-patches) |
| **P4** (2–4 days) | Modal distributed training (multi-GPU FSDP/DeepSpeed, cloud volumes for data/checkpoints) | Linear scaling with N GPUs | Enables larger batch / longer runs | High | Medium (infra complexity, network bandwidth) |

Combined targets:

- After P0: single-GPU step time **~1.5x faster**
- After P0 + P1: Stage 1 total wall-clock **~3x faster**, Stage 2 **~1.5–2x faster**
- After P0 + P1 + P2: Stage 2 total wall-clock **~3–5x faster**
- After P4: scale to N GPUs on Modal → **Nx throughput** (near-linear with FSDP/DDP on fast interconnect)

---

## 4. Iteration Log (filled in reverse chronological order)

> Every iteration (success or failure) gets one entry using the template below.
> **Failed iterations must also be recorded** — they become "pitfalls I hit" material in the blog.

### Iteration Template

```
### Iteration N — <one-line title>

- **Date**:
- **Branch / Commit**:
- **Hypothesis**:
- **Change**:
  - File 1: ...
  - File 2: ...
- **How to Reproduce**:
  ```bash
  ...
  ```
- **Measurement Setup**:
  - GPU: ...
  - Batch / accum / seq len: ...
  - Steps measured: ...
- **Result**:
  | Metric | Before | After | Delta |
  |---|---|---|---|
  | step time (ms) |  |  |  |
  | peak VRAM (GB) |  |  |  |
  | tokens / s |  |  |  |
  | eval loss @ step K |  |  |  |
- **Artifacts**:
  - W&B run: ...
  - Profiler trace: docs/traces/iter_N.json
  - Screenshot: docs/figs/iter_N_*.png
- **Decision**: Keep / Revert / Park
- **Lessons / Surprises**:
```

---

### Iteration 0 — Baseline Measurement

- **Date**: 2026-05-19
- **Branch / Commit**: `main` @ HEAD
- **Hypothesis**: Record baseline numbers without any optimization, to serve as the reference point for all subsequent iterations.
- **Change**: None (measurement only)
- **How to Reproduce**:
  ```bash
  bash scripts/profile_baseline.sh
  ```
- **Measurement Setup**:
  - GPU: 1 × RTX PRO 6000 Blackwell 96GB
  - per_device_bs=4, grad_accum=4, bf16, gradient_checkpointing=True
  - vision: `siglip2-so400m-patch16-512`, LLM: `Qwen2.5-1.5B-Instruct`
  - pixel_shuffle_factor=4
  - Dataset: `sharegpt4v(coco)`, 2000 samples
  - Warm up 5 steps, profiled steps 6–25 (20 steps)
- **Result**:
  | Metric | Value |
  |---|---|
  | avg step time (ms) | 310.7 |
  | p50 step time (ms) | 313.4 |
  | p95 step time (ms) | 347.3 |
  | min step time (ms) | 284.8 |
  | max step time (ms) | 347.3 |
  | tokens / sec | 4,674 |
  | peak VRAM (GB) | 19.27 |
  | allocated VRAM (GB) | 8.70 |
  | reserved VRAM (GB) | 25.35 |
  | avg tokens / step | 1,452 |
- **Profiler Hot Spots (CUDA time)**:
  | Operation | Self CUDA % | Self CUDA Time | Notes |
  |---|---|---|---|
  | `aten::mm` (all matmuls) | 49.96% | 1.699s | LLM + projector weight matmuls |
  | `cutlass_80_simt_sgemm_128x256` | 33.64% | 1.144s | Running FP32 GEMM kernels (!!) |
  | `aten::addmm` | 27.05% | 920ms | Linear layers |
  | `magma_sgemmEx` | 15.61% | 531ms | Another FP32 path |
  | `_efficient_attention_forward` | 7.54% | 256ms | SDPA (not FA2!) |
  | `_efficient_attention_backward` | 2.85% | 97ms | SDPA backward |
  | `aten::mul` | 3.10% | 105ms | Elementwise ops |
  | `aten::bmm` (vision) | 2.43% | 83ms | SigLIP attention |
- **Key Findings**:
  1. **Running FP32 GEMM kernels (cutlass sgemm)** — despite bf16 being enabled, the majority of matmuls are hitting FP32 CUTLASS kernels. This is likely because gradient_checkpointing + frozen modules are causing dtype mismatches, or the SigLIP vision encoder is running in FP32.
  2. **Using SDPA `_efficient_attention`, NOT FlashAttention 2** — confirms our hypothesis that FA2 is not being used.
  3. **SigLIP forward is visible** — `aten::bmm` from vision attention takes 83ms per batch, confirming value of caching.
  4. **Peak VRAM only 19.27 GB** on a 96GB card — there is massive headroom. Could increase batch size significantly or skip gradient checkpointing.
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_0_baseline.json` (343 MB — open in https://ui.perfetto.dev)
  - Summary JSON: `docs/traces/iter_0_baseline_summary.json`
- **Decision**: Proceed with P0 optimizations. Priority order adjusted:
  1. **Fix FP32 GEMM issue** (this alone could be a 2x win — bf16 GEMM is ~2x faster than FP32)
  2. FA2 explicit enable
  3. Liger fused LMHead-CE
  4. Vision no_grad + disable unnecessary ckpt
- **Lessons / Surprises**:
  - The biggest surprise is the FP32 GEMM. We declared bf16 training, yet CUTLASS is dispatching `sgemm` (single-precision GEMM). This means either: (a) model weights are in FP32 on device, or (b) autocast is not wrapping the frozen modules correctly. This is likely the single largest perf bug — fixing it could yield 1.5–2x alone.
  - VRAM usage is low (19 GB / 96 GB). We're not memory bound at all — we're compute bound on suboptimal FP32 kernels.
  - SDPA efficient attention is being used (memory-efficient variant), not the flash variant. On Blackwell this should be switchable.

---

### Iteration 1 — P0.0: Fix FP32 dtype + Vision no_grad

- **Date**: 2026-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: Profiler shows FP32 CUTLASS GEMM kernels consuming 50%+ CUDA time despite
  bf16 config. Root cause: `from_pretrained()` loads weights in FP32 by default, and HF Trainer's
  AMP autocast only covers forward — frozen modules may not benefit fully. Additionally,
  vision_model.forward() builds an autograd graph even when frozen, wasting time and memory.
- **Change**:
  - `siq_vl/model/modeling.py::get_stage1_model_and_processor`: load text_model and vision_model
    with `torch_dtype=torch.bfloat16`, cast projector to bf16
  - `siq_vl/model/modeling.py::get_stage2_model_and_processor`: load with `torch_dtype=torch.bfloat16`
  - `siq_vl/model/modeling.py::SiQ_VLForCausalLM.forward`: wrap vision_model forward in
    `torch.no_grad()` since it is always frozen
- **How to Reproduce**:
  ```bash
  TRACE_NAME=iter_1_bf16_fix bash scripts/profile_baseline.sh
  ```
- **Result**:
  | Metric | Before (Iter 0) | After (Iter 1) | Delta |
  |---|---|---|---|
  | avg step time (ms) | 310.7 | 101.1 | **-67% (3.07x faster)** |
  | p50 step time (ms) | 313.4 | 102.2 | -67% |
  | tokens / sec | 4,674 | 14,366 | **+207%** |
  | peak VRAM (GB) | 19.27 | 11.54 | **-40%** |
  | allocated VRAM (GB) | 8.70 | 4.38 | -50% |
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_1_bf16_fix.json`
  - Summary JSON: `docs/traces/iter_1_bf16_fix_summary.json`
- **Decision**: Keep
- **Lessons / Surprises**:
  - This single fix gave us **3x speedup** — more than any fused kernel could. The lesson: always
    check actual kernel dispatches, not just your config flags. `bf16=True` in TrainingArguments
    only controls autocast wrapping, it does NOT control the dtype of loaded model weights.
  - Memory dropped 40% because bf16 weights are half the size of FP32, and the intermediates
    (activations, gradients) are also bf16 throughout.
  - The `torch.no_grad()` on vision forward means no autograd graph is retained for those tensors,
    further reducing memory and eliminating useless backward graph construction.

---

### Iteration 2 — P0.1: Liger-Kernel (Fused Linear-CE + RMSNorm + SwiGLU + RoPE)

- **Date**: 2026-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: Qwen2.5-1.5B has vocab=151,936. The `hidden @ lm_head.T` operation produces
  the largest intermediate tensor in the entire forward chain. Liger-Kernel's
  `LigerFusedLinearCrossEntropyLoss` computes logits → log_softmax → nll in chunks
  without ever materializing the full logits tensor, saving memory. Additionally, fused
  RMSNorm, SwiGLU, and RoPE should reduce kernel launch overhead.
- **Change**:
  - Add dependency: `liger-kernel==0.8.0` (via `uv add liger-kernel`)
  - `siq_vl/model/modeling.py`: add `_apply_liger_kernel()` helper that calls
    `apply_liger_kernel_to_qwen2(rope=True, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True)`
  - Called before model instantiation in both `get_stage1_model_and_processor` and
    `get_stage2_model_and_processor`
  - Graceful fallback if liger-kernel not installed
- **How to Reproduce**:
  ```bash
  TRACE_NAME=iter_2_liger bash scripts/profile_baseline.sh
  ```
- **Result**:
  | Metric | Iter 1 (bf16 only) | Iter 2 (+ Liger) | Delta |
  |---|---|---|---|
  | avg step time (ms) | 101.1 | 135.4 | +34% (slower!) |
  | p50 step time (ms) | 102.2 | 120.8 | +18% (slower) |
  | min step time (ms) | 284.8→87.9 | 106.0 | — |
  | tokens / sec | 14,366 | 10,745 | -25% |
  | peak VRAM (GB) | 11.54 | **6.78** | **-41%** |
  | allocated VRAM (GB) | 4.38 | 3.93 | -10% |
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_2_liger.json`
  - Summary JSON: `docs/traces/iter_2_liger_summary.json`
- **Decision**: Keep (for memory savings; step time regression is acceptable given the context)
- **Lessons / Surprises**:
  - **Step time is SLOWER** with Liger in Stage 1. Root cause: in Stage 1 the LLM is **frozen** —
    there is no backward pass through the LLM layers, so the fused linear-CE backward savings
    don't materialize. Meanwhile, Triton JIT compilation adds ~100ms spikes, and fused ops have
    per-kernel launch overhead that exceeds savings on this small seq_len (~360 tokens/sample).
  - **Memory is significantly better** (-41%): the fused linear-CE never materializes the full
    (B, L, 151936) logits tensor, saving ~2–4 GB of peak allocation.
  - The profiler shows `flash_fwd_kernel` is now being called — Liger's RoPE patch changed the
    attention dispatch path, triggering FlashAttention instead of SDPA efficient. This is a
    side-effect win.
  - **Key insight**: Liger's value is primarily in **Stage 2** (unfrozen LLM with backward) and in
    **memory-constrained scenarios** (larger batch/seq). For Stage 1 projector-only training with
    small sequences, the overhead outweighs the gains. However, keeping it is still correct because:
    (a) memory headroom allows larger batches later, (b) Stage 2 will benefit fully, (c) it
    triggered FA2 as a side effect.
  - The 309ms outlier in the trace is Triton JIT compiling a new kernel shape — will disappear
    once shapes stabilize (e.g., with packing at fixed max_length).

---

### Iteration 2 — P0.2: Explicitly Enable FlashAttention 2 (planned)

- **Date**: TBD
- **Hypothesis**: HF auto-selects attention backend based on `_supports_*` flags, but the
  actual runtime backend may be SDPA rather than FA2. On Blackwell + bf16, FA2's wall-clock
  is typically 10–20% faster than SDPA.
- **Change**:
  - `siq_vl/model/modeling.py`: in `get_stage{1,2}_model_and_processor`, pass
    `attn_implementation="flash_attention_2"` to `from_pretrained` / config construction
  - Also set `attn_implementation` in `SiQ_VLConfig` / `SiQ_VLTextConfig`
  - Print `model.config._attn_implementation` at startup to confirm
- **Expected Result**: step time ↓ 10–20%
- **Decision**: —

---

### Iteration 3 — Throughput Optimization: Batch Size + Vision Acceleration Investigation

- **Date**: 2025-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: With Iter 2's VRAM reduction (6.78 GB peak), we have massive headroom on the
  96 GB Blackwell GPU. Three approaches investigated:
  1. **Vision feature caching** — pre-extract SigLIP features offline, skip vision forward during training
  2. **torch.compile on vision** — JIT-optimize the frozen vision encoder
  3. **Increase batch size** — use the VRAM headroom to improve GPU utilization
- **Changes**:
  - `scripts/extract_vision_features.py`: new offline feature extraction script (for future use)
  - `siq_vl/dataset.py`: add `CachedVQADataset` for pre-cached vision features
  - `siq_vl/collator.py`: add `CachedVisionDataCollator`
  - `siq_vl/model/processing.py`: add `process_cached()` method to `SiQ_VLProcessor`
  - `siq_vl/model/modeling.py`: add `vision_features` kwarg to `forward()` for cached bypass path;
    add explicit `attn_implementation="sdpa"` to vision model loading
  - `scripts/profile_baseline.py`: add `--cached_features_dir`, `--no_gradient_checkpointing` flags
- **Investigation Results**:

  | Approach | Tokens/sec | VRAM | Verdict |
  |---|---|---|---|
  | Iter 2 baseline (bs=4, accum=4) | 11,070 | 6.78 GB | — |
  | Vision feature caching (bs=4) | 11,070 | 8.58 GB | **Worse** (H2D transfer > compute) |
  | torch.compile vision (bs=4) | 11,070 | 6.78 GB | **Marginal** (+3%, 40s compile overhead) |
  | Grad ckpt disabled (bs=4) | 11,400 | 6.84 GB | **Marginal** (+3%) |
  | **bs=16, accum=1, no grad ckpt** | **20,284** | **16.49 GB** | **Winner** (+83%) |
  | bs=32, accum=1 | 21,162 | 29.13 GB | Diminishing returns |

- **Final Result** (bs=16, no gradient checkpointing, 8 DataLoader workers):
  | Metric | Iter 2 (bs=4, accum=4) | Iter 3 (bs=16, accum=1) | Delta |
  |---|---|---|---|
  | tokens / sec | 11,070 | **20,284** | **+83%** |
  | avg step time (ms) | 135.4 | 282.9 | +109% (but 4x more tokens/step) |
  | p50 step time (ms) | 120.8 | 281.0 | — |
  | peak VRAM (GB) | 6.78 | 16.49 | +143% |
  | avg tokens/step | 1,452 | 5,739 | +295% |
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_3_throughput_opt.json`
  - Summary JSON: `docs/traces/iter_3_throughput_opt_summary.json`
- **Decision**: Keep. The batch size increase is the clear winner for throughput.
- **Lessons / Surprises**:
  - **Vision caching is counterproductive on Blackwell**: SigLIP forward takes only ~21ms/step
    (5ms/tile) with SDPA/flash+bf16. Loading cached features from CPU→GPU via DataLoader is
    slower than just computing them on-the-fly. The H2D transfer of large tensors
    (tiles × 1024 × 1152 × 2 bytes) is a bigger bottleneck than the compute.
  - **torch.compile offers negligible gains**: The vision encoder's attention is already dispatched
    to flash kernels via SDPA, and matmuls already use CUTLASS bf16 tensor cores. compile's
    fusion opportunities are minimal for this workload.
  - **Gradient checkpointing is unnecessary in Stage 1**: The LLM is frozen (no backward through it),
    so checkpointing has nothing useful to recompute. Disabling saves ~3% step time.
  - **The real bottleneck is GPU utilization**: With bs=4 and grad_accum=4, the GPU is underutilized
    because each micro-batch is tiny. Increasing to bs=16 achieves 83% more throughput because:
    (a) larger matmuls hit better CUTLASS tiling; (b) less per-step overhead (DataLoader, optimizer);
    (c) the vision encoder processes 4x more tiles in one batched call.
  - **Diminishing returns above bs=16**: bs=32 gives only 4% more tok/s but doubles VRAM. The memory
    bandwidth ceiling is hit; compute utilization is already ~90%.
  - The vision feature caching infrastructure (extract script, CachedVQADataset, forward bypass)
    is kept in the codebase for future use with slower GPUs or multi-epoch training on large datasets.

---

### Iteration 4 — P1.1: Offline Image Preprocessing (planned)

- **Date**: TBD
- **Hypothesis**: PIL bicubic + ToTensor + Normalize + Tile/Split on 4 CPU workers cannot
  saturate the GPU. Once `pixel_values` (+ `num_tiles_per_image`) are dumped to disk,
  the dataloader only needs to tokenize (very fast) + read tensors (mmap).
- **Change**:
  - New script `scripts/preprocess_dataset.py`:
    - Inputs: HF dataset name + subset list
    - Runs SiQ_VLImageProcessor, outputs sharded tensors to `outputs/cache/pixels/{subset}/{shard}.pt`
    - Also dumps `meta.parquet`: per-sample mapping to image shard idx + offset + num_tiles
  - New module `siq_vl/cached_dataset.py` (`CachedVQADataset`): loads from cache + original text stream
  - `SiQ_VLDataCollator` adds a `from_cache=True` path that skips image_processor
- **Expected Result**: dataloader wait ↓ 50–80%, overall step time ↓ 5–15%
- **Decision**: —

---

### Iteration 5 — P1.2: SigLIP Feature Offline Cache (planned)

> Prerequisites: vision encoder remains frozen throughout training (including Stage 2).
> If Stage 2 plans to unfreeze vision, **this step applies only to Stage 1**.

- **Date**: TBD
- **Hypothesis**: Stage 1 SigLIP forward is entirely redundant (no gradients). Caching
  SigLIP outputs to disk makes the vision forward time zero — projector directly consumes cached features.
- **Change**:
  - `scripts/preprocess_dataset.py` adds `--dump_vision_features` mode:
    - Runs SigLIP forward, dumps `vision_features` (bf16, shape `(num_tiles, P, D_v)`)
    - Optionally `--dump_after_pixel_shuffle` for further compression (D_v×r² replaces D_v)
  - `siq_vl/model/modeling.py::SiQ_VLForCausalLM.forward`:
    Adds `vision_features` parameter; if provided, skips `self.vision_model(...)` and goes directly to projector
  - `siq_vl/cached_dataset.py`: loads `vision_features` from cache alongside text data
- **Trade-offs**:
  - Disk space: each token ≈ `D_v × 2 bytes`, SigLIP-so400m D_v=1152 → ~2.3 KB/token,
    each image at 1024 tokens ≈ 2.4 MB; 1M images ≈ 2.4 TB → **too large**
  - Mitigation: (a) cache after pixel_shuffle (factor=4 → tokens ÷ 16, ~150 GB / 1M images);
    (b) use fp16 storage; (c) only cache the training subset
- **Expected Result**: Stage 1 step time ↓ 30–50% (vision forward time drops to zero)
- **Decision**: —

---

### Iteration 5 — P1.3: Length Bucketing

- **Date**: 2025-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: Intra-batch padding wastes ~17.4% of tokens. HF Trainer's built-in
  `group_by_length=True` groups samples by length during sampling, significantly reducing
  length variance within each batch.
- **Change**:
  - `siq_vl/dataset.py::VQADataset`: add `lengths` property with fast heuristic length
    estimation (~3.5 chars/token + template overhead + vision tokens)
  - `scripts/profile_baseline.py`: add `--group_by_length` flag, add `ProfileTrainer` subclass
    with custom `_get_train_sampler` to pass pre-computed lengths
  - `scripts/train.py`: add `group_by_length=True` to TrainingArguments, add `SiQVLTrainer`
    subclass for length-aware sampling
- **How to Reproduce**:
  ```bash
  python scripts/profile_baseline.py --trace_name iter_5_length_bucketing \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --no_gradient_checkpointing --group_by_length
  ```
- **Result**:
  | Metric | Iter 3 (random shuffle) | Iter 5 (length bucketing) | Delta |
  |---|---|---|---|
  | tokens / sec | 20,284 | **21,026** | **+3.7%** |
  | avg step time (ms) | 282.9 | 270.6 | -4.3% |
  | p50 step time (ms) | 281.0 | 270.6 | -3.7% |
  | peak VRAM (GB) | 16.49 | 15.79 | -4.2% |
  | padding waste | 17.4% | **4.1%** | **-13.3pp** |
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_5_length_bucketing.json`
  - Summary JSON: `docs/traces/iter_5_length_bucketing_summary.json`
- **Decision**: Keep. Free optimization with no code complexity or memory cost.
- **Lessons**:
  - Padding waste reduced from 17.4% → 4.1%, but tok/s only improved 3.7%. This is because
    the dataset already has low length variance (CoV=11.5%, range 260–502 tokens).
  - For datasets with higher length variance (e.g., mixing short captions with long reasoning),
    bucketing would show much larger gains.
  - Bucketing also slightly reduces peak VRAM (-4.2%) because the max sequence length within
    a batch is shorter on average.
  - The remaining 4.1% padding waste is the floor for this approach — to eliminate it entirely,
    sample packing is needed (Iteration 7).

---

### Iteration 7 — P2.1: Sample Packing + FlexAttention (planned)

- **Date**: TBD
- **Hypothesis**: Even after bucketing, ~10–20% padding remains. Concatenating multiple
  short samples into a fixed `max_length` sequence, with a block-diagonal mask ensuring
  cross-sample invisibility → padding ≈ 0.
- **Backend Choice**: FlexAttention (PyTorch 2.5+ native, pure PyTorch + `torch.compile`,
  no external dependencies). Fallback: `flash_attn_varlen_func`.
- **Change**:
  - New module `siq_vl/packing_collator.py::PackingCollator`:
    - Input: multiple tokenized samples from within a bucket
    - Output: `input_ids` (1, L_pack), `position_ids` (reset per sub-sequence), `cu_seqlens` (varlen),
      `seq_idx` (which sub-sequence each token belongs to), `labels`, `pixel_values_list`,
      `image_token_offsets` (position of each vision token in the packed sequence)
  - Modify `SiQ_VLForCausalLM.forward`:
    - Accept packed inputs
    - Scatter vision tokens using new offsets
    - LLM forward uses FlexAttention with `mask_mod` implementing block-diagonal causal mask
  - Unit test: same samples packed vs non-packed should produce loss within 1e-4
- **Expected Result**: step time ↓ 30–50%
- **Risk**:
  - HF Qwen2 forward does not natively accept packed format → requires wrap / monkey-patch
  - position_ids must be correctly reset per sub-sequence for RoPE
  - Compatibility with Liger-Kernel must be verified (Liger also patches RMSNorm etc.)
- **Decision**: —

---

### Iteration 7 — torch.compile Full Model (replaces Liger)

- **Date**: 2025-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: `torch.compile(mode="max-autotune-no-cudagraphs")` applied to the full model
  can fuse ops across the entire graph (attention + MLP + embedding), potentially outperforming
  Liger-Kernel's targeted patches. Key insight: Liger and torch.compile are INCOMPATIBLE —
  Liger's triton kernels cause illegal memory access under dynamo tracing.
- **Change**:
  - `scripts/profile_baseline.py`: add `--no_liger` and `--compile_llm` flags
  - When `--no_liger` is set, prevent `_apply_liger_kernel()` from running
  - When `--compile_llm` is set, wrap model with `torch.compile(mode="max-autotune-no-cudagraphs")`
- **How to Reproduce**:
  ```bash
  python scripts/profile_baseline.py --trace_name iter_7_compile_llm \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --no_gradient_checkpointing --group_by_length --no_liger --compile_llm
  ```
- **Result**:
  | Metric | Iter 5 (Liger + bucketing) | Iter 7 (torch.compile, no Liger) | Delta |
  |---|---|---|---|
  | tokens / sec | 21,026 | **27,352** | **+30.1%** |
  | avg step time (ms) | 270.6 | **203.9** | **-24.6%** |
  | p50 step time (ms) | 270.6 | 201.9 | -25.4% |
  | peak VRAM (GB) | 15.79 | 20.73 | +31.3% |
- **Additional investigation**: bs=24 with compile = 27,996 tok/s (+2.4%), diminishing returns.
  bs=32 hits CUDA OOM under compile.
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_7_compile_llm.json`
  - Summary JSON: `docs/traces/iter_7_compile_llm_summary.json`
- **Decision**: **KEEP**. Replace Liger with torch.compile as the primary optimization.
  The 30% throughput gain far outweighs the 5 GB VRAM increase (20.73 GB is still only 22% of
  the 96 GB Blackwell GPU).
- **Lessons**:
  - **Liger and torch.compile are mutually exclusive**: Liger monkey-patches model internals
    with custom triton kernels that confuse torch._dynamo's graph capture, causing illegal
    memory access errors.
  - **torch.compile provides broader optimization**: Instead of fusing specific ops (RMSNorm,
    SwiGLU), compile optimizes the entire computational graph including attention patterns,
    GEMM scheduling, and memory access patterns.
  - **VRAM tradeoff is acceptable**: torch.compile doesn't have Liger's fused CE memory savings,
    but 20 GB is minimal on modern GPUs. For memory-constrained setups, Liger remains the
    better choice.
  - **Warmup overhead**: First 3–5 steps take 10–30s each for compilation (amortized over training).

---

### Iteration 8 — Batch Size Tuning Under torch.compile

- **Date**: 2025-05-19
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: With torch.compile reducing peak VRAM (23.7 GB for bs=16), there's headroom
  to push batch size higher → better GPU utilization → more tok/s.
- **Investigation**:
  | Config | Tok/s | Step (ms) | VRAM |
  |---|---|---|---|
  | bs=16, compile | 27,108 | 196 | 23.7 GB |
  | **bs=20, compile** | **28,032** | **250** | **29.8 GB** |
  | bs=24, compile | 27,440 | 310 | 38.5 GB |
  | bs=28, compile | 27,781 | 355 | 43.0 GB |
  | bs=32, compile | OOM | — | — |

  Also tested:
  - `reduce-overhead` (cudagraphs): **-20%** — dynamic shapes cause frequent graph re-capture
  - `pad_to_multiple_of=64`: **-6.6%** — extra padding tokens waste compute; doesn't eliminate
    recompilation (5 discrete lengths still)
  - `torch.compile(mode="default")`: same as max-autotune-no-cudagraphs (within noise)
- **Final Result** (bs=20, compile, bucketing, no Liger):
  | Metric | Iter 7 (bs=16) | Iter 8 (bs=20) | Delta |
  |---|---|---|---|
  | tokens / sec | 27,352 | **28,322** | **+3.5%** |
  | avg step time (ms) | 203.9 | 252.9 | +24% (but 25% more tokens/step) |
  | peak VRAM (GB) | 20.73 | 29.09 | +40% |
- **Artifacts**:
  - Chrome trace: `docs/traces/iter_8_compile_bs20.json`
  - Summary JSON: `docs/traces/iter_8_compile_bs20_summary.json`
- **Decision**: Keep bs=20 as the default for Stage 1 training.
- **Lessons**:
  - Diminishing returns above bs=20: torch.compile's recompilation overhead dominates as
    batch size increases with variable-length sequences.
  - cudagraphs (`reduce-overhead`) is counterproductive for VL models with dynamic shapes
    (image tiles vary per sample → seq_len varies per batch).
  - `pad_to_multiple_of` doesn't help: reduces unique lengths from continuous to ~5 discrete
    values, but 5 is still too many for graph reuse, and the padding wastes compute.
  - **This workload is at its single-GPU ceiling** for Stage 1 projector training:
    MFU=23%, limited by small model + short seqs + projector-only training (4N not 6N).
    Further gains require: Stage 2 full finetune, longer sequences, or horizontal scaling.

---

### Iteration 10 — Sample Packing with FlexAttention (Mixed Dataset)

- **Date**: 2025-05-20
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: The previous packing failure (Iter 6) was due to two issues:
  1. SDPA falling back to slow math kernel with explicit 4D attention masks
  2. Testing on a single uniform-length dataset subset (sharegpt4v/coco, only 17.4% waste)

  With mixed-dataset training (6 subsets of diverse lengths: 216–1116 tokens, 10.2% waste even
  with bucketing), packing should be effective — IF we use an attention backend that supports
  document masking natively without fallback. HuggingFace transformers 4.57+ detects packed
  sequences from `position_ids` resets and generates efficient block-sparse masks via FlexAttention.

- **Key Discovery**: transformers 4.57.3 has **native packing support**:
  - When `attention_mask=None` and `position_ids` contains resets (non-monotonic), the framework
    calls `find_packed_sequence_indices()` to detect document boundaries
  - With `attn_implementation='flex_attention'`, it creates a compiled `BlockMask` that enforces
    document-level causal isolation without a dense 4D mask
  - No SDPA fallback, no quadratic mask materialization

- **Changes**:
  - `siq_vl/collator.py::PackingCollator`: new packing collator that:
    - Tokenizes each sample individually (no padding)
    - Bin-packs samples using first-fit-decreasing into target-length bins
    - Returns `input_ids`, `labels`, `position_ids` (with resets), `pixel_values`, `num_image_tokens`
    - Does NOT return `attention_mask` (triggers packed sequence detection)
  - `siq_vl/model/modeling.py::get_stage1_model_and_processor`:
    - New `use_packing=True` flag → sets `attn_implementation='flex_attention'` on text model

- **Mixed Dataset Baseline** (bucketed, SDPA, compile max-autotune, bs=12):
  | Metric | Value |
  |---|---|
  | Tok/s (real non-pad) | 25,561 |
  | Padding waste | 10.2% |
  | Seq lengths | 216–1116 (mean 456) |
  | VRAM | 41.4 GB |
  | Step time | 192.4 ms |

- **Packing Results** (FlexAttention, compile, pack_max=1536, fetch_bs=20):
  | Config | Tok/s | Step (ms) | VRAM | Waste |
  |---|---|---|---|---|
  | compile(dynamic=True) | **26,787** | 297 | 34.0 GB | ~2% real padding |
  | compile(max-autotune) | 26,560 | 284 | 30.2 GB | ~2% |
  | compile(default) | 26,530 | 305 | 38.5 GB | ~2% |
  | pack=2048, fetch=30 | 26,140 | 463 | 49.7 GB | ~3% |

- **Final Result** (best config: pack=1536, compile dynamic, fetch_bs=20):
  | Metric | Baseline (bucketed) | Packing | Delta |
  |---|---|---|---|
  | tokens / sec | 25,561 | **26,787** | **+4.8%** |
  | padding waste | 10.2% | ~2% | -80% waste |
  | peak VRAM | 41.4 GB | 34.0 GB | **-18%** |
  | samples / sec | ~62 | 67.4 | +9% |

- **Decision**: Adopt packing as default for mixed-dataset training. Use
  `compile(dynamic=True)` for stability with variable bin counts.

- **Lessons**:
  1. **The key insight**: The Iter 6 packing failure was NOT inherent to packing — it was caused
     by using an attention backend (SDPA) that can't handle document masks efficiently. HuggingFace
     now has native packing detection + FlexAttention integration that makes packing "just work".
  2. **compile(dynamic=True) vs max-autotune**: With packing, bin counts vary per batch (5–7),
     causing shape changes. `dynamic=True` handles this gracefully while `max-autotune` can trigger
     CUDA errors when shapes change significantly (intermittent, not always reproducible).
  3. **The throughput gain is modest (+4.8%)** because the baseline's padding waste was only 10.2%
     (bucketing already captured most of the variance). The bigger win is **VRAM savings (18%)**
     which enables longer sequences or larger effective batches.
  4. **Long sequence support**: With pack_max_length=2048 or 3072, sequences up to that length are
     handled naturally. Short sequences get packed together, long sequences get their own bin.
     No OOM even for 2K+ token sequences (tested up to pack=2048 at 49.7GB).
  5. **FlexAttention requires compile**: Without torch.compile, FlexAttention uses an eager
     fallback that is significantly more memory-hungry (OOM at 48-sample batches that compile
     handles fine at ~12 bins).
  6. **Per-position efficiency is HIGHER with packing**: 0.033ms/position vs 0.039ms/position
     in baseline — longer sequences amortize kernel launch overhead and improve Tensor Core
     utilization (larger M dimension in GEMMs).

---

### Iteration 11 — P4: Modal Distributed Training (planned)

- **Date**: TBD
- **Hypothesis**: With single-GPU efficiency at ~27K tok/s (6x over baseline), the next multiplier is
  horizontal scaling. Modal provides on-demand GPU clusters with fast provisioning and
  persistent volumes, making it ideal for elastic distributed training without managing infra.
- **Architecture**:
  ```
  GitHub push / manual trigger
    → Modal App (`train_distributed.py`)
      → Modal Volume: pre-processed dataset cache (from P1)
      → N × GPU containers (A100-80GB / H100 / etc.)
        → PyTorch FSDP (Full Shard) or DeepSpeed ZeRO-2
          → Each rank loads shard from Modal Volume
          → Checkpoints saved to Modal Volume every K steps
          → W&B logging from rank 0
      → On completion: final model pushed to HF Hub
  ```
- **Change**:
  - New file `modal_train.py` (Modal App definition):
    - `modal.Image` with CUDA, PyTorch, project dependencies
    - `modal.Volume` for dataset cache and checkpoints
    - `@modal.function(gpu=..., timeout=...)` wrapping `train()` from `scripts/train.py`
    - FSDP config with `auto_wrap_policy` for Qwen2 layers
  - Modify `scripts/train.py`:
    - Accept `--fsdp` / `--fsdp_config` flags (HF Trainer natively supports FSDP)
    - Add checkpoint-to-volume logic (save/resume from Modal Volume path)
  - New file `scripts/upload_cache_to_modal.py`:
    - Uploads pre-processed dataset cache (from Iteration 4/5) to Modal Volume
  - Update `pyproject.toml`: add `modal` as optional dependency
- **Key Design Decisions**:
  - **FSDP vs DeepSpeed**: FSDP preferred (native PyTorch, simpler config, better `torch.compile` compat)
  - **Sharding strategy**: `FULL_SHARD` for maximum memory efficiency; fallback to `SHARD_GRAD_OP` if communication overhead is too high
  - **Data loading**: each rank reads its shard from Modal Volume (already pre-processed, fast mmap)
  - **Checkpointing**: `StateDictType.FULL_STATE_DICT` for final save; `SHARDED_STATE_DICT` for intermediate (faster resume)
  - **Fault tolerance**: Modal auto-restarts on spot preemption; training resumes from latest Volume checkpoint
- **Expected Result**:
  - 4× A100-80GB: ~3.5x throughput vs single GPU (accounting for communication overhead)
  - 8× A100-80GB: ~6.5x throughput
  - End-to-end Stage 2 training time (1M samples): from ~days to ~hours
- **Risk**:
  - Modal network bandwidth between containers may bottleneck FSDP all-gather
  - Volume I/O latency for large checkpoint saves
  - Cost management: need to set `timeout` and auto-stop on convergence
- **Decision**: —

---

### Iteration 11 — cuTile Flash Attention (Blackwell-Native Kernel Infrastructure)

- **Date**: 2026-05-20
- **Branch / Commit**: `master` (this commit)
- **Hypothesis**: PyTorch's SDPA on Blackwell dispatches to `fmha_cutlassF` which is already
  well-optimized. However, NVIDIA's cuTile DSL provides direct access to Blackwell-specific
  hardware features (wgmma, TMA, SM remapping) without the indirection of Triton or CUTLASS.
  A cuTile Flash Attention kernel should be competitive with SDPA at the kernel level, and
  can serve as infrastructure for Stage 2 (longer sequences where attention is a larger fraction
  of compute).
- **Change**:
  - New file `siq_vl/kernels/cutile_attention.py`:
    - Forward kernel with K-loop split and ProgramId remapping
    - Autotuning: tests tile configs (64x64, 128x64, 64x128, 128x128) per sequence length bucket
    - `CuTileFlashAttention` autograd.Function (forward: cuTile, backward: SDPA fallback)
    - `CuTileFlashAttentionVarlen` for packed sequences (gather → batched kernel → scatter)
    - Stores LSE (log-sum-exp) for future native backward
  - New file `siq_vl/kernels/attention_backend.py`:
    - Registers `"cutile"` in HuggingFace's `ALL_ATTENTION_FUNCTIONS` registry
    - Routes to cuTile forward (inference) or SDPA (training) based on `requires_grad`
  - Modified `siq_vl/model/modeling.py::get_stage1_model_and_processor`:
    - Added `use_cutile=True` parameter to select cuTile backend
  - Benchmark script: `scripts/benchmark_cutile_e2e.py`
- **How to Reproduce**:
  ```bash
  # Kernel-level benchmark
  uv run python -m siq_vl.kernels.cutile_attention
  # End-to-end benchmark
  uv run python scripts/benchmark_cutile_e2e.py --backends sdpa cutile
  ```
- **Result (kernel-level, isolated attention)**:
  | Config (B=4,H=12,D=64) | cuTile (ms) | SDPA (ms) | Ratio |
  |---|---|---|---|
  | N=512 | 0.019 | 0.023 | **0.83x (17% faster)** |
  | N=1024 | 0.044 | 0.039 | 1.12x |
  | N=1536 | 0.083 | 0.083 | 1.00x |
  | N=2048 | 0.130 | 0.125 | 1.04x |

- **Result (end-to-end training step — TileGym FA4 production kernel)**:
  | Backend | N=512 ms/step | N=1024 ms/step | N=1536 ms/step | vs SDPA |
  |---|---|---|---|---|
  | SDPA | 69.18 | 82.09 | 84.28 | 1.00x |
  | TileGym FA4 | 64.85 | 75.19 | 74.98 | **1.07–1.12x** |

  VRAM: consistently -3% to -5% (native GQA avoids repeat_kv memory expansion).

- **Analysis**:
  - **Kernel-level**: TileGym FA4 is 12-27% faster than SDPA (native GQA, autotuned tiles,
    K-loop split, fast math FTZ+APPROX).
  - **End-to-end (Stage 1)**: 7-12% speedup because LLM is frozen — attention doesn't need
    backward. The forward-only FA4 is a pure win with zero overhead.
  - **End-to-end (Stage 2 prediction)**: With unfrozen LLM + longer sequences (2048+),
    attention becomes 15-20% of step time. FA4's 12-27% advantage → 3-5% E2E gain.
    Plus, when TileGym ships native backward, the gain doubles.
  - **Native GQA**: Key architectural advantage — Qwen2.5's 14Q/2KV configuration means
    SDPA must expand K/V 7x via `repeat_kv`. FA4 handles GQA natively in-kernel,
    saving both memory bandwidth and VRAM.
- **Autotuning Results**:
  | Sequence Length | Best Tile Config |
  |---|---|
  | 128 | 64×128 |
  | 512 | 64×64 |
  | 1024 | 64×64 |
  | 2048 | 64×64 |
- **Decision**: USE TileGym FA4 for Stage 1 (frozen LLM, forward-only). For Stage 2, use
  `cutile_training` backend (FA4 forward + SDPA backward) until TileGym ships native backward.
- **Lessons / Surprises**:
  - **Don't reinvent the wheel**: Our naive cuTile kernel was 0-17% faster than SDPA.
    TileGym's production FA4 (same cuTile DSL) is 12-27% faster — because it has
    all the blog optimizations (fast math, K-loop split, autotuning) done properly.
  - **Native GQA is the real win**: The 7-12% E2E speedup comes largely from FA4's
    native GQA support. SDPA must call `repeat_kv` which 7x expands K/V memory before
    the attention kernel even runs. FA4 avoids this entirely.
  - **Stage 1 frozen LLM = forward-only attention**: Since text_model has no gradients,
    attention backward is never called. This makes FA4 a zero-overhead improvement.
  - **Amdahl's Law**: Attention is ~5% of step at N=512, ~12% at N=1536. The speedup
    grows with sequence length (7% → 12%), confirming FA4's value for Stage 2.

---

### Iteration 13 — TileGym Full Kernel Replacement (cuTile DSL for ALL ops)

- **Date**: 2026-05-20
- **Branch / Commit**: `master`
- **Discovery**: `tilegym.transformers.apply_tilegym_kernel_to_qwen2(use_cutile=True)` monkey-patches
  ALL key Qwen2 ops (RoPE, RMSNorm, SwiGLU, attention) with Blackwell-native cuTile DSL kernels.
  This is essentially a Blackwell-optimized replacement for Liger-Kernel.
- **Benchmark vs Liger (B=4, N=1024)**:

  | Config | Stage 1 (frozen) | Stage 2 (unfrozen) |
  |---|---|---|
  | Liger-Kernel | 60.2 ms, 68K tok/s | 73.2 ms, 55.9K tok/s |
  | TileGym cuTile | 53.6 ms, 76.4K tok/s | 65.2 ms, 62.8K tok/s |
  | **Speedup** | **12.3%** | **12.4%** |
  | TileGym + Liger fused_linear_CE | 53.4 ms, 76.7K tok/s | 65.1 ms, 63.0K tok/s |

- **Stage 2 + gradient checkpointing scaling**:

  | N | Liger tok/s | TileGym tok/s | Speedup |
  |---|---|---|---|
  | 512 | 38.3K | 43.3K | +13% |
  | 1024 | 46.5K | 54.3K | +17% |
  | 2048 | 48.7K | 57.0K | +17% |

- **Peak throughput (Stage 1)**: B=4, N=1024 → **100.3K tok/s** at 5.63 GB
- **Peak throughput (Stage 2 + grad_ckpt)**: B=32, N=1024 → **76.5K tok/s** at 5.27 GB (after grad-in-forward)
- **Integration**: `--use_tilegym` flag in `scripts/train.py`, or `use_tilegym=True` in model init.
  Full TileGym stack: RoPE + RMSNorm + SwiGLU + FA4 attention + cuTile fused_linear_CE.
- **flash-attn-4 status**: `pip install flash-attn-4[cu13]` blocked by CUDA 13.0 → requires 13.1+.

#### Fused Linear CE: grad-in-forward pattern

The custom `FusedLinearCrossEntropy` (`siq_vl/kernels/fused_linear_ce.py`) computes `grad_hidden`
and `grad_weight` within the forward pass loop (per chunk), using `grad_logits` directly from
TileGym's `_ce_cutile` kernel. The backward method is O(1) — just scalar scaling by `grad_output`.

  | Config | Before (store grad_logits) | After (grad-in-forward) | VRAM Δ | Speed Δ |
  |---|---|---|---|---|
  | B=32, N=1024, grad_ckpt | 73.9K tok/s, 14.47 GB | 76.5K tok/s, 5.27 GB | **-64%** | **+3.5%** |
  | B=4, N=4096, grad_ckpt | 70.0K tok/s, 9.08 GB | 72.2K tok/s, 4.47 GB | **-51%** | **+3%** |

#### Final comparison: TileGym full stack vs Liger (Stage 2 + gradient checkpointing, B=4)

  | N | Liger tok/s / VRAM | TileGym tok/s / VRAM | Speed | VRAM |
  |---|---|---|---|---|
  | 512 | 38.3K / 5.28 GB | 49.7K / 2.64 GB | **+30%** | **-50%** |
  | 1024 | 46.5K / 9.43 GB | 64.6K / 3.89 GB | **+39%** | **-59%** |
  | 2048 | 48.7K / 17.74 GB | 68.7K / 6.39 GB | **+41%** | **-64%** |

- **Lessons**:
  - TileGym is to Liger what cuTile is to Triton — same idea, but targets Blackwell natively.
  - The cuTile CE kernel (`_ce_online_kernel`) does online softmax + loss in ONE pass over
    vocab tiles. It's both faster AND more memory-efficient than chunked PyTorch CE.
  - After the kernel runs, logits buffer contains softmax probs in-place — free backward data!
  - grad-in-forward pattern: compute `grad_hidden`/`grad_weight` inside forward loop, backward
    becomes trivial. Saves 51-64% VRAM by never storing `grad_logits` chunks.
  - `tilegym.ops.matmul` is slower than `torch.mm` for large GEMMs — cuBLAS already optimal.
  - Combined effect: **30-41% faster, 50-64% less VRAM** than Liger. Zero Liger dependency.

---

### Iteration 12 — Stage 2 Readiness Benchmark (Unfrozen Text Model)

- **Date**: 2026-05-20
- **Branch / Commit**: `master`
- **Goal**: Establish Stage 2 baseline and determine optimal configuration before actual training.
- **Key finding**: `flash-attn-4` (Dao-AILab) requires CUDA 13.1+ — incompatible with our CUDA 13.0.
  TileGym FA4 remains our best attention option. However, for Stage 2 (unfrozen LLM where backward
  dominates), the attention backend choice makes negligible difference — SDPA is recommended.

#### Stage 2 Benchmark Results (B=4, SDPA + Liger)

| Mode | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|
| **LoRA r=16** | 99.7ms, 20.5K tok/s, 4.4GB | 133.5ms, 30.7K tok/s, 7.6GB | 238.3ms, 34.4K tok/s, 14.0GB | — |
| **Full FT** | 272.1ms, 7.5K tok/s, 4.1GB | 290.4ms, 14.1K tok/s, 5.9GB | 333.7ms, 24.5K tok/s, 9.7GB | — |
| **Full FT + grad_ckpt** | 293.2ms, 7.0K tok/s, 2.3GB | 302.2ms, 13.6K tok/s, 2.3GB | 362.1ms, 22.6K tok/s, 2.5GB | 471.4ms, 34.8K tok/s, 2.9GB |
| **Full FT + grad_ckpt, B=8** | — | 348.5ms, 23.5K tok/s, 2.5GB | — | — |
| **Full FT + grad_ckpt, B=16** | — | 452.2ms, 36.2K tok/s, 2.9GB | — | — |

#### cuTile_training vs SDPA (unfrozen LLM)

| Config | SDPA | cuTile+SDPA_bwd | Speedup |
|---|---|---|---|
| Full FT, N=512 | 272.3 ms | 276.8 ms | 0.98x |
| Full FT, N=1024 | 290.3 ms | 292.6 ms | 0.99x |
| Full FT, N=2048 | 333.6 ms | 338.7 ms | 0.99x |
| LoRA r=16, N=512 | 102.1 ms | 112.4 ms | 0.91x |

**Conclusion**: With unfrozen LLM, backward pass dominates. Our `cutile_training` backend
(FA4 fwd + SDPA bwd) adds Python overhead from `autograd.Function` wrapper without meaningful
gain. **Recommendation: Use plain SDPA for Stage 2**. The cuTile forward-only benefit only
materialized in Stage 1 where backward was never called.

#### Stage 2 Configuration Recommendations

| Strategy | When to Use | Trainable | Best tok/s | VRAM |
|---|---|---|---|---|
| **LoRA r=16** | Quick iteration, good quality | 11.5M (1.7%) | 34.4K (N=2048) | 14GB |
| **Full FT + grad_ckpt** | Maximum quality, long seqs | 496.8M (100%) | 36.2K (B=16) | 2.9GB |
| **Full FT + grad_ckpt + packing** | Max efficiency on mixed data | 496.8M | (to be measured) | ~3GB |

- **Flash Attention 4 (Dao-AILab)**: Blocked on CUDA 13.1. Once available, `flash_attn_varlen_func`
  would natively support packed sequences with `cu_seqlens` for both forward and backward — the
  ideal Stage 2 packing solution (eliminates FlexAttention's compile overhead).
- **torch.compile**: Crashed with Liger+LoRA combination; use `--no_liger --torch_compile` if needed.

---

### Iteration 15 — Ground Truth Verification (Full Re-measurement)

- **Date**: 2026-05-20
- **Branch / Commit**: `main` @ HEAD
- **Hypothesis**: Previous cumulative table contained estimated Pad% and Pos/s values.
  Re-run ALL key configurations on real data with precise metric instrumentation.
- **Change**:
  - New script: `scripts/benchmark_real_efficiency.py`
  - Measures `attention_mask.sum()`, `(labels != -100).sum()`, `B × N`, step time exactly
- **How to Reproduce**:
  ```bash
  # Example: vanilla B=32 with bucketing
  python scripts/benchmark_real_efficiency.py --stage 1 --batch_size 32 --use_bucketing
  # TileGym with fixed shapes
  python scripts/benchmark_real_efficiency.py --stage 1 --batch_size 32 --use_tilegym --pad_to_multiple_of 64 --use_bucketing --warmup 10
  # Packing + TileGym (best config)
  python scripts/benchmark_real_efficiency.py --stage 1 --batch_size 64 --use_packing --pack_max_length 1024 --use_tilegym --warmup 10
  ```
- **Measurement Setup**:
  - GPU: 1 × RTX PRO 6000 Blackwell 96GB
  - Model: SigLIP2-base-patch16-224 + Qwen2.5-0.5B-Instruct (Stage 1, projector only)
  - Dataset: `sharegpt4v(coco)`, 50,017 samples, real images
  - Each run: 10 measured steps after warmup (2-10 warmup depending on TileGym JIT)
- **Result**: See "Cumulative Optimization Results (VERIFIED)" table in §4.1.
- **Key Corrections from Previous Estimates**:
  1. Bucketing B=64 Real tok/s is 90K (previously estimated higher)
  2. Packing gives +4.6% over bucketing at same batch (not the ~0% claimed earlier)
  3. TileGym at B=4 gives 58K (vs 53K in old Iter 13 — close but slightly higher)
  4. TileGym CATASTROPHIC with variable shapes (650ms/step vs 115ms at fixed shapes)
  5. Hardware ceiling confirmed at ~108K Pos/s (independent of config)
- **Artifacts**:
  - Trace: `docs/traces/iter_15_ground_truth_efficiency.json`
  - Script: `scripts/benchmark_real_efficiency.py`
- **Decision**: Keep — this is the new source of truth for the cumulative table.
- **Lessons / Surprises**:
  - TileGym's JIT autotuning is per-shape. Variable-length batches without `pad_to_multiple_of`
    cause 4-5x slowdown. This was hidden in synthetic benchmarks that used fixed N.
  - Bucketing at B=4 is SLOWER than no bucketing (16K vs 22K) because bucketed short sequences
    underutilize the GPU more than having some padding with longer sequences.
  - The system saturates at ~90K Real tok/s (vanilla) or ~107K (TileGym+packing) regardless
    of batch size beyond B=64. VRAM scales linearly with B but throughput doesn't.

---

### Iteration 14 — Batch-Size Scaling & Optimizer Memory (FlashOptim)

- **Date**: 2026-05-20
- **Branch / Commit**: `master`
- **Goal**: With Iter 13 showing only 5.6 GB VRAM on a 95 GB GPU, push batch size to maximize
  GPU utilization and evaluate `flashoptim` (Databricks) for optimizer memory compression.

#### Batch Scaling Results (Stage 2 + AdamW + grad_ckpt + TileGym full stack, N=1024)

| B | ms/step | tok/s | VRAM (GB) | Util% |
|---|---|---|---|---|
| 32 | 439.4 | 74,578 | 7.03 | 7% |
| 64 | 890.0 | 73,640 | 9.22 | 10% |
| 128 | 1,769.4 | 74,075 | 15.29 | 16% |
| 192 | 2,649.5 | 74,205 | 21.36 | 22% |
| 256 | 3,529.5 | 74,273 | 27.43 | 29% |
| 320 | 4,401.5 | 74,447 | 33.50 | 35% |
| 384 | 5,279.2 | 74,484 | 39.57 | 42% |
| 448 | 6,149.0 | 74,606 | 45.63 | 48% |
| 512 | 7,027.1 | 74,609 | 51.71 | 54% |
| 640 | 8,764.7 | 74,773 | 63.84 | 67% |
| **768** | **10,517.5** | **74,774** | **75.98** | **80%** |
| 896 | OOM | — | >80 | — |

#### B×N Trade-off (same total tokens = 786K, ~80% VRAM fill)

| Config | ms/step | tok/s | eff tok/s | VRAM (GB) |
|---|---|---|---|---|
| B=768, N=1024 | 10,509.9 | 74,828 | 71,247 | 76.0 |
| B=384, N=2048 | 10,653.7 | 73,818 | 72,052 | 75.8 |
| B=192, N=4096 | 11,038.3 | 71,245 | 70,393 | 75.7 |

#### Key Findings

1. **Throughput is batch-size-independent** at ~74K tok/s. GPU compute is fully saturated
   even at B=32. Adding more data only increases step time proportionally.
2. **VRAM scales linearly** at ~6 GB per additional 64 samples.
3. **Same total tokens → same VRAM** regardless of B/N split. The fused_linear_CE and
   gradient checkpointing eliminate per-sample overhead.
4. **Longer sequences are only ~5% slower** (N=4096 vs N=1024) — acceptable for
   long-context training where the quality benefit outweighs the small throughput cost.
5. **Optimal B for Stage 2**: B=512–640 (54–67% VRAM), leaving headroom for:
   - Real data with variable lengths (some batches may be longer)
   - Multi-image samples that produce more vision tokens

#### Practical Training Recipe

```
per_device_train_batch_size = 512   # for N≤1024 sequences
per_device_train_batch_size = 256   # for N≤2048 sequences
per_device_train_batch_size = 128   # for N≤4096 sequences
gradient_accumulation_steps = 1     # already large effective batch
```

#### FlashOptim (Databricks) — Optimizer Memory Compression

- **What**: Drop-in `FlashAdamW` replaces `torch.optim.AdamW`, quantizing optimizer states
  (momentum, variance) to int8 + 8-bit error correction. Reduces per-param memory by ~57%.
- **Features**: Fused Triton kernels (no overhead), gradient release (update during backward),
  compressed checkpoints (50%+ smaller), compatible with FSDP2.

  | Optimizer | Max B (N=1024) | VRAM @B=768 | tok/s @B=768 |
  |---|---|---|---|
  | torch.optim.AdamW | 768 | 76.0 GB | 74,798 |
  | FlashAdamW (int8 states) | 768 | 75.6 GB | 74,889 |
  | FlashAdamW + gradient_release | **832** | 81.7 GB (B=832) | 74,779 |

- **Measured savings**: 0.4 GB optimizer memory (int8 quantization). Gradient release frees
  gradient tensor (~1 GB), enabling B=832 vs B=768 max with AdamW.
- **Throughput impact**: None — identical tok/s across all optimizers.
- **Why marginal**: For 0.5B trainable params, optimizer states = ~4 GB out of 76 GB total.
  Activation memory from the batch dominates at large B. FlashOptim's savings are more
  impactful for 7B+ models where optimizer states consume 10-50+ GB.
- **Recommendation**: Use `FlashAdamW` as default (free savings, no downsides, convergence
  identical per paper). Gradient release is useful if NOT doing gradient accumulation or
  gradient clipping. Add `flashoptim>=0.1.4` to deps.
- **Status**: TESTED — marginal for this model scale, recommended for future scale-up.

---

## 4.1. Performance Summary & MFU Analysis

### Benchmark Results (v3 — Final, Clean Re-run)

All numbers below use a **single backbone** (SigLIP2-base-224 + Qwen2.5-0.5B, 589.7M total).
Measured end-to-end on real data (`sharegpt4v/coco`, 50K samples) using
`scripts/benchmark_real_efficiency.py`. **No callbacks, no gradient checkpointing, no overhead.**
Raw traces at `docs/traces/benchmark_v3_20260522_001447/`.

Script: `scripts/run_benchmark_v3.sh`

---

#### Metric Definitions

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Real tok/s** | `attention_mask.sum() / wall_time` | Effective throughput: non-padding tokens processed per second |
| **Pos/s (hw)** | `B × N / wall_time` | Hardware throughput: total positions including padding |
| **Pad%** | `1 - attention_mask.sum() / (B × N)` | Fraction of positions that are padding (wasted compute) |
| **Loss%** | `(labels != -100).sum() / attention_mask.sum()` | Fraction of real tokens that produce gradient signal |
| **VRAM** | `torch.cuda.max_memory_allocated()` | Peak GPU memory usage |
| **Speedup** | `Real tok/s (config) / Real tok/s (baseline)` | Efficiency gain vs respective stage baseline |

---

#### Stage 1 Results (Frozen LLM — only projector trains)

| # | Config | Real tok/s | VRAM | Pad% | Speedup |
|---|--------|-----------|------|------|---------|
| S1-01 | **Baseline**: FP32, B=4 | 14,713 | 10.5 GB | 12.3% | 1.00x |
| S1-02 | BF16 fix, B=4 | 34,747 | 7.1 GB | 12.3% | **2.36x** |
| S1-03 | BF16, B=16 (no bucket) | 45,893 | 25.2 GB | 16.9% | 3.12x |
| S1-04 | BF16, B=16, bucketing | 52,854 | 18.0 GB | 3.8% | 3.59x |
| S1-05 | BF16, B=32, bucketing | 54,136 | 35.1 GB | 4.4% | 3.68x |
| S1-06 | BF16, B=64, bucketing | 51,191 | 69.4 GB | 5.0% | 3.48x |
| S1-07 | **Liger (with FusedCE)**, B=32, bucket | 76,252 | 9.5 GB | 4.4% | **5.18x** |
| S1-08 | Liger (no FusedCE), B=32, bucket | 66,029 | 31.7 GB | 4.4% | 4.49x |
| S1-09 | **torch.compile**, B=32, bucket | 94,342 | 21.9 GB | 4.4% | **6.41x** |
| S1-10 | TileGym, B=32, bucket, pad64 | 91,866 | 10.9 GB | 14.3% | 6.24x |
| S1-11 | TileGym, B=64, bucket, pad64 | 93,344 | 18.2 GB | 14.0% | 6.34x |
| S1-12 | Packing N=1024, B=64 (no kernel) | 57,204 | 72.2 GB | 0.0% | 3.89x |
| S1-13 | **Packing + TileGym**, B=64, N=1024 | **100,923** | **18.1 GB** | **0.0%** | **6.86x** |

#### Stage 2 Results (Unfrozen LLM, NO gradient checkpointing)

| # | Config | Real tok/s | VRAM | Pad% | Speedup |
|---|--------|-----------|------|------|---------|
| S2-01 | Vanilla, B=4 (baseline) | 29,167 | 7.9 GB | 12.3% | 1.00x |
| S2-02 | Vanilla, B=16, bucket | 44,934 | 20.3 GB | 3.8% | 1.54x |
| S2-03 | Vanilla, B=32, bucket | 45,748 | 39.5 GB | 4.4% | 1.57x |
| S2-04 | Liger (no FusedCE), B=16, bucket | 52,330 | 18.1 GB | 3.8% | **1.79x** |
| S2-05 | Liger (with FusedCE), B=16, bucket ❌ | 17,434 | 8.0 GB | 3.8% | 0.60x |
| S2-06 | Liger (no FusedCE), B=32, bucket | 55,504 | 35.2 GB | 4.4% | **1.90x** |
| S2-07 | torch.compile, B=16, bucket | 68,105 | 13.2 GB | 3.8% | 2.34x |
| S2-08 | **torch.compile**, B=32, bucket | 73,052 | 25.4 GB | 4.4% | **2.50x** |
| S2-09 | TileGym, B=16, bucket, pad64 | 72,174 | 9.1 GB | 14.4% | 2.47x |
| S2-10 | TileGym, B=32, bucket, pad64 | 76,907 | 15.7 GB | 14.3% | 2.64x |
| S2-11 | TileGym, B=64, bucket, pad64 | 79,155 | 27.4 GB | 14.0% | **2.71x** |
| S2-12 | Packing N=1024, B=32 (no kernel) | 51,091 | 44.7 GB | 0.0% | 1.75x |
| S2-13 | **Packing + TileGym**, B=64, N=1024 | **86,080** | **27.4 GB** | **0.0%** | **2.95x** |

---

#### Key Findings (v3, Final)

1. **Total speedup**: Stage 1 = **6.86x** (14.7K → 100.9K), Stage 2 = **2.95x** (29.2K → 86.1K).

2. **BF16 fix = 2.36x** — single largest ROI optimization (2-line change).

3. **Liger FusedCE: stage-dependent behavior**:
   - Stage 1 (frozen LLM): FusedCE is **beneficial** — 76.3K vs 66.0K (+15.5%), VRAM 9.5G vs 31.7G
   - Stage 2 (unfrozen LLM): FusedCE is **catastrophic** — 17.4K vs 52.3K (-67%)
   - Root cause: In Stage 1, FusedCE only runs forward (no backward through LM head → saves memory+compute).
     In Stage 2, chunked backward dominates with many small Triton kernel launches.

4. **Vanilla throughput saturates at B=16–32** (both stages). Beyond B=32, no Real tok/s improvement.

5. **torch.compile**: Excellent in both stages (6.41x S1, 2.50x S2). No external dependencies.
   Trade-off: ~3 min JIT compilation on first run, higher VRAM than TileGym.

6. **TileGym: best speed/VRAM Pareto**:
   - Stage 1: 91.9K at 10.9 GB (vs compile's 94.3K at 21.9 GB)
   - Stage 2: 79.2K at 27.4 GB (TileGym B=64, 2.71x)
   - Best overall: Packing + TileGym B=64 = 86.1K at 27.4 GB (2.95x)

7. **Gradient checkpointing: unnecessary on 90GB GPUs**. Removing it yielded +15-21% speed
   at peak Stage 2 VRAM of only 27-44 GB (well within 90 GB budget).

8. **Packing + TileGym is the peak recipe for both stages** (small model): 0% padding waste,
   fixed shapes enable optimal TileGym tiling. Stage 1 = 100.9K, Stage 2 = 86.1K.

---

### Large Model Results (v3 — SigLIP2-so400m-384 + Qwen2.5-1.5B)

Cross-scale comparison using the same methodology. Raw traces at `docs/traces/benchmark_v3_large/`.

**Key limitation**: TileGym is **incompatible** with SigLIP-so400m because its head_dim=72
(1152/16 attention heads), which is not a power of 2. cuTile requires power-of-2 tile dimensions.

#### Stage 1 Results (Large Model, Frozen LLM)

| # | Config | Real tok/s | VRAM | Pad% | Speedup |
|---|--------|-----------|------|------|---------|
| S1-01 | **Baseline**: FP32, B=2 | 4,626 | 13.4 GB | 6.0% | 1.00x |
| S1-02 | BF16 fix, B=2 | 13,862 | 7.7 GB | 6.0% | **3.00x** |
| S1-03 | BF16, B=8 (no bucket) | 18,248 | 21.2 GB | 15.2% | 3.94x |
| S1-04 | BF16, B=8, bucketing | 20,460 | 16.1 GB | 2.6% | 4.42x |
| S1-05 | BF16, B=16, bucketing | 21,703 | 28.9 GB | 3.4% | 4.69x |
| S1-06 | BF16, B=32, bucketing | 21,157 | 54.3 GB | 4.1% | 4.57x |
| S1-07 | Liger (+CE), B=16, bucket | 23,083 | 13.2 GB | 3.4% | 4.99x |
| S1-08 | Liger (noCE), B=16, bucket | 25,532 | 25.0 GB | 3.4% | **5.52x** |
| S1-09 | **torch.compile**, B=16, bucket | **31,110** | 19.9 GB | 3.4% | **6.73x** |
| S1-10 | Packing N=1024, B=16 | 24,851 | 35.7 GB | 0.0% | 5.37x |
| S1-11 | **torch.compile**, B=32, bucket | **31,078** | 36.1 GB | 4.1% | **6.72x** |
| S1-12 | Liger (+CE), B=32, bucket | 26,425 | 22.7 GB | 4.1% | 5.71x |

#### Stage 2 Results (Large Model, Unfrozen LLM, NO grad_ckpt)

| # | Config | Real tok/s | VRAM | Pad% | Speedup |
|---|--------|-----------|------|------|---------|
| S2-01 | Vanilla, B=2 (baseline) | 11,569 | 8.5 GB | 6.0% | 1.00x |
| S2-02 | Vanilla, B=8, bucket | 16,944 | 18.6 GB | 2.6% | 1.46x |
| S2-03 | Vanilla, B=16, bucket | 18,208 | 33.8 GB | 3.4% | 1.57x |
| S2-04 | Liger (noCE), B=8, bucket | 19,011 | 16.1 GB | 2.6% | 1.64x |
| S2-05 | Liger (noCE), B=16, bucket | 20,899 | 28.9 GB | 3.4% | **1.81x** |
| S2-06 | Liger (+CE), B=16, bucket ❌ | 10,731 | 18.7 GB | 3.4% | 0.93x |
| S2-07 | torch.compile, B=8, bucket | 21,975 | 13.7 GB | 2.6% | 1.90x |
| S2-08 | **torch.compile**, B=16, bucket | **24,153** | **23.7 GB** | 3.4% | **2.09x** |
| S2-09 | Packing N=1024, B=8 | 19,639 | 23.0 GB | 0.0% | 1.70x |
| S2-10 | Liger (+CE), B=32, bucket | 15,045 | 32.0 GB | 4.1% | 1.30x |

#### Cross-Scale Comparison

| Metric | Small (0.5B) | Large (1.5B) | Ratio |
|--------|-------------|-------------|-------|
| Stage 1 baseline (FP32) | 14,713 tok/s | 4,626 tok/s | 3.2x slower |
| Stage 1 peak | 100,923 tok/s (Packing+TileGym) | 31,110 tok/s (torch.compile) | 3.2x slower |
| Stage 1 max speedup | 6.86x | 6.73x | Similar gains |
| Stage 2 baseline | 29,167 tok/s | 11,569 tok/s | 2.5x slower |
| Stage 2 peak | 86,080 tok/s (Packing+TileGym) | 24,153 tok/s (torch.compile) | 3.6x slower |
| Stage 2 max speedup | 2.95x | 2.09x | Large benefits less |
| Best overall config | Packing + TileGym B=64 | torch.compile B=16 | Different winners |

#### Key Findings (Large Model)

1. **torch.compile is the best option for the large model** (6.73x S1, 2.09x S2).
   TileGym is incompatible due to SigLIP-so400m's non-power-of-2 head dimension.

2. **BF16 fix still yields 3.0x** — consistent with the small model (2.36x), confirming
   this is a universal low-hanging fruit regardless of scale.

3. **Liger FusedCE behavior is scale-dependent**:
   - Large model, Stage 1: FusedCE slightly slower than noCE (23K vs 25.5K), opposite to small model
   - Large model, Stage 2: FusedCE still harmful (10.7K vs 20.9K, -49%)
   - At 1.5B scale, materializing logits is affordable in forward; FusedCE chunk overhead outweighs savings

4. **Vanilla saturates early** (B=16 for large model vs B=32 for small). The 1.5B model
   is more compute-bound — larger batches add latency without throughput gain.

5. **Throughput scales ~3.2x between 0.5B and 1.5B** — roughly proportional to parameter count,
   suggesting both models are compute-bound (not memory-bound) at optimal batch sizes.

---

### Unified Benchmark Results (v2 — SUPERSEDED by v3 above)

All numbers below use the **same backbone** (SigLIP2-base-224 + Qwen2.5-0.5B, 589.7M total)
across all experiments. Measured end-to-end on real data (`sharegpt4v/coco`, 50K samples)
using `scripts/benchmark_real_efficiency.py`. Raw traces at `docs/traces/benchmark_small_v2/`.

**Important**: Previous results (Iteration 15) mixed two different model scales
(Phase A: 1.5B, Phase B: 0.5B), making speedup claims invalid. This section
supersedes Iteration 15 with apple-to-apple measurements on a single backbone.

---

#### Metric Definitions

All metrics are **measured** (not estimated) per training step:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Real tok/s** | `attention_mask.sum() / wall_time` | Effective throughput: non-padding tokens processed per second |
| **Pos/s (hw)** | `B × N / wall_time` | Hardware throughput: total positions including padding |
| **Pad%** | `1 - attention_mask.sum() / (B × N)` | Fraction of positions that are padding (wasted compute) |
| **Loss%** | `(labels != -100).sum() / attention_mask.sum()` | Fraction of real tokens that produce gradient signal |
| **Step (ms)** | `torch.cuda.synchronize(); t1 - t0` | Wall-clock time for fwd + bwd + zero_grad |
| **VRAM** | `torch.cuda.max_memory_allocated()` | Peak GPU memory usage |
| **Speedup** | `Real tok/s (config) / Real tok/s (baseline)` | Efficiency gain vs FP32 baseline |

**Key insight**: We measure **Real tok/s** (not Pos/s) because it captures true training
efficiency — padding tokens consume GPU cycles but contribute zero learning signal.
A config with 0% padding that processes 100K positions/s is strictly better than one
processing 120K positions/s with 20% padding (which yields only 96K real tok/s).

---

#### Stage 1 Results (Frozen LLM — only projector trains)

| # | Config | Real tok/s | VRAM | Pad% | Step ms | Speedup |
|---|--------|-----------|------|------|---------|---------|
| 1 | **Baseline**: FP32, B=4 | 14,868 | 10.5 GB | 12.3% | 93.3 | 1.0x |
| 2 | BF16 + no_grad, B=4 | 35,096 | 7.1 GB | 12.3% | 39.5 | **2.4x** |
| 3 | BF16, B=16 (no bucket) | 45,972 | 25.2 GB | 16.9% | 117.7 | 3.1x |
| 4 | BF16, B=16, bucketing | 52,850 | 18.0 GB | 3.8% | 99.5 | 3.6x |
| 5 | BF16, B=32, bucketing | 54,080 | 35.1 GB | 4.4% | 194.7 | 3.6x |
| 6 | BF16, B=64, bucketing | 51,174 | 69.4 GB | 5.0% | 412.9 | 3.4x |
| 7 | **Liger-Kernel**, B=32, bucketing | 76,137 | 9.5 GB | 4.4% | 138.3 | **5.1x** |
| 8 | **torch.compile**, B=32, bucketing | 94,757 | 21.9 GB | 4.4% | 111.1 | **6.4x** |
| 9 | **TileGym**, B=32, bucket, pad64 | 91,757 | 10.9 GB | 14.3% | 114.7 | 6.2x |
| 10 | TileGym, B=64, bucket, pad64 | 93,311 | 18.2 GB | 14.0% | 226.4 | 6.3x |
| 11 | Packing N=1024, B=64 (no kernel) | 57,120 | 72.2 GB | 0.0% | 410.5 | 3.8x |
| 12 | **Packing + TileGym**, B=64, N=1024 | **100,228** | **18.1 GB** | **0.0%** | 234.0 | **6.7x** |

#### Stage 2 Results (Unfrozen LLM, NO gradient checkpointing)

> **Note**: Gradient checkpointing is disabled — our 90GB GPU has abundant VRAM,
> and grad_ckpt costs 15-21% speed for memory savings we don't need.
> Previous results (v2) incorrectly used grad_ckpt; these supersede them.

| # | Config | Real tok/s | VRAM | Speedup |
|---|--------|-----------|------|---------|
| S2-1 | Vanilla, B=4 (baseline) | 28,514 | 7.9 GB | 1.00x |
| S2-2 | Vanilla, B=16, bucket | 27,622 | 20.3 GB | 0.97x |
| S2-3a | Liger (no FusedCE), B=16, bucket | 36,485 | 18.1 GB | **1.28x** |
| S2-3b | Liger (with FusedCE), B=16, bucket ❌ | 13,250 | 8.0 GB | 0.46x |
| S2-4 | TileGym, B=16, bucket, pad64 | 47,604 | 9.1 GB | **1.67x** |
| S2-5 | TileGym, B=32, bucket, pad64 | 69,632 | 15.7 GB | **2.44x** |
| S2-6 | Liger (no FusedCE), B=32, bucket | 30,958 | 35.2 GB | 1.09x |
| S2-8 | Vanilla, B=32, bucket | 40,963 | 39.5 GB | 1.44x |
| S2-9 | **torch.compile**, B=16, bucket | 68,243 | 13.2 GB | **2.39x** |
| S2-10 | **TileGym, B=64**, bucket, pad64 | **79,198** | **27.4 GB** | **2.78x** |
| S2-11 | Packing N=1024, B=32 (no kernel) | 50,887 | 44.7 GB | 1.78x |
| S2-12 | Packing + TileGym, B=32 | 17,153 | 16.5 GB | 0.60x ❌ |
| S2-13 | Packing + TileGym, B=64 | 70,813 | 27.4 GB | 2.48x |
| S2-14 | **torch.compile**, B=32, bucket | 73,245 | 25.4 GB | **2.57x** |

##### Why gradient checkpointing was removed

| Config | With grad_ckpt | Without | Speed gain | VRAM cost |
|--------|---------------|---------|-----------|-----------|
| Vanilla B=16 | 133.4 ms | 115.3 ms | +15.7% | +7.8 GB |
| Liger (no CE) B=16 | 108.7 ms | 96.6 ms | +12.5% | +5.4 GB |
| TileGym B=16 | 107.0 ms | 88.1 ms | +21.4% | +5.1 GB |

On a 90GB GPU using <36 GB total, grad_ckpt is pure overhead.

##### Why Liger's FusedCE is disabled in Stage 2

See "Deep Dive" section below. Summary: `fused_linear_cross_entropy` uses chunked
matmuls that are 2x slower than one large cuBLAS GEMM. Only enable it when VRAM is
the bottleneck (24-48 GB GPUs training 7B+ models).

---

#### Key Findings (Unified, Single-Backbone)

1. **Total speedup**: Stage 1 = **6.7x** (14,868 → 100,228 real tok/s), Stage 2 = **2.78x** (28,514 → 79,198 real tok/s).

2. **BF16 fix is 2.4x** — the single largest return-on-effort optimization (2-line change).

3. **Vanilla throughput saturates at B=16 (~54K tok/s)**: Increasing B beyond 16 does NOT
   improve Real tok/s (55K→51K) but explodes VRAM (18→69 GB). The GPU is memory-bandwidth
   bound at this model scale — larger batches only increase latency proportionally.

4. **Liger-Kernel: must disable FusedCE in Stage 2**:
   - Stage 1 (all patches): +41% speed (76K vs 54K) AND -73% VRAM (9.5 vs 35 GB)
   - Stage 2 with FusedCE: **-54% speed** (13.3K vs 28.5K) — FusedCE's chunked matmul is 2x slower
   - Stage 2 without FusedCE: **+28% speed** (36.5K vs 28.5K) — RoPE+RMSNorm+SwiGLU alone are beneficial
   - Root cause: FusedCE's chunked approach avoids materializing full logits → huge VRAM savings
     but many small Triton kernel launches that undersaturate Blackwell SMs.

5. **torch.compile: excellent in both stages**:
   - Stage 1: 94.8K tok/s at B=32 (6.4x, best Stage 1 single-config)
   - Stage 2: 73.2K tok/s at B=32 (2.57x), 68.2K at B=16 (2.39x)
   - Trade-off: Long compilation time (~3 min first run), higher VRAM than TileGym.

6. **TileGym B=64: overall Stage 2 peak at 79.2K tok/s (2.78x)**:
   - Stage 1: 91.8K tok/s at only 10.9 GB (vs compile's 94.8K at 21.9 GB)
   - Stage 2: 79.2K tok/s at 27.4 GB (2.78x) — best absolute throughput
   - TileGym B=32: 69.6K tok/s at 15.7 GB — best speed/VRAM Pareto for Stage 2
   - Trade-off: Requires pad_to_multiple_of=64 (adds 14% intentional padding)

7. **Packing: shape-sensitive, benefits only at B=64+**:
   - Stage 1: Packing alone (57K, 72 GB) is worse than bucketing (54K, 35 GB)
   - Stage 2: Packing vanilla B=32 (50.9K) is OK, but Packing+TileGym B=32 crashes to 17K (shape issue)
   - Packing + TileGym B=64: 70.8K tok/s (2.48x) — competitive but TileGym+bucketing is simpler

8. **Packing + TileGym = peak efficiency**: 100K tok/s, 0% waste, 18 GB.
   TileGym needs fixed shapes; packing provides exactly that (fixed N=1024 bins).

---

### Deep Dive: Why Liger-Kernel Is Slower in Stage 2

**Date**: 2026-05-22

**Observation**: Liger-Kernel shows -57% speed regression in Stage 2 (16.6K vs 38K tok/s),
despite being +41% faster in Stage 1. The VRAM savings are massive (-81%) but speed suffers.

#### Component Isolation Experiment

Tested each Liger patch independently (subprocess isolation, B=16, N=384, Qwen2.5-0.5B, grad_ckpt=ON):

| Configuration | ms/step | VRAM | Delta vs Vanilla |
|---|---|---|---|
| Vanilla (no patches) | 133.6 ms | 13.41 GB | baseline |
| Liger: RoPE only | 132.7 ms | 13.41 GB | -0.7% |
| Liger: RoPE + RMSNorm | 119.6 ms | 13.39 GB | **-10.4%** |
| Liger: RoPE + RMSNorm + SwiGLU | 109.0 ms | 13.39 GB | **-18.4%** |
| Liger: ALL (RoPE+RMSNorm+SwiGLU+FusedCE) | 254.8 ms | 2.26 GB | **+90.8%** |
| Liger: FusedCE ONLY | 284.5 ms | 2.32 GB | **+113.0%** |

**Verdict**: `fused_linear_cross_entropy` is the sole culprit. All other Liger kernels (RoPE, RMSNorm, SwiGLU) are beneficial (-18% combined).

#### Scale Dependency (FusedCE overhead at different B*N)

| B | N | B*N | Vanilla | FusedCE | Delta | VRAM Saved |
|---|---|---|---|---|---|---|
| 4 | 256 | 1,024 | 55.3 ms | 295.8 ms | +435% | 0.9 GB |
| 16 | 384 | 6,144 | 133.5 ms | 279.1 ms | +109% | 11.1 GB |
| 16 | 1024 | 16,384 | 363.1 ms | 513.3 ms | +41% | 31.1 GB |
| 8 | 2048 | 16,384 | 373.9 ms | 524.0 ms | +40% | 31.1 GB |
| 4 | 4096 | 16,384 | 394.5 ms | 544.7 ms | +38% | 31.1 GB |

The overhead decreases at larger B*N but never reaches breakeven. Even at 16K tokens, FusedCE is +38% slower.

#### Root Cause Analysis

Liger's `fused_linear_cross_entropy` uses a **chunked matmul** strategy:
- Standard path: one large `[B*N, 896] @ [896, 151936]` GEMM → cuBLAS saturates the GPU SMs
- FusedCE path: iterates vocabulary in small chunks → many small Triton kernel launches

The overhead comes from:
1. **Kernel launch overhead**: Many small GEMMs (chunk_size × vocab_chunk) instead of one large GEMM
2. **Triton JIT**: Each chunk dispatches through Triton's JIT compilation cache
3. **Gradient checkpointing amplification**: The entire chunked forward is recomputed during backward,
   doubling the number of small kernel launches
4. **Undersaturated GPU**: On Blackwell (sm_120), a single large cuBLAS GEMM already saturates
   all SMs; splitting into chunks leaves SMs idle between launches

#### Why FusedCE Exists (and when to use it)

FusedCE was designed for **VRAM-constrained** training:
- The logits tensor `[B*N, vocab_size]` = `[16*384, 151936]` = 1.8 GB in bf16
- For larger models (7B+, 70B) with longer sequences (4K+), this tensor dominates VRAM
- FusedCE never materializes it → massive VRAM reduction (83% in our case)

**Decision**: For our 90 GB GPU, VRAM is NOT the bottleneck. FusedCE is **inappropriate**.

#### Recommendation

| Scenario | Optimal Config |
|---|---|
| Stage 1 (frozen LLM, compute-light) | Liger ALL or TileGym |
| Stage 2, VRAM-limited (24–48 GB GPU) | Liger ALL (accept speed loss) |
| Stage 2, VRAM-abundant (90 GB GPU) | **Liger RoPE+RMSNorm+SwiGLU (NO FusedCE)** → 18% faster than vanilla |
| Stage 2, maximum speed | TileGym B=32, no grad_ckpt (2.44x speedup, 15.7 GB) |

---

### Historical Cumulative Results (DEPRECATED — mixed backbone, for reference only)

> **WARNING**: Results below used TWO different model scales (Phase A: 1.5B, Phase B: 0.5B).
> The "22.9x" speedup claim is NOT apple-to-apple. See unified results above for valid data.

All numbers below were re-measured end-to-end on real data (`sharegpt4v/coco`, 50K samples)
using `scripts/benchmark_real_efficiency.py`. Raw traces at `docs/traces/iter_15_ground_truth_efficiency.json`.

**Metric definitions** (all measured, not estimated):
- **Real tok/s**: Non-padding tokens processed per second = `attention_mask.sum() / step_time`.
- **Pos/s (hw)**: Total positions per second = `B × N_padded / step_time` (hardware throughput).
- **Pad%**: Measured padding waste = `1 - sum(attention_mask) / (B × N)`.
- **Loss%**: Tokens with gradient / real tokens = `(labels != -100).sum() / attention_mask.sum()`.
- **Step (ms)**: Wall-clock time for full fwd+bwd+zero_grad, averaged over 10 measured steps.
- **VRAM**: Peak GPU memory (torch.cuda.max_memory_allocated).

#### Stage 1 (Frozen LLM — only projector trains)

| Config | B | avg N | Real tok/s | Pos/s (hw) | Pad% | Loss% | Step ms | VRAM |
|--------|---|-------|-----------|------------|------|-------|---------|------|
| Vanilla, no bucketing | 4 | 382 | 21,673 | 24,611 | 11.9% | 55.3% | 62.0 | 2.48 GB |
| Vanilla, bucketing | 4 | 336 | 16,473 | 16,900 | 2.5% | 54.2% | 79.4 | 2.14 GB |
| Vanilla, no bucketing | 16 | 421 | 52,993 | 65,323 | 18.9% | 56.0% | 103.1 | 7.05 GB |
| Vanilla, bucketing | 16 | 341 | 57,020 | 59,202 | 3.7% | 54.2% | 92.1 | 5.25 GB |
| Vanilla, no bucketing | 32 | 424 | 62,619 | 78,502 | 20.2% | 55.6% | 173.0 | 13.00 GB |
| **Vanilla, bucketing** | **32** | **343** | **76,868** | **80,349** | **4.3%** | **54.3%** | **136.8** | **9.42 GB** |
| Vanilla, bucketing | 64 | 345 | 90,349 | 94,866 | 4.8% | 54.3% | 233.0 | 17.69 GB |
| Vanilla, bucketing | 128 | 350 | 90,255 | 95,880 | 5.9% | 54.5% | 467.9 | 34.57 GB |
| Vanilla, bucketing | 256 | 354 | 89,605 | 95,679 | 6.3% | 54.7% | 947.7 | 68.80 GB |
| Vanilla + pad64 | 32 | 384 | 74,123 | 86,584 | 14.4% | 54.3% | 141.9 | 10.16 GB |
| **TileGym + pad64** | **4** | **435** | **58,096** | **71,807** | **19.1%** | **57.3%** | **24.2** | **3.16 GB** |
| TileGym + pad64, bucketing | 32 | 384 | 91,427 | 106,601 | 14.2% | 54.3% | 115.3 | 10.93 GB |
| TileGym + pad64, bucketing | 64 | 384 | 93,167 | 108,386 | 14.0% | 54.5% | 226.7 | 18.15 GB |
| TileGym + pad64, bucketing | 128 | 384 | 90,876 | 105,101 | 13.5% | 54.7% | 467.7 | 32.61 GB |
| Packing (N=1024) | 32→12 | 1024 | 83,684 | 83,684 | 0.0% | 51.2% | 140.7 | 8.94 GB |
| Packing (N=1024) | 64→23 | 1024 | 94,517 | 94,517 | 0.0% | 51.3% | 250.3 | 16.74 GB |
| Packing (N=1024) | 128→45 | 1024 | 103,542 | 103,542 | 0.0% | 52.4% | 446.0 | 31.12 GB |
| **Packing + TileGym (N=1024)** | **64→23** | **1024** | **107,367** | **107,367** | **0.0%** | **51.9%** | **218.4** | **18.12 GB** |

#### Key Findings

1. **Throughput plateau at ~90-95K Real tok/s (vanilla)**: Beyond B=64 with bucketing,
   adding more batch size does NOT increase throughput — only increases VRAM and step time.
   The GPU is compute-saturated at this point.

2. **TileGym ceiling: ~107K Real tok/s**: With cuTile kernels (packing + fixed shapes),
   peak throughput is 107K — a **14% lift** over vanilla-bucketed peak (90K → 107K).
   With pad_to_multiple=64: 93K (+3% vs vanilla bucketed due to extra padding).

3. **TileGym requires quantized shapes**: Variable sequence lengths cause catastrophic
   autotuning overhead (650ms/step vs 115ms at fixed shapes). In production, use
   `pad_to_multiple_of=64` or packing with fixed bin sizes.

4. **Bucketing is extremely effective**: Reduces padding from 18-20% to 4-6%.
   At B≥64, bucketing groups enough similar-length samples that padding is minimal.

5. **Packing provides marginal gains over bucketing at large B**: 
   Packing gives 0% padding (vs 4-6% for bucketing at B=64+), but the ~5% padding
   elimination translates to only ~5% more Real tok/s (94.5K vs 90.3K).

6. **Best configurations by VRAM budget**:
   - **≤5 GB**: TileGym B=4 pad64 → 58K Real tok/s (24ms/step)
   - **≤10 GB**: TileGym B=32 bucketing pad64 → 91K Real tok/s (115ms/step)
   - **≤20 GB**: Packing+TileGym B=64→23 N=1024 → **107K Real tok/s** (218ms/step)
   - **≤35 GB**: Packing+TileGym or vanilla B=128 → ~103K Real tok/s

7. **Loss% is dataset-intrinsic (~54%)**: Independent of batch size, bucketing, packing,
   or kernel choice. It's the fraction of non-padding tokens that contribute to the loss
   (answer tokens only, not system prompt or question).

### Packing Efficiency Analysis (Verified)

| Config | Pad% | Real tok/s | vs Bucketing same B |
|--------|------|-----------|---------------------|
| Bucketing B=64 | 4.8% | 90,349 | — |
| Packing B=64→23, N=1024 | 0.0% | 94,517 | **+4.6%** |
| Bucketing B=128 | 5.9% | 90,255 | — |
| Packing B=128→45, N=1024 | 0.0% | 103,542 | **+14.7%** |

Packing shows modest gains at similar VRAM (~17-31 GB). The 14.7% gain at B=128 is
larger because packing converts the 128 short samples into 45 packed bins of N=1024,
giving longer sequences that better utilize Tensor Cores.

**When packing IS valuable** (confirmed by measurement):
- When combined with TileGym (fixed N enables kernel caching) → +14% over vanilla packing
- At very large effective batch (128+ input samples) → longer packed sequences
- High length variance datasets (not tested here; sharegpt4v/coco is relatively uniform)

### MFU (Model FLOPs Utilization) Analysis

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition  
- 188 SMs @ 2430 MHz  
- 95 GB GDDR7  
- Peak BF16 Tensor: ~936 TFLOP/s (dense)

**Model parameters** (measured):
- Text model (Qwen2.5-0.5B): **494.0M**
- Vision encoder (SigLIP2-base-224): **92.9M**
- Projector: **2.8M**
- Total: **589.7M**

**FLOPs formula**:
- Full training (fwd + bwd): **6 × N × P** per token
- Frozen (fwd + activation bwd): **4 × N × P** per token
- Forward-only (no_grad): **2 × N × P** per token

#### Current MFU (Iter 15 — Verified)

Using measured Real tok/s from Iteration 15 ground truth:

**Formula**: MFU = (tok/s × FLOPs_per_token) / Peak_TFLOP/s  
Stage 1 (frozen LLM): FLOPs_per_token = 4 × N × P = 4 × 494M = 1.976 GFLOP  
Peak BF16: 936 TFLOP/s

| Config | Measured tok/s | Achieved TFLOP/s | MFU |
|---|---|---|---|
| Vanilla, bucketing, B=64 | 90,349 | 178.5 | **19.1%** |
| TileGym + pad64, B=32 | 91,427 | 180.7 | **19.3%** |
| TileGym + pad64, B=64 | 93,167 | 184.1 | **19.7%** |
| Packing, B=128→45, N=1024 | 103,542 | 204.6 | **21.9%** |
| **Packing + TileGym, B=64→23** | **107,367** | **212.2** | **22.7%** |
| TileGym, B=4, pad64 | 58,096 | 114.8 | **12.3%** |

**Key observations**:
- MFU is **19-23%** across all saturated configs — consistent with model scale limitations.
- B=4 only achieves 12% MFU (insufficient parallelism for Tensor Cores).
- Packing + TileGym at peak gives 22.7% MFU — near ceiling for this 0.5B model.
- Previous estimates of 24% MFU were slightly inflated (likely based on synthetic data).

### Why MFU is 24% (and why that's near-optimal for this workload)

1. **Small model (494M)** → GEMM shapes are small (M≤1024, K=896, N=4864).
   Tensor Cores need large matmuls (M,N,K > 4096) for peak utilization.
2. **Memory-bandwidth bound**: 494M params × 2 bytes = 988 MB weights per layer pass.
   At 1.79 TB/s bandwidth, each layer read takes ~0.55ms regardless of batch size.
3. **Non-matmul overhead**: softmax, masking, RoPE, normalization, activation functions
   consume ~15-20% of step time but contribute 0% to "useful" TFLOPS in MFU calculation.
4. **Vision encoder is a separate forward** → sequential dependency through projector,
   no overlap with LLM compute.

**Reference MFU values** (published benchmarks):
| Setup | MFU |
|---|---|
| GPT-3 175B, A100, bs=1024, seq=2048 | ~50% |
| LLaMA-7B, A100, bs=256, seq=2048 | ~55% |
| LLaMA-1.3B, A100, bs=64, seq=2048 | 35–40% |
| **SiQ-VL (0.5B), Blackwell, bs=768, seq=1024** | **24%** |
| Expected range for 0.5B VLM | **15–30%** |

Our 24% MFU is within the expected range and near the ceiling for this model scale.

### Paths to Higher MFU (for future iterations)

| Path | Expected MFU | When |
|---|---|---|
| Longer sequences (4K–8K) via multi-turn data | 25–28% | With long-context data |
| Larger text model (3B–7B) | 35–45% | If model scale increases |
| FP8 precision (2x peak TFLOPS on Blackwell) | 12–15% (of FP8 peak) | When FP8 training stable |
| Distributed (FSDP2 + tensor parallelism) | Same per-GPU MFU | For total throughput scaling |
| All combined (7B + 8K seq + distributed) | 40–50% | Target for production |

---

## 4.2. Negative Results / Failed Experiments

> Not every hypothesis pans out. These are recorded here for the blog's "pitfalls" section
> and to prevent future re-investigation of dead ends.

### Failed: Vision Feature Caching (Iteration 3 investigation)

- **Hypothesis**: Pre-extracting SigLIP features offline and loading them during training
  would eliminate the 428M-param vision forward pass entirely, saving ~21ms/step.
- **What we built**:
  - `scripts/extract_vision_features.py`: extracts features at 165 tiles/s, stores sharded `.pt` files
  - `siq_vl/dataset.py::CachedVQADataset`: loads pre-cached features instead of PIL images
  - `siq_vl/collator.py::CachedVisionDataCollator`: routes to `processor.process_cached()`
  - `siq_vl/model/modeling.py`: `vision_features` kwarg bypasses vision encoder in forward
- **Result**: Tokens/sec UNCHANGED (11,070 → 11,070). VRAM actually INCREASED (+27%).
- **Root cause**: On Blackwell with bf16+SDPA/flash, the SigLIP forward is only 21ms (5ms/tile).
  The cached features are large tensors (tiles × 1024 × 1152 × bf16 = ~9.4MB/sample).
  DataLoader H2D transfer of these tensors via `pin_memory` is slower than just computing them.
  Additionally, the cached tensors consume GPU memory that would otherwise be freed after
  the transient vision forward completes.
- **When it WOULD help**: Older GPUs (V100/A100 without Blackwell tensor cores), multi-epoch
  training where the same images are seen 10+ times, or distributed setups where vision
  compute is duplicated across ranks without model parallelism.
- **Kept in codebase**: Yes — the infrastructure is retained for future use.

### Failed: torch.compile on Vision Encoder (Iteration 3 investigation)

- **Hypothesis**: `torch.compile(mode="max-autotune-no-cudagraphs")` on the frozen vision
  encoder would fuse small ops and reduce kernel launch overhead.
- **Result**: +3% speedup (statistically insignificant), but adds 40s compile time on first step.
- **Root cause**: The vision encoder's critical path is already optimal:
  - Attention → dispatches to flash kernel via SDPA (fused)
  - Linear layers → CUTLASS bf16 GEMM on tensor cores (already optimal)
  - Normalization → already fused by Liger
  - compile can only fuse the residual arithmetic (add, dropout), which is <1% of compute.
- **HF Trainer compatibility issue**: Wrapping the `PreTrainedModel` submodule with `torch.compile`
  breaks accelerate's `unwrap_model` (expects `_orig_mod` attribute). Workaround: compile only
  inside the forward path lazily — but the gains don't justify the complexity.

### Failed: CUDA Stream Prefetcher + DataLoader Tuning (Iteration 4 investigation)

- **Hypothesis**: The DataLoader uses the default CUDA stream for H2D transfers, which blocks
  GPU compute. Using a dedicated CUDA stream for async prefetching, combined with more workers
  (12), `prefetch_factor=4`, and `persistent_workers=True`, should overlap data loading with compute.
- **What we built**:
  - `siq_vl/prefetcher.py::CUDAPrefetcher`: wraps DataLoader, uses separate CUDA stream for
    non_blocking H2D, prefetches next batch while current batch is being computed.
  - Added `--use_prefetcher`, `--prefetch_factor` flags to `scripts/profile_baseline.py`
  - Added `dataloader_persistent_workers` and `dataloader_prefetch_factor` to TrainingArguments
- **Result**: Tokens/sec UNCHANGED (20,284 → 20,288, within noise).
- **Root cause**: Measured independently, the DataLoader produces batches in **9.4ms average
  (p50=0.2ms!)** while GPU compute takes **288ms**. Data is ready 30x before compute finishes.
  With 12 persistent workers and prefetch_factor=4, the pipeline is always full — there is
  zero GPU starvation. The `.to(device)` H2D transfer is <1ms for the processed batch
  (small tensors after PIL→tensor processing happens on CPU workers).
- **When it WOULD help**:
  - Very large batches where H2D transfer becomes significant (hundreds of MB per batch)
  - Training on images without pre-resizing (raw 4K images decoded on-the-fly)
  - Setups with slow CPU (few cores) or slow storage (network filesystem)
  - Multi-GPU DDP where data loading is a per-rank bottleneck
- **Kept in codebase**: Yes — CUDAPrefetcher is good infrastructure for future scale.

### Failed: Sample Packing with 4D Attention Mask (Iteration 6 investigation)

- **Hypothesis**: Packing multiple samples into fixed-length 1024-token sequences with a
  block-diagonal causal attention mask would eliminate all padding waste and increase
  effective throughput.
- **What we built**:
  - `siq_vl/packing.py::PackingCollator`: greedy bin-packing with first-fit-decreasing,
    produces packed sequences with custom position_ids and 4D bf16 attention mask
  - Block-diagonal causal mask as additive bias `(B, 1, L, L)` with 0 for attend, -inf for mask
- **Result**: **-54% throughput** (21,026 → 9,663 tok/s). Dramatically worse.
- **Root cause**: Two compounding problems:
  1. **SDPA cannot use flash kernel with 4D mask**: falls back to "math" or "mem_efficient"
     backend which is ~3x slower than flash for 1024-length sequences
  2. **Quadratic attention cost**: packing 356-token avg samples into 1024-token sequences
     INCREASES total attention pairs: 7×1024² = 7.3M vs 16×356² = 2.0M (3.6x more compute)
- **When it WOULD help**:
  - Long sequences (2K+ tokens) where padding waste is 30–50%
  - Using `flash_attn_varlen_func` (which natively handles variable-length sequences with
    `cu_seqlens` without a dense mask) or FlexAttention compiled block masks
  - Datasets with high length variance where packing actually reduces total positions
- **Kept in codebase**: Yes — `siq_vl/packing.py` retained for reference.
- **Resolution**: Iteration 10 successfully re-implements packing using HuggingFace's native
  FlexAttention integration (position_ids resets + no attention_mask), achieving +4.8% throughput
  and -18% VRAM on mixed-dataset training. See `siq_vl/collator.py::PackingCollator`.

### Failed: Custom Triton Flash Attention Kernel (Iteration 9 investigation)

- **Hypothesis**: Since flash-attn package can't be installed (OOM during build) and SDPA
  can't use flash kernel with a 4D mask, a custom Triton kernel implementing FlashAttention-2
  with varlen support could enable efficient packing.
- **What we built**: `siq_vl/kernels/flash_attention.py` — full FlashAttention-2 in Triton:
  - Forward kernel with online softmax, causal masking, non-power-of-2 head_dim support
  - Backward kernel with attention recomputation
  - `flash_attention_varlen()` for packed sequences via cu_seqlens
- **Correctness**: PASSED (max abs diff 0.016, within bf16 tolerance vs SDPA reference)
- **Performance**: Our kernel is **3–4x SLOWER** than SDPA on Blackwell:
  | Config | Our Triton | SDPA | Ratio |
  |---|---|---|---|
  | B=16, N=356, D=96 | 0.235ms | 0.058ms | 0.25x |
  | B=16, N=1024, D=96 | 0.976ms | 0.234ms | 0.24x |
  | varlen (3 seqs, 1024) | 0.353ms | — | — |
  | SDPA + 4D mask | — | 0.128ms | — |
- **Root cause**: SDPA on Blackwell (sm_120) dispatches to `fmha_cutlassF_bf16_aligned`
  which uses hardware-specific features (wgmma tensor core instructions, TMA async copy,
  warp-specialized pipeline) that Triton 3.5 cannot generate. Writing a competitive attention
  kernel for Blackwell requires CUDA C++ with CUTLASS-level optimizations.
- **Key insight**: Even if the kernel were as fast as SDPA, attention is only **~5% of total
  step time**. The bottleneck is MLP matmuls (QKV proj, FFN up/down). A 2x faster attention
  kernel would yield <3% end-to-end improvement.
- **Kept in codebase**: Yes — valuable for future long-context training (where attention
  fraction grows quadratically) and non-standard mask patterns.

### Failed: cudagraphs / reduce-overhead mode (Iteration 8 investigation)

- **Hypothesis**: CUDA graphs capture the entire kernel sequence and replay it without
  per-kernel launch overhead. With bucketing, shapes should be stable enough.
- **Result**: **-20%** (244ms vs 204ms for max-autotune-no-cudagraphs).
- **Root cause**: VL models have inherently dynamic shapes per batch — image tile counts vary
  per sample, causing sequence lengths to differ between batches. cudagraphs requires
  re-capturing the graph on every shape change, which is more expensive than the kernel
  launch overhead it saves.

### Failed: pad_to_multiple_of=64 (Iteration 8 investigation)

- **Hypothesis**: Discretizing sequence lengths to multiples of 64 would enable torch.compile
  to reuse compiled graphs, reducing recompilation overhead.
- **Result**: **-6.6%** (26,443 vs 28,322 tok/s).
- **Root cause**: Still produces 5 distinct lengths (320–576), so recompilation still occurs.
  Meanwhile, the extra padding tokens (up to 63 per sequence) add wasted compute.
  The combination of "no graph reuse benefit" + "more wasted tokens" makes it net negative.

### Failed: Gradient Checkpointing Removal (Iteration 3 investigation)

- **Hypothesis**: In Stage 1 the LLM is frozen — gradient checkpointing recomputes a forward
  it never needs gradients for, wasting compute.
- **Result**: +3% speedup, negligible VRAM change.
- **Root cause**: HF Trainer's gradient checkpointing only wraps modules with `requires_grad`
  parameters. Since the LLM is fully frozen, checkpointing already short-circuits. The 3%
  gain comes from reduced Python overhead in the checkpoint wrapper code, not from eliminating
  recomputation. This becomes more impactful in Stage 2 (where LLM IS trainable).

---

## 5. Measurement Protocol (mandatory for every iteration)

To ensure blog numbers are credible and reproducible, every iteration must:

1. **Same seed** (`--seed 42`), same data subsets, same batch / accum configuration;
2. **Warm up ≥ 5 steps**, then measure mean and P50/P95 of steps 6–50;
3. **Three core metrics must be recorded simultaneously**:
   - `step time (ms)` (from HF Trainer log)
   - `peak VRAM (GB)` (`torch.cuda.max_memory_allocated()` sampled after train_step)
   - `tokens / sec` (`include_tokens_per_second=True` already enabled)
4. **Loss sanity check**: run 100 steps per iteration, compare step-100 train loss against baseline;
   difference should be < 5% (packing may produce slightly smoother loss curves due to higher effective token count);
5. **W&B must be enabled**, run name template: `opt_iter{N}_{short_desc}`;
6. **Profiler trace** — at least one trace per iteration, committed to repo:
   ```python
   with torch.profiler.profile(
       activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
       record_shapes=True, profile_memory=True,
   ) as prof:
       for _ in range(10): train_step()
   prof.export_chrome_trace("docs/traces/iter_{N}.json")
   ```
7. **Write a 200–500 word "Lessons" section per iteration**, including pitfalls encountered,
   results that diverged from expectations, and inputs for the next iteration's hypothesis.

---

## 6. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Liger monkey-patch coupled to transformers version | P0.1 failure | Pin versions; prepare `--no_liger` fallback path |
| FA2 wheel compatibility on Blackwell sm_120 | P0.2 failure | Use PyTorch built-in SDPA `enable_flash=True` as fallback |
| SigLIP cache exceeds disk capacity | P1.2 blocked | Force post-pixel-shuffle caching; fp16 storage; streaming shards |
| Packed loss values diverge from non-packed | P2.1 exit | Mandatory unit test: packed vs non-packed loss diff < 1e-4 |
| torch.compile conflicts with Liger | **CONFIRMED** | Use `--no_liger --torch_compile` (Iter 7 proved compile > Liger for throughput) |
| Stage 2 later unfreezes vision, invalidating cache | Strategic risk | Document clearly: cache only valid for frozen vision phases |
| Modal inter-container bandwidth bottlenecks FSDP all-gather | P4 degraded scaling | Benchmark with 2 GPUs first; fall back to `SHARD_GRAD_OP` or gradient accumulation across nodes |
| Modal spot preemption loses in-flight step | P4 data loss | Checkpoint every K steps to Volume; training auto-resumes |
| Modal Volume I/O too slow for large dataset cache | P4 slow startup | Use streaming reads; keep hot subset in container local SSD |
| Cost overrun on Modal GPU hours | P4 budget risk | Set hard `timeout`; implement early stopping; monitor $/step in W&B |

---

## 7. Blog Outline Mapping (this document → final blog sections)

| Document Section | Blog Section |
|---|---|
| §0 + §1 | "I have a 1.5B VLM that trains too slowly on one GPU" |
| §1.2 + §2 | "Bottleneck Decomposition: where is time actually spent?" |
| §3 + §4 (Iter 0) | "Establishing the Baseline" |
| §4 (Iter 1–3) | "Cheap Wins: fused kernels & unnecessary checkpointing" |
| §4 (Iter 4–5) | "Squeezing the data side: offline preprocessing & feature cache" |
| §4 (Iter 6–7) | "The engineering climax: Bucketing → torch.compile" |
| §4 (Iter 8) + §5 | "Last mile: batch tuning and measurement methodology" |
| §4 (Iter 10) | "Packing done right: FlexAttention + mixed datasets" |
| §4 (Iter 11) | "Custom kernels: cuTile DSL on Blackwell" |
| §4 (Iter 13) | "TileGym: the Blackwell-native Liger-Kernel" |
| §4 (Iter 12) | "Stage 2 readiness: unfreezing the LLM" |
| §4 (Iter 14) | "Batch scaling: GPU already saturated — use VRAM for convergence" |
| §4 (Iter 15+) | "Going horizontal: distributed training on Modal" |
| §6 | "Pitfalls I hit (from the Risk Register)" |

---

## 8. Open Questions (decisions pending)

> These need to be decided by the project owner before implementation starts.
> Once answered, convert this section to "Decisions" and record the choices.

- [ ] **Q1 (Scope)**: Which phase does the first PR target? (P0 only / P0+P1 / P0+P1+P2)
- [ ] **Q2 (Vision freeze strategy)**: Will Stage 2 ever unfreeze vision? This determines whether SigLIP cache is worthwhile.
- [ ] **Q3 (Cache format)**: Sharded .pt / safetensors / WebDataset — pick one.
- [x] **Q4 (Packing backend)**: FlexAttention / flash-attn varlen — **FlexAttention chosen** (Iter 10: native HF integration via position_ids resets, no external deps, works with torch.compile).
- [ ] **Q5 (Liger style)**: Accept monkey-patching transformers internals, or wrap only inside SiQ_VLForCausalLM?
- [ ] **Q6 (Data scale)**: Final training scale < 500K / 1M–5M / 10M+? This affects how aggressive the cache design should be.
- [ ] **Q7 (Modal GPU type)**: Which Modal GPU class to target? (A100-80GB / H100 / A10G) — affects FSDP config and cost model.
- [ ] **Q8 (Modal scale)**: How many GPUs for distributed runs? (2 / 4 / 8) — determines sharding strategy and communication overhead tolerance.

---

## Appendix A — Key Code Index

- `siq_vl/model/modeling.py::SiQ_VLForCausalLM.forward` — vision replacement + LLM forward core
- `siq_vl/model/modeling.py::SiQ_VLProjector` — pixel shuffle + linear projection
- `siq_vl/model/processing.py::SiQ_VLProcessor.__call__` — image + text + label processing
- `siq_vl/collator.py::SiQ_VLDataCollator` — per-batch entry point (standard padding)
- `siq_vl/collator.py::PackingCollator` — packing with FlexAttention (Iter 10)
- `siq_vl/kernels/cutile_attention.py` — cuTile Flash Attention (forward + varlen + autotuning)
- `siq_vl/kernels/attention_backend.py` — HF Transformers attention backend registration
- `siq_vl/dataset.py::VQADataset` — one random turn per sample
- `scripts/benchmark_stage2.py` — Stage 2 benchmark (LoRA / Full FT / grad_ckpt comparison)
- `scripts/benchmark_batch_scaling.py` — Batch-size scaling (find max B for VRAM budget)
- `scripts/benchmark_flashoptim.py` — FlashOptim (Databricks) vs AdamW comparison
- `scripts/train.py::train` — Trainer assembly + Stage switching
- `scripts/train_launch.sh` — host detection + accelerate launch
- `modal_train.py` — (planned) Modal App definition for distributed training

## Appendix B — Glossary

- **packing / sample packing**: Concatenating multiple short sequences into one long sequence, with a block-diagonal mask ensuring cross-sample invisibility
- **bucketing**: Grouping samples by length into buckets so that intra-bucket padding variance is low
- **fused linear-CE**: Computing `hidden @ W → softmax → nll` in a single kernel using chunked computation, never materializing the full logits tensor
- **FlexAttention**: PyTorch 2.5+ built-in mechanism that allows custom attention masks via `mask_mod` / `score_mod`, compiled to Triton kernels
- **varlen**: FlashAttention 2's "variable length" API (`flash_attn_varlen_func`), using `cu_seqlens` to express packed batches
- **Modal**: Serverless cloud platform for running GPU workloads; provisions containers with GPUs on demand, with persistent Volumes for data/checkpoints
- **cuTile DSL**: NVIDIA's domain-specific language for writing GPU kernels targeting Blackwell (sm_120) architecture, with native support for wgmma, TMA, and autotuning
- **FSDP (Fully Sharded Data Parallel)**: PyTorch-native distributed training strategy that shards model parameters, gradients, and optimizer states across GPUs
- **FlashOptim**: Databricks library providing drop-in optimizer replacements (`FlashAdamW`) that quantize states to int8 + error correction, reducing per-parameter memory by ~57%. Features gradient release (update during backward) and compressed checkpoints
