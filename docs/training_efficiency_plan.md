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

### Iteration 1 — P0.1: Liger Fused LMHead-CE / RMSNorm / SwiGLU (planned)

- **Date**: TBD
- **Hypothesis**: Qwen2.5-1.5B has vocab=151,936. The `hidden @ lm_head.T` operation produces
  the largest intermediate tensor in the entire forward chain. Liger-Kernel's
  `LigerFusedLinearCrossEntropyLoss` computes logits → log_softmax → nll in chunks
  without ever materializing the full logits tensor, saving both memory and compute.
- **Change**:
  - Add dependency: `liger-kernel` (in `pyproject.toml`)
  - `siq_vl/model/modeling.py`: after `SiQ_VLForCausalLM.__init__`, call
    `apply_liger_kernel_to_qwen2(rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=True)`
  - `forward` no longer relies on HF Qwen2 default loss computation; instead passes `labels=None` to get `hidden_states`, then calls fused linear-CE
  - Verification: loss values align with baseline to within 1e-3 (run 10 steps with same seed)
- **How to Reproduce**:
  ```bash
  STAGE=1 bash scripts/train_launch.sh \
    --max_steps 50 --logging_steps 1 --max_samples 2000
  ```
- **Expected Result**:
  - step time ↓ 10–25%
  - peak VRAM ↓ 20–30%
  - eval loss aligns with baseline
- **Decision**: —

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

### Iteration 3 — P0.3: Disable Stage 1 Checkpointing + Vision no_grad (planned)

- **Date**: TBD
- **Hypothesis**:
  - Stage 1 LLM is frozen, so gradient checkpointing has no gradients to recompute — pure waste
  - Even with `param.requires_grad=False`, `vision_model.forward` still builds the autograd
    graph and retains activations; wrapping it with `torch.no_grad()` saves both time and memory
- **Change**:
  - `siq_vl/model/modeling.py::SiQ_VLForCausalLM.forward`:
    Wrap `self.vision_model(pixel_values)` in `torch.no_grad()` when vision is frozen
  - `scripts/train_launch.sh` or `scripts/train.py`: set `--gradient_checkpointing False` for Stage 1
    (or decide dynamically in `train.py` based on stage)
- **Expected Result**: Stage 1 step time ↓ 10–20%, VRAM ↓ 5–10%
- **Decision**: —

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

### Iteration 6 — P1.3: Length Bucketing (planned)

- **Date**: TBD
- **Hypothesis**: Intra-batch padding wastes 30–50%. HF Trainer's built-in
  `group_by_length=True` groups samples by length during sampling, significantly reducing
  length variance within each batch.
- **Change**:
  - Modify `VQADataset` to add a `length` field (estimated via tokenizer = `len(question) + len(answer) + tokens_per_tile × num_tiles + template overhead`)
  - Add `group_by_length=True`, `length_column_name="length"` to `TrainingArguments`
- **Expected Result**: step time ↓ 15–25%
- **Decision**: —

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

### Iteration 8 — P3: torch.compile LLM (planned)

- **Date**: TBD
- **Hypothesis**: After P0+P1+P2, applying `torch.compile(mode="reduce-overhead")`
  to the LLM forward should yield an additional 5–15% speedup.
- **Risk**: Graph capture compatibility with Liger / FlexAttention; dynamic shape recompilation overhead.
  → After packing, shapes are fixed (consistent max_length), reducing recompilation pressure.
- **Decision**: —

---

### Iteration 9 — P4: Modal Distributed Training (planned)

- **Date**: TBD
- **Hypothesis**: Once single-GPU efficiency is maximized (P0–P3), the next multiplier is
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
| torch.compile conflicts with Liger | P3 failure | P3 is optional; park if incompatible |
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
| §4 (Iter 6–7) | "The engineering climax: Bucketing → Packing → FlexAttention" |
| §4 (Iter 8) + §5 | "Last mile: torch.compile, and measurement methodology" |
| §4 (Iter 9) | "Going horizontal: distributed training on Modal" |
| §6 | "Pitfalls I hit (from the Risk Register)" |

---

## 8. Open Questions (decisions pending)

> These need to be decided by the project owner before implementation starts.
> Once answered, convert this section to "Decisions" and record the choices.

- [ ] **Q1 (Scope)**: Which phase does the first PR target? (P0 only / P0+P1 / P0+P1+P2)
- [ ] **Q2 (Vision freeze strategy)**: Will Stage 2 ever unfreeze vision? This determines whether SigLIP cache is worthwhile.
- [ ] **Q3 (Cache format)**: Sharded .pt / safetensors / WebDataset — pick one.
- [ ] **Q4 (Packing backend)**: FlexAttention / flash-attn varlen — pick one.
- [ ] **Q5 (Liger style)**: Accept monkey-patching transformers internals, or wrap only inside SiQ_VLForCausalLM?
- [ ] **Q6 (Data scale)**: Final training scale < 500K / 1M–5M / 10M+? This affects how aggressive the cache design should be.
- [ ] **Q7 (Modal GPU type)**: Which Modal GPU class to target? (A100-80GB / H100 / A10G) — affects FSDP config and cost model.
- [ ] **Q8 (Modal scale)**: How many GPUs for distributed runs? (2 / 4 / 8) — determines sharding strategy and communication overhead tolerance.

---

## Appendix A — Key Code Index

- `siq_vl/model/modeling.py::SiQ_VLForCausalLM.forward` — vision replacement + LLM forward core
- `siq_vl/model/modeling.py::SiQ_VLProjector` — pixel shuffle + linear projection
- `siq_vl/model/processing.py::SiQ_VLProcessor.__call__` — image + text + label processing
- `siq_vl/collator.py::SiQ_VLDataCollator` — per-batch entry point
- `siq_vl/dataset.py::VQADataset` — one random turn per sample
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
- **FSDP (Fully Sharded Data Parallel)**: PyTorch-native distributed training strategy that shards model parameters, gradients, and optimizer states across GPUs
