"""Batch-size scaling benchmark for SiQ-VL Stage 1 & Stage 2.

Measures throughput and VRAM across a range of batch sizes to find the optimal
per_device_train_batch_size for a given GPU memory budget.

Key findings (RTX PRO 6000 Blackwell, 95 GB):
- Stage 1 (frozen LLM): throughput saturates at B=16 (~110K tok/s), OOM at B=192.
- Stage 2 (unfrozen + grad_ckpt + AdamW): throughput flat ~74K tok/s,
  B=768 fills 80% VRAM at N=1024. Increasing B does NOT improve speed —
  GPU compute is fully saturated at B=32.
- Same total tokens (BxN) uses the same VRAM regardless of B/N split.
- Longer sequences (N=4096) are only ~5% slower than N=1024.

Usage:
    python scripts/benchmark_batch_scaling.py --stage 2 --seq_len 1024
    python scripts/benchmark_batch_scaling.py --stage 1 --batch_sizes 4 16 32 64 128
"""

import argparse
import time

import torch

from tilegym.transformers import apply_tilegym_kernel_to_qwen2

from siq_vl.kernels.fused_linear_ce import patch_qwen2_fused_linear_ce
from siq_vl.model.configuration import get_siq_vl_config
from siq_vl.model.modeling import SiQ_VLForCausalLM, SiQ_VLTextModel, SiQ_VLVisionModel


def build_model(stage: int):
    apply_tilegym_kernel_to_qwen2(use_cutile=True)
    patch_qwen2_fused_linear_ce()

    config = get_siq_vl_config(
        text_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        vision_model_name_or_path="google/siglip2-base-patch16-224",
    )
    model = SiQ_VLForCausalLM(config)
    model.text_model = SiQ_VLTextModel.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.vision_model = SiQ_VLVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.projector = model.projector.to(torch.bfloat16)
    model.freez_vision_model()

    if stage == 1:
        model.freez_text_model()
    # Stage 2: text model stays unfrozen

    model = model.to("cuda").train()

    if stage == 2:
        model.text_model.gradient_checkpointing_enable()

    return model


def make_batch(model, B: int, N: int):
    tokens_per_image = (
        model.vision_model.config.image_size
        // model.vision_model.config.patch_size
        // model.projector.vision_pixel_shuffle_factor
    ) ** 2
    img_tok = model.config.image_token_index
    safe_range = img_tok  # sample from [0, img_tok) to avoid collision
    input_ids = torch.randint(0, safe_range, (B, N), device="cuda")
    input_ids[:, :tokens_per_image] = img_tok
    pixel_values = torch.randn(B, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    labels = input_ids.clone()
    labels[:, :tokens_per_image] = -100
    attention_mask = torch.ones(B, N, device="cuda", dtype=torch.long)
    return dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
    )


def benchmark_config(model, optimizer, B: int, N: int, steps: int = 8, warmup: int = 2):
    batch = make_batch(model, B, N)

    for _ in range(warmup):
        out = model(**batch)
        out.loss.backward()
        if optimizer:
            optimizer.step()
            optimizer.zero_grad()
        else:
            model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for _ in range(steps):
        out = model(**batch)
        out.loss.backward()
        if optimizer:
            optimizer.step()
            optimizer.zero_grad()
        else:
            model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / steps * 1000
    toks = B * N / (elapsed / steps)
    mem = torch.cuda.max_memory_allocated() / 1024**3

    del batch
    torch.cuda.empty_cache()
    return ms, toks, mem


def main():
    parser = argparse.ArgumentParser(description="Batch-size scaling benchmark")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--use_optimizer", action="store_true", default=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.batch_sizes is None:
        if args.stage == 1:
            args.batch_sizes = [4, 16, 32, 64, 128, 192, 256]
        else:
            args.batch_sizes = [32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896]

    model = build_model(args.stage)
    optimizer = None
    if args.use_optimizer:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name()} | {total_gb:.0f} GB total")
    print(f"Stage: {args.stage} | seq_len: {args.seq_len} | optimizer: {args.use_optimizer}")
    print(f"Config: TileGym (RoPE+RMSNorm+SwiGLU+FA4) + fused_linear_ce (grad-in-forward)")
    if args.stage == 2:
        print("  text_model: unfrozen + gradient_checkpointing")
    print()
    print(f"{'B':<6}{'N':<6}{'ms/step':<11}{'tok/s':<12}{'VRAM':<10}{'Util%':<8}{'BxN':<10}")
    print("-" * 63)

    for B in args.batch_sizes:
        try:
            ms, toks, mem = benchmark_config(
                model, optimizer, B, args.seq_len, steps=args.steps, warmup=args.warmup
            )
            util = mem / total_gb * 100
            print(f"{B:<6}{args.seq_len:<6}{ms:<11.1f}{toks:<12.0f}{mem:<10.2f}{util:<8.0f}{B*args.seq_len:<10}")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "memory" in str(e).lower():
                print(f"{B:<6}{args.seq_len:<6}OOM")
                torch.cuda.empty_cache()
                break
            raise

    print()
    print("Summary: throughput is B-independent once compute is saturated.")
    print("Use max B that fits in ~80% VRAM for best convergence per wall-clock hour.")


if __name__ == "__main__":
    main()
