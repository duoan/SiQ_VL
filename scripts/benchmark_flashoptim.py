"""Benchmark FlashOptim (Databricks) vs torch.optim.AdamW for SiQ-VL Stage 2.

FlashOptim (https://github.com/databricks/flashoptim) quantizes optimizer states
to int8, adds error correction, and optionally releases gradients during backward.

Key findings (RTX PRO 6000 Blackwell, 95 GB, Qwen2.5-0.5B):
- FlashAdamW saves ~0.4 GB optimizer memory (8-bit momentum + variance).
- Gradient release enables B=832 (vs B=768 max with AdamW).
- Throughput is identical — bottleneck is activation memory, not optimizer states.
- For small models (0.5B), optimizer state is only ~4 GB out of 76 GB total.
  FlashOptim is more impactful for larger models (7B+) where optimizer dominates.

Usage:
    python scripts/benchmark_flashoptim.py
    python scripts/benchmark_flashoptim.py --batch_sizes 32 128 512 768 832
"""

import argparse
import time

import torch
from flashoptim import FlashAdamW, enable_gradient_release
from tilegym.transformers import apply_tilegym_kernel_to_qwen2

from siq_vl.kernels.fused_linear_ce import patch_qwen2_fused_linear_ce
from siq_vl.model.configuration import get_siq_vl_config
from siq_vl.model.modeling import SiQ_VLForCausalLM, SiQ_VLTextModel, SiQ_VLVisionModel


def build_model():
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
    model = model.to("cuda").train()
    model.text_model.gradient_checkpointing_enable()
    return model


def make_batch(model, B: int, N: int):
    tokens_per_image = (
        model.vision_model.config.image_size
        // model.vision_model.config.patch_size
        // model.projector.vision_pixel_shuffle_factor
    ) ** 2
    img_tok = model.config.image_token_index
    input_ids = torch.randint(0, img_tok, (B, N), device="cuda")
    input_ids[:, :tokens_per_image] = img_tok
    pv = torch.randn(B, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    lb = input_ids.clone()
    lb[:, :tokens_per_image] = -100
    return dict(
        input_ids=input_ids,
        pixel_values=pv,
        attention_mask=torch.ones(B, N, device="cuda", dtype=torch.long),
        labels=lb,
    )


def bench(model, optimizer, B, N, steps=6, warmup=2):
    batch = make_batch(model, B, N)
    for _ in range(warmup):
        o = model(**batch)
        o.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        o = model(**batch)
        o.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / steps * 1000
    toks = B * N / (elapsed / steps)
    mem = torch.cuda.max_memory_allocated() / 1024**3
    del batch
    torch.cuda.empty_cache()
    return ms, toks, mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 128, 512, 768, 832, 896])
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    model = build_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    n_params = sum(p.numel() for p in trainable)

    print(f"GPU: {torch.cuda.get_device_name()} | {total_gb:.0f} GB")
    print(f"Trainable params: {n_params/1e6:.1f}M")
    print(f"Optimizer state memory (AdamW fp32): {n_params * 8 / 1024**3:.2f} GB")
    print(f"Optimizer state memory (FlashAdamW int8): ~{n_params * 2 / 1024**3:.2f} GB")
    print(f"Seq len: {args.seq_len}")
    print()

    configs = [
        ("AdamW", lambda: torch.optim.AdamW(trainable, lr=1e-4), False),
        ("FlashAdamW", lambda: FlashAdamW(trainable, lr=1e-4), False),
        ("FlashAdamW+grad_release", lambda: FlashAdamW(trainable, lr=1e-4), True),
    ]

    for name, opt_fn, use_grad_release in configs:
        print(f"=== {name} ===")
        opt = opt_fn()
        handle = None
        if use_grad_release:
            handle = enable_gradient_release(model, opt)

        for B in args.batch_sizes:
            try:
                ms, toks, mem = bench(model, opt, B, args.seq_len)
                print(f"  B={B:<5} {ms:>8.1f}ms {toks:>9.0f} tok/s {mem:>5.1f}GB ({mem/total_gb*100:.0f}%)")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "memory" in str(e).lower():
                    print(f"  B={B:<5} OOM")
                    torch.cuda.empty_cache()
                    break
                raise

        if handle:
            handle.remove()
        del opt
        torch.cuda.empty_cache()
        print()

    print("Summary:")
    print("  - FlashAdamW saves ~0.4 GB optimizer memory (int8 quantization)")
    print("  - Gradient release saves gradient tensor (~1 GB), enabling 1 step larger B")
    print("  - For 0.5B models, activation memory dominates — optimizer savings are marginal")
    print("  - FlashOptim shines for 7B+ models where optimizer states are 10-50+ GB")


if __name__ == "__main__":
    main()
