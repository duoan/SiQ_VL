"""End-to-end benchmark: cuTile vs SDPA vs FlexAttention for training step.

Compares attention backends on a full training forward+backward pass using
the actual SiQ-VL model architecture.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


def create_synthetic_batch(batch_size, seq_len, vocab_size, image_tokens, image_token_id, device="cuda"):
    """Create a synthetic training batch mimicking real data."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Insert image placeholder tokens at the start of each sequence
    input_ids[:, :image_tokens] = image_token_id
    labels = input_ids.clone()
    labels[:, :image_tokens] = -100
    attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=torch.bfloat16)
    num_image_tokens = torch.tensor([image_tokens] * batch_size, device=device)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "num_image_tokens": num_image_tokens,
    }


def benchmark_backend(backend: str, steps: int = 20, warmup: int = 5, batch_size: int = 4, seq_len: int = 512):
    """Benchmark a specific attention backend."""
    from siq_vl.model.modeling import get_stage1_model_and_processor

    use_packing = backend == "flex_attention"
    use_cutile = backend == "cutile"

    model, processor = get_stage1_model_and_processor(
        pretrained_vision_model_path="google/siglip2-base-patch16-224",
        pretrained_text_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_packing=use_packing,
        use_cutile=use_cutile,
    )
    model = model.to("cuda")
    model.train()

    vocab_size = model.vocab_size
    image_token_id = model.config.image_token_index
    image_tokens = 49  # siglip2-base-224 after pixel shuffle (196 patches / 4)

    # Create batch
    batch = create_synthetic_batch(batch_size, seq_len, vocab_size, image_tokens, image_token_id)

    # Warmup
    print(f"  Warming up ({warmup} steps)...", flush=True)
    for _ in range(warmup):
        outputs = model(**batch)
        outputs.loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"  Benchmarking ({steps} steps)...", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        outputs = model(**batch)
        outputs.loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    ms_per_step = elapsed / steps * 1000
    tokens_per_sec = batch_size * seq_len / (elapsed / steps)

    # Cleanup
    del model, batch
    torch.cuda.empty_cache()

    return {
        "backend": backend,
        "ms_per_step": ms_per_step,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_gb": peak_mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["sdpa", "cutile"], choices=["sdpa", "cutile", "flex_attention"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: batch_size={args.batch_size}, seq_len={args.seq_len}, steps={args.steps}")
    print(f"Backends: {args.backends}")
    print("=" * 70)

    results = []
    for backend in args.backends:
        print(f"\n[{backend.upper()}]")
        result = benchmark_backend(
            backend,
            steps=args.steps,
            warmup=args.warmup,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        results.append(result)
        print(f"  -> {result['ms_per_step']:.2f} ms/step, {result['tokens_per_sec']:.0f} tok/s, {result['peak_mem_gb']:.2f} GB peak")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Backend':<20} {'ms/step':<12} {'tok/s':<12} {'Peak GB':<10} {'vs SDPA':<10}")
    print("-" * 70)

    sdpa_ms = next((r["ms_per_step"] for r in results if r["backend"] == "sdpa"), None)
    for r in results:
        speedup = f"{sdpa_ms / r['ms_per_step']:.2f}x" if sdpa_ms else "N/A"
        print(f"{r['backend']:<20} {r['ms_per_step']:<12.2f} {r['tokens_per_sec']:<12.0f} {r['peak_mem_gb']:<10.2f} {speedup:<10}")


if __name__ == "__main__":
    main()
