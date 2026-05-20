"""Benchmark packing vs bucketing on mixed dataset.

Compares throughput, VRAM, and padding waste between:
1. Standard collation + length bucketing + SDPA + torch.compile
2. PackingCollator + FlexAttention + torch.compile

Used in Iteration 10 to validate that packing with FlexAttention outperforms
bucketing with SDPA on diverse mixed-length datasets.

Usage:
    python scripts/benchmark_packing.py
    python scripts/benchmark_packing.py --mode packing --pack_max_length 2048
    python scripts/benchmark_packing.py --mode both --fetch_bs 20 --warmup_steps 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import LengthGroupedSampler

ALL_SUBSETS = [
    "coco_colors",
    "densefusion_1m",
    "sharegpt4v(coco)",
    "sharegpt4v(knowledge)",
    "laion_gpt4v",
    "sharegpt4o",
]


def load_mixed_dataset(subsets: list[str], max_per_subset: int = 150):
    """Load and concatenate multiple dataset subsets."""
    all_ds = []
    for name in subsets:
        ds = load_dataset("HuggingFaceM4/FineVision", name=name, split="train", num_proc=4)
        all_ds.append(ds.select(range(min(max_per_subset, len(ds)))))
    return concatenate_datasets(all_ds).shuffle(seed=42)


def benchmark_bucketed(model, processor, dataset, args) -> dict:
    """Benchmark standard bucketed approach (SDPA + compile)."""
    from siq_vl.collator import SiQ_VLDataCollator

    collator = SiQ_VLDataCollator(processor=processor, max_length=args.max_length)
    model_keys = ["input_ids", "pixel_values", "attention_mask", "labels", "num_image_tokens"]

    sampler = LengthGroupedSampler(args.fetch_bs, lengths=dataset.lengths)
    dl = DataLoader(
        dataset, batch_size=args.fetch_bs, sampler=sampler, num_workers=2, collate_fn=collator
    )
    data_iter = iter(dl)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Warmup
    print("  Warmup...", flush=True)
    for i in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(**{k: v for k, v in batch_gpu.items() if k in model_keys})
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Measure
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, total_real, total_positions = [], 0, 0
    seq_lens = []

    for i in range(args.measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        n_real = int(batch["attention_mask"].sum().item())
        n_total = batch["input_ids"].numel()
        seq_lens.append(batch["input_ids"].shape[1])
        total_real += n_real
        total_positions += n_total
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**{k: v for k, v in batch_gpu.items() if k in model_keys})
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": "bucketed",
        "steps": len(times),
        "avg_step_ms": sum(times) / len(times),
        "real_tok_per_sec": total_real / (sum(times) / 1000),
        "padding_waste_pct": (total_positions - total_real) / total_positions * 100,
        "vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "seq_len_min": min(seq_lens),
        "seq_len_max": max(seq_lens),
        "seq_len_mean": float(np.mean(seq_lens)),
        "real_tok_per_step": total_real / len(times),
        "batch_size": args.fetch_bs,
    }


def benchmark_packing(model, processor, dataset, args) -> dict:
    """Benchmark packing approach (FlexAttention + compile)."""
    from siq_vl.collator import PackingCollator

    collator = PackingCollator(
        processor=processor, pack_max_length=args.pack_max_length, max_length=args.max_length
    )
    model_keys = ["input_ids", "pixel_values", "position_ids", "labels", "num_image_tokens"]
    pad_token_id = processor.tokenizer.pad_token_id

    dl = DataLoader(dataset, batch_size=args.fetch_bs, num_workers=2, collate_fn=collator, shuffle=True)
    data_iter = iter(dl)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Warmup
    print("  Warmup...", flush=True)
    for i in range(args.warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(**{k: v for k, v in batch_gpu.items() if k in model_keys})
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Measure
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, total_real, total_positions = [], 0, 0
    bin_counts = []

    for i in range(args.measure_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch = next(data_iter)
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B, L = batch["input_ids"].shape
        n_real = int((batch["input_ids"] != pad_token_id).sum().item())
        n_total = B * L
        total_real += n_real
        total_positions += n_total
        bin_counts.append(B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**{k: v for k, v in batch_gpu.items() if k in model_keys})
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": "packing",
        "steps": len(times),
        "avg_step_ms": sum(times) / len(times),
        "real_tok_per_sec": total_real / (sum(times) / 1000),
        "padding_waste_pct": (total_positions - total_real) / total_positions * 100,
        "vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "pack_max_length": args.pack_max_length,
        "bins_min": min(bin_counts),
        "bins_max": max(bin_counts),
        "bins_mean": float(np.mean(bin_counts)),
        "real_tok_per_step": total_real / len(times),
        "fetch_batch_size": args.fetch_bs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark packing vs bucketing")
    parser.add_argument("--mode", choices=["bucketed", "packing", "both"], default="both")
    parser.add_argument("--subsets", nargs="+", default=ALL_SUBSETS)
    parser.add_argument("--max_per_subset", type=int, default=150)
    parser.add_argument("--fetch_bs", type=int, default=20)
    parser.add_argument("--pack_max_length", type=int, default=1536)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=15)
    parser.add_argument("--compile_mode", type=str, default="dynamic", help="dynamic or max-autotune-no-cudagraphs")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    import logging

    logging.disable(logging.CRITICAL)

    from siq_vl.dataset import VQADataset
    from siq_vl.model import modeling as _m

    _m._LIGER_APPLIED = True

    results = {
        "config": {
            "subsets": args.subsets,
            "max_per_subset": args.max_per_subset,
            "fetch_bs": args.fetch_bs,
            "pack_max_length": args.pack_max_length,
            "max_length": args.max_length,
            "compile_mode": args.compile_mode,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
        },
        "results": {},
    }

    # Load dataset
    mixed_ds = load_mixed_dataset(args.subsets, args.max_per_subset)
    dataset = VQADataset(mixed_ds)
    print(f"Dataset: {len(dataset)} samples from {len(args.subsets)} subsets\n")

    if args.mode in ("bucketed", "both"):
        print("=== BUCKETED (SDPA + compile) ===", flush=True)
        from siq_vl.model.modeling import get_stage1_model_and_processor

        model, processor = get_stage1_model_and_processor(
            pretrained_vision_model_path="google/siglip2-so400m-patch16-512",
            pretrained_text_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            vision_pixel_shuffle_factor=4,
            use_packing=False,
        )
        model = model.cuda().train()
        if args.compile_mode == "dynamic":
            model = torch.compile(model, dynamic=True)
        else:
            model = torch.compile(model, mode=args.compile_mode)

        bucketed_result = benchmark_bucketed(model, processor, dataset, args)
        results["results"]["bucketed"] = bucketed_result
        print(f"  Result: {bucketed_result['real_tok_per_sec']:.0f} tok/s, "
              f"{bucketed_result['padding_waste_pct']:.1f}% waste, "
              f"{bucketed_result['vram_gb']:.1f} GB\n")
        del model, processor
        torch.cuda.empty_cache()

    if args.mode in ("packing", "both"):
        print("=== PACKING (FlexAttention + compile) ===", flush=True)
        from siq_vl.model.modeling import get_stage1_model_and_processor

        model, processor = get_stage1_model_and_processor(
            pretrained_vision_model_path="google/siglip2-so400m-patch16-512",
            pretrained_text_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            vision_pixel_shuffle_factor=4,
            use_packing=True,
        )
        model = model.cuda().train()
        if args.compile_mode == "dynamic":
            model = torch.compile(model, dynamic=True)
        else:
            model = torch.compile(model, mode=args.compile_mode)

        packing_result = benchmark_packing(model, processor, dataset, args)
        results["results"]["packing"] = packing_result
        print(f"  Result: {packing_result['real_tok_per_sec']:.0f} tok/s, "
              f"{packing_result['padding_waste_pct']:.1f}% waste, "
              f"{packing_result['vram_gb']:.1f} GB\n")
        del model, processor
        torch.cuda.empty_cache()

    # Summary
    if args.mode == "both" and "bucketed" in results["results"] and "packing" in results["results"]:
        b = results["results"]["bucketed"]
        p = results["results"]["packing"]
        speedup = p["real_tok_per_sec"] / b["real_tok_per_sec"]
        vram_saving = (b["vram_gb"] - p["vram_gb"]) / b["vram_gb"] * 100
        print("=" * 60)
        print("COMPARISON:")
        print(f"  Bucketed: {b['real_tok_per_sec']:.0f} tok/s, {b['vram_gb']:.1f} GB")
        print(f"  Packing:  {p['real_tok_per_sec']:.0f} tok/s, {p['vram_gb']:.1f} GB")
        print(f"  Speedup:  {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")
        print(f"  VRAM saved: {vram_saving:.1f}%")
        results["comparison"] = {
            "throughput_speedup": speedup,
            "vram_saving_pct": vram_saving,
        }

    output_path = args.output or "docs/traces/iter_10_packing_benchmark.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
