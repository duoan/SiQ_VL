"""Analyze token length distribution across FineVision dataset subsets.

Used in Iteration 10 to understand padding waste across mixed-data training
and determine whether sample packing would be effective.

Usage:
    python scripts/analyze_dataset_lengths.py
    python scripts/analyze_dataset_lengths.py --subsets sharegpt4v(coco) densefusion_1m
    python scripts/analyze_dataset_lengths.py --max_samples 500
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import Qwen2TokenizerFast

ALL_SUBSETS = [
    "coco_colors",
    "densefusion_1m",
    "sharegpt4v(coco)",
    "sharegpt4v(llava)",
    "sharegpt4v(knowledge)",
    "laion_gpt4v",
    "sharegpt4o",
]

CHAT_TEMPLATE_OVERHEAD = 85  # approximate tokens for system prompt + chat formatting
IMAGE_TOKENS_PER_TILE = 64  # after pixel shuffle factor=4


def estimate_token_length(tokenizer, question: str, answer: str) -> int:
    """Estimate total token count for a VQA sample (including image tokens and chat template)."""
    text_tokens = len(tokenizer.encode(question + answer, add_special_tokens=False))
    return text_tokens + CHAT_TEMPLATE_OVERHEAD + IMAGE_TOKENS_PER_TILE


def analyze_subset(tokenizer, subset_name: str, max_samples: int) -> dict:
    """Analyze length distribution for a single dataset subset."""
    ds = load_dataset("HuggingFaceM4/FineVision", name=subset_name, split="train", num_proc=4)
    n = min(max_samples, len(ds))
    ds = ds.select(range(n))

    lengths = []
    for item in ds:
        conversations = item.get("conversations", [])
        if len(conversations) >= 2:
            q = conversations[0].get("value", "")
            a = conversations[1].get("value", "")
        else:
            q = "Describe this image."
            a = ""
        lengths.append(estimate_token_length(tokenizer, q, a))

    lengths = np.array(lengths)
    return {
        "subset": subset_name,
        "n_samples": n,
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }


def compute_padding_stats(all_lengths: np.ndarray, batch_size: int) -> dict:
    """Compute padding waste under different strategies."""
    n = len(all_lengths)

    # Random order padding waste
    random_waste_positions = 0
    random_total_positions = 0
    indices = np.random.default_rng(42).permutation(n)
    for i in range(0, n - batch_size + 1, batch_size):
        batch_lens = all_lengths[indices[i : i + batch_size]]
        max_len = batch_lens.max()
        random_waste_positions += (max_len * batch_size) - batch_lens.sum()
        random_total_positions += max_len * batch_size

    # Sorted (bucketed) padding waste
    sorted_lens = np.sort(all_lengths)
    bucketed_waste_positions = 0
    bucketed_total_positions = 0
    for i in range(0, n - batch_size + 1, batch_size):
        batch_lens = sorted_lens[i : i + batch_size]
        max_len = batch_lens.max()
        bucketed_waste_positions += (max_len * batch_size) - batch_lens.sum()
        bucketed_total_positions += max_len * batch_size

    # Packing waste (first-fit-decreasing into bins of p95 length)
    pack_target = int(np.percentile(all_lengths, 95))
    sorted_desc = np.sort(all_lengths)[::-1]
    bin_fills = []
    for length in sorted_desc:
        placed = False
        for i, fill in enumerate(bin_fills):
            if fill + length <= pack_target:
                bin_fills[i] += length
                placed = True
                break
        if not placed:
            bin_fills.append(length)
    packing_total = len(bin_fills) * pack_target
    packing_real = int(all_lengths.sum())
    packing_waste = (packing_total - packing_real) / packing_total * 100

    return {
        "batch_size": batch_size,
        "random_waste_pct": random_waste_positions / random_total_positions * 100 if random_total_positions else 0,
        "bucketed_waste_pct": bucketed_waste_positions / bucketed_total_positions * 100 if bucketed_total_positions else 0,
        "packing_waste_pct": packing_waste,
        "packing_target_length": pack_target,
        "packing_n_bins": len(bin_fills),
        "packing_avg_utilization_pct": np.mean(bin_fills) / pack_target * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze token length distribution across dataset subsets")
    parser.add_argument("--subsets", nargs="+", default=ALL_SUBSETS, help="Subsets to analyze")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples per subset")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for padding analysis")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.tokenizer)
    print(f"Analyzing {len(args.subsets)} subsets, max {args.max_samples} samples each\n")

    subset_stats = []
    all_lengths = []

    for subset in args.subsets:
        print(f"  Processing {subset}...", end=" ", flush=True)
        stats = analyze_subset(tokenizer, subset, args.max_samples)
        subset_stats.append(stats)
        # Re-collect lengths for combined analysis
        ds = load_dataset("HuggingFaceM4/FineVision", name=subset, split="train", num_proc=4)
        n = min(args.max_samples, len(ds))
        ds = ds.select(range(n))
        for item in ds:
            conversations = item.get("conversations", [])
            if len(conversations) >= 2:
                q = conversations[0].get("value", "")
                a = conversations[1].get("value", "")
            else:
                q, a = "Describe this image.", ""
            all_lengths.append(estimate_token_length(tokenizer, q, a))
        print(f"mean={stats['mean']:.0f}, p95={stats['p95']:.0f}, max={stats['max']}")

    all_lengths = np.array(all_lengths)

    print(f"\n{'='*60}")
    print(f"COMBINED ({len(all_lengths)} samples from {len(args.subsets)} subsets)")
    print(f"{'='*60}")
    print(f"  Mean: {np.mean(all_lengths):.0f} tokens")
    print(f"  Median: {np.median(all_lengths):.0f}")
    print(f"  Std: {np.std(all_lengths):.0f}")
    print(f"  Range: {np.min(all_lengths)} – {np.max(all_lengths)}")
    print(f"  P95: {np.percentile(all_lengths, 95):.0f}")
    print(f"  P99: {np.percentile(all_lengths, 99):.0f}")

    padding_stats = compute_padding_stats(all_lengths, args.batch_size)
    print(f"\nPadding Analysis (batch_size={args.batch_size}):")
    print(f"  Random order waste:  {padding_stats['random_waste_pct']:.1f}%")
    print(f"  Bucketed waste:      {padding_stats['bucketed_waste_pct']:.1f}%")
    print(f"  Packing waste:       {padding_stats['packing_waste_pct']:.1f}% (target={padding_stats['packing_target_length']})")
    print(f"  Packing utilization: {padding_stats['packing_avg_utilization_pct']:.1f}%")

    results = {
        "config": {
            "subsets": args.subsets,
            "max_samples_per_subset": args.max_samples,
            "batch_size": args.batch_size,
            "tokenizer": args.tokenizer,
        },
        "per_subset": subset_stats,
        "combined": {
            "n_samples": len(all_lengths),
            "mean": float(np.mean(all_lengths)),
            "median": float(np.median(all_lengths)),
            "std": float(np.std(all_lengths)),
            "min": int(np.min(all_lengths)),
            "max": int(np.max(all_lengths)),
            "p95": float(np.percentile(all_lengths, 95)),
            "p99": float(np.percentile(all_lengths, 99)),
        },
        "padding_analysis": padding_stats,
    }

    output_path = args.output or "docs/traces/dataset_length_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
