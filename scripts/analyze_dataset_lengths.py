"""Comprehensive token length analysis for seq_length selection.

Produces per-subset statistics and simulates truncation/packing efficiency
at multiple candidate seq_lengths. Used to make data-driven decisions about
ConstantLengthDataset's seq_length parameter for Stage 1 and Stage 2.

Usage:
    python scripts/analyze_dataset_lengths.py
    python scripts/analyze_dataset_lengths.py --max_samples 5000
    python scripts/analyze_dataset_lengths.py --subsets sharegpt4v(coco) densefusion_1m
    python scripts/analyze_dataset_lengths.py --tokenizer Qwen/Qwen2.5-0.5B-Instruct --image_tokens 49
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ALL_SUBSETS = [
    "coco_colors",
    "densefusion_1m",
    "face_emotion",
    "google_landmarks",
    "laion_gpt4v",
    "sharegpt4o",
    "sharegpt4v(coco)",
    "sharegpt4v(llava)",
    "sharegpt4v(knowledge)",
    "sharegpt4v(sam)",
]

STAGE1_SUBSETS = [
    "coco_colors",
    "sharegpt4v(coco)",
    "face_emotion",
    "google_landmarks",
    "sharegpt4v(sam)",
]

STAGE2_SUBSETS = ALL_SUBSETS

CANDIDATE_SEQ_LENGTHS = [512, 768, 1024, 1536, 2048, 3072, 4096]

CHAT_TEMPLATE_OVERHEAD = 85


def extract_qa_from_item(item: dict) -> tuple[str, str]:
    """Extract question/answer from a FineVision item, handling both field formats."""
    texts = item.get("texts", [])
    if texts and len(texts) > 0:
        turn = texts[0]
        return turn.get("user", ""), turn.get("assistant", "")

    conversations = item.get("conversations", [])
    if len(conversations) >= 2:
        return conversations[0].get("value", ""), conversations[1].get("value", "")

    return "Describe this image.", ""


def tokenize_length(tokenizer, question: str, answer: str, image_tokens: int) -> int:
    """Compute exact token count for a VQA sample including image + chat template."""
    text_tokens = len(tokenizer.encode(question + answer, add_special_tokens=False))
    return text_tokens + CHAT_TEMPLATE_OVERHEAD + image_tokens


def analyze_subset(tokenizer, subset_name: str, max_samples: int, image_tokens: int, data_path: str) -> tuple:
    """Analyze length distribution for a single dataset subset. Returns (stats_dict, lengths_array)."""
    try:
        ds = load_dataset(data_path, name=subset_name, split="train", num_proc=4)
    except Exception as e:
        print(f"    SKIP {subset_name}: {e}")
        return None, np.array([])

    n = min(max_samples, len(ds))
    ds = ds.select(range(n))

    lengths = []
    skipped = 0
    for item in ds:
        q, a = extract_qa_from_item(item)
        if not q and not a:
            skipped += 1
            continue
        lengths.append(tokenize_length(tokenizer, q, a, image_tokens))

    if not lengths:
        return None, np.array([])

    lengths = np.array(lengths)
    stats = {
        "subset": subset_name,
        "n_samples": len(lengths),
        "n_skipped": skipped,
        "mean": round(float(np.mean(lengths))),
        "median": round(float(np.median(lengths))),
        "std": round(float(np.std(lengths))),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": round(float(np.percentile(lengths, 25))),
        "p50": round(float(np.percentile(lengths, 50))),
        "p75": round(float(np.percentile(lengths, 75))),
        "p90": round(float(np.percentile(lengths, 90))),
        "p95": round(float(np.percentile(lengths, 95))),
        "p99": round(float(np.percentile(lengths, 99))),
    }
    return stats, lengths


def simulate_seq_length(lengths: np.ndarray, seq_length: int) -> dict:
    """Simulate truncation and packing for a given seq_length target."""
    total_tokens = int(lengths.sum())
    n = len(lengths)

    truncated_mask = lengths > seq_length
    truncation_rate = float(truncated_mask.sum()) / n
    tokens_lost = int(np.maximum(lengths - seq_length, 0).sum())
    token_loss_rate = tokens_lost / total_tokens if total_tokens > 0 else 0

    clamped = np.minimum(lengths, seq_length)

    bins = []
    bin_fills = []
    sorted_desc = np.sort(clamped)[::-1]
    for length in sorted_desc:
        placed = False
        for i, fill in enumerate(bin_fills):
            if fill + length <= seq_length:
                bin_fills[i] += length
                bins[i] += 1
                placed = True
                break
        if not placed:
            bin_fills.append(int(length))
            bins.append(1)

    packing_total = len(bin_fills) * seq_length
    packing_real = int(clamped.sum())
    packing_efficiency = packing_real / packing_total if packing_total > 0 else 0
    avg_samples_per_bin = float(np.mean(bins)) if bins else 0

    return {
        "seq_length": seq_length,
        "truncation_rate": round(truncation_rate, 4),
        "token_loss_rate": round(token_loss_rate, 4),
        "packing_efficiency": round(packing_efficiency, 4),
        "avg_samples_per_bin": round(avg_samples_per_bin, 2),
        "num_bins": len(bin_fills),
        "samples_truncated": int(truncated_mask.sum()),
        "tokens_lost": tokens_lost,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive token length analysis for seq_length selection")
    parser.add_argument("--subsets", nargs="+", default=ALL_SUBSETS)
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples per subset")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--image_tokens", type=int, default=49, help="Image tokens per tile (49 for small, 169 for large)")
    parser.add_argument("--data_path", type=str, default="HuggingFaceM4/FineVision")
    parser.add_argument("--candidates", nargs="+", type=int, default=CANDIDATE_SEQ_LENGTHS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"=== Seq Length Analysis ===")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Image tokens/tile: {args.image_tokens}")
    print(f"Subsets: {len(args.subsets)}, max {args.max_samples} samples each")
    print(f"Candidates: {args.candidates}\n")

    subset_stats = []
    all_lengths = []
    subset_lengths = {}

    for subset in args.subsets:
        print(f"  [{args.subsets.index(subset)+1}/{len(args.subsets)}] {subset}...", end=" ", flush=True)
        stats, lens = analyze_subset(tokenizer, subset, args.max_samples, args.image_tokens, args.data_path)
        if stats is None:
            print("FAILED")
            continue
        subset_stats.append(stats)
        subset_lengths[subset] = lens.tolist()
        all_lengths.extend(lens.tolist())
        print(f"n={stats['n_samples']}, mean={stats['mean']}, p95={stats['p95']}, max={stats['max']}")

    all_lengths = np.array(all_lengths)
    print(f"\n{'='*70}")
    print(f"COMBINED: {len(all_lengths)} samples from {len(subset_stats)} subsets")
    print(f"  Mean: {np.mean(all_lengths):.0f}, Median: {np.median(all_lengths):.0f}, Std: {np.std(all_lengths):.0f}")
    print(f"  Range: {np.min(all_lengths)} – {np.max(all_lengths)}")
    print(f"  P75: {np.percentile(all_lengths, 75):.0f}, P90: {np.percentile(all_lengths, 90):.0f}")
    print(f"  P95: {np.percentile(all_lengths, 95):.0f}, P99: {np.percentile(all_lengths, 99):.0f}")

    combined_stats = {
        "n_samples": len(all_lengths),
        "mean": round(float(np.mean(all_lengths))),
        "median": round(float(np.median(all_lengths))),
        "std": round(float(np.std(all_lengths))),
        "min": int(np.min(all_lengths)),
        "max": int(np.max(all_lengths)),
        "p25": round(float(np.percentile(all_lengths, 25))),
        "p50": round(float(np.percentile(all_lengths, 50))),
        "p75": round(float(np.percentile(all_lengths, 75))),
        "p90": round(float(np.percentile(all_lengths, 90))),
        "p95": round(float(np.percentile(all_lengths, 95))),
        "p99": round(float(np.percentile(all_lengths, 99))),
    }

    print(f"\n{'='*70}")
    print("SEQ_LENGTH SIMULATION (all data combined)")
    print(f"{'seq_len':>8} {'trunc%':>8} {'tok_loss%':>10} {'pack_eff%':>10} {'samp/bin':>9}")
    print(f"{'-'*8:>8} {'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*9:>9}")

    combined_simulations = []
    for sl in args.candidates:
        sim = simulate_seq_length(all_lengths, sl)
        combined_simulations.append(sim)
        print(f"{sl:>8} {sim['truncation_rate']*100:>7.1f}% {sim['token_loss_rate']*100:>9.2f}% {sim['packing_efficiency']*100:>9.1f}% {sim['avg_samples_per_bin']:>9.2f}")

    stage1_lengths = []
    for s in STAGE1_SUBSETS:
        if s in subset_lengths:
            stage1_lengths.extend(subset_lengths[s])
    stage1_lengths = np.array(stage1_lengths) if stage1_lengths else np.array([0])

    stage2_lengths = []
    for s in STAGE2_SUBSETS:
        if s in subset_lengths:
            stage2_lengths.extend(subset_lengths[s])
    stage2_lengths = np.array(stage2_lengths) if stage2_lengths else np.array([0])

    print(f"\n{'='*70}")
    print(f"STAGE 1 data ({len(stage1_lengths)} samples from {[s for s in STAGE1_SUBSETS if s in subset_lengths]})")
    if len(stage1_lengths) > 0 and stage1_lengths.sum() > 0:
        print(f"  Mean: {np.mean(stage1_lengths):.0f}, P95: {np.percentile(stage1_lengths, 95):.0f}, P99: {np.percentile(stage1_lengths, 99):.0f}, Max: {np.max(stage1_lengths)}")
        print(f"\n  {'seq_len':>8} {'trunc%':>8} {'tok_loss%':>10} {'pack_eff%':>10} {'samp/bin':>9}")
        stage1_simulations = []
        for sl in args.candidates:
            sim = simulate_seq_length(stage1_lengths, sl)
            stage1_simulations.append(sim)
            print(f"  {sl:>8} {sim['truncation_rate']*100:>7.1f}% {sim['token_loss_rate']*100:>9.2f}% {sim['packing_efficiency']*100:>9.1f}% {sim['avg_samples_per_bin']:>9.2f}")
    else:
        stage1_simulations = []

    print(f"\n{'='*70}")
    print(f"STAGE 2 data ({len(stage2_lengths)} samples from {[s for s in STAGE2_SUBSETS if s in subset_lengths]})")
    if len(stage2_lengths) > 0 and stage2_lengths.sum() > 0:
        print(f"  Mean: {np.mean(stage2_lengths):.0f}, P95: {np.percentile(stage2_lengths, 95):.0f}, P99: {np.percentile(stage2_lengths, 99):.0f}, Max: {np.max(stage2_lengths)}")
        print(f"\n  {'seq_len':>8} {'trunc%':>8} {'tok_loss%':>10} {'pack_eff%':>10} {'samp/bin':>9}")
        stage2_simulations = []
        for sl in args.candidates:
            sim = simulate_seq_length(stage2_lengths, sl)
            stage2_simulations.append(sim)
            print(f"  {sl:>8} {sim['truncation_rate']*100:>7.1f}% {sim['token_loss_rate']*100:>9.2f}% {sim['packing_efficiency']*100:>9.1f}% {sim['avg_samples_per_bin']:>9.2f}")
    else:
        stage2_simulations = []

    results = {
        "config": {
            "tokenizer": args.tokenizer,
            "image_tokens_per_tile": args.image_tokens,
            "chat_template_overhead": CHAT_TEMPLATE_OVERHEAD,
            "max_samples_per_subset": args.max_samples,
            "data_path": args.data_path,
            "candidate_seq_lengths": args.candidates,
            "stage1_subsets": STAGE1_SUBSETS,
            "stage2_subsets": STAGE2_SUBSETS,
        },
        "per_subset": subset_stats,
        "combined": combined_stats,
        "combined_simulations": combined_simulations,
        "stage1": {
            "n_samples": len(stage1_lengths),
            "mean": round(float(np.mean(stage1_lengths))) if len(stage1_lengths) > 0 else 0,
            "p95": round(float(np.percentile(stage1_lengths, 95))) if len(stage1_lengths) > 0 else 0,
            "p99": round(float(np.percentile(stage1_lengths, 99))) if len(stage1_lengths) > 0 else 0,
            "max": int(np.max(stage1_lengths)) if len(stage1_lengths) > 0 else 0,
            "simulations": stage1_simulations,
        },
        "stage2": {
            "n_samples": len(stage2_lengths),
            "mean": round(float(np.mean(stage2_lengths))) if len(stage2_lengths) > 0 else 0,
            "p95": round(float(np.percentile(stage2_lengths, 95))) if len(stage2_lengths) > 0 else 0,
            "p99": round(float(np.percentile(stage2_lengths, 99))) if len(stage2_lengths) > 0 else 0,
            "max": int(np.max(stage2_lengths)) if len(stage2_lengths) > 0 else 0,
            "simulations": stage2_simulations,
        },
    }

    output_path = args.output or "docs/traces/seq_length_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
