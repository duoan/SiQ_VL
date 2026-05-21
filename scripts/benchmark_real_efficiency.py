"""Ground-truth efficiency measurement on REAL data.

Measures ALL metrics precisely by running actual training steps with the real
data pipeline (not synthetic data). Shows exactly how many positions are padding
vs real tokens, and computes true effective throughput.

Metrics:
  - Pos/s (hw): Total positions processed per second (B × N_padded / time)
  - Real tok/s: Non-padding tokens per second (attention_mask.sum() / time)
  - Loss tok/s: Tokens contributing to loss per second ((labels != -100).sum() / time)
  - Pad%: Padding waste = 1 - Real/Pos
  - Loss%: Fraction of real tokens that produce loss

Usage:
    python scripts/benchmark_real_efficiency.py --stage 1
    python scripts/benchmark_real_efficiency.py --stage 2 --use_tilegym
    python scripts/benchmark_real_efficiency.py --stage 1 --batch_size 16 --use_bucketing
"""

import argparse
import time

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

from siq_vl.collator import PackingCollator, SiQ_VLDataCollator
from siq_vl.model.modeling import get_stage1_model_and_processor


class RealVQADataset(Dataset):
    """Adapts LLaVA-OneVision format to {image, question, answer} dicts."""

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self._lengths = None

    @property
    def lengths(self) -> list[int]:
        if self._lengths is None:
            self._lengths = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                convs = item.get("conversations", [])
                total_chars = sum(len(c.get("value", "")) for c in convs)
                self._lengths.append(int(total_chars / 3.5) + 80 + 196)
        return self._lengths

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item.get("image")
        convs = item.get("conversations", [])

        if image is None or len(convs) < 2:
            return None

        if not isinstance(image, Image.Image):
            return None

        if image.mode != "RGB":
            image = image.convert("RGB")

        question = convs[0].get("value", "").replace("<image>\n", "").replace("<image>", "")
        answer = convs[1].get("value", "")

        if not question or not answer:
            return None

        return {"image": image, "question": question, "answer": answer}


def measure_batch_stats(batch: dict) -> dict:
    """Extract precise token statistics from a collated batch."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    B, N = input_ids.shape
    total_positions = B * N

    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        real_tokens = int(attention_mask.sum().item())
    else:
        # Packed sequences: all positions are real (no padding)
        real_tokens = total_positions

    loss_tokens = int((labels != -100).sum().item())
    padding_tokens = total_positions - real_tokens

    return {
        "B": B,
        "N": N,
        "total_positions": total_positions,
        "real_tokens": real_tokens,
        "loss_tokens": loss_tokens,
        "padding_tokens": padding_tokens,
        "pad_pct": padding_tokens / total_positions * 100,
        "loss_pct": loss_tokens / real_tokens * 100 if real_tokens > 0 else 0,
    }


def run_benchmark(
    model,
    dataloader,
    steps: int = 20,
    warmup: int = 3,
    device: str = "cuda",
):
    """Run training steps and measure all efficiency metrics."""
    model.train()

    # Accumulate stats
    all_stats = []
    step_times = []

    step_count = 0
    for batch in dataloader:
        if step_count >= warmup + steps:
            break

        if batch is None:
            continue

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Measure batch composition BEFORE forward pass
        stats = measure_batch_stats(batch)

        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(**batch)
            outputs.loss.backward()
            model.zero_grad()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        except Exception as e:
            print(f"  [step {step_count}] ERROR: {e}")
            step_count += 1
            continue

        if step_count >= warmup:
            stats["step_ms"] = elapsed * 1000
            all_stats.append(stats)
            step_times.append(elapsed)
            if step_count == warmup:
                print(f"  [first measured step] B={stats['B']}, N={stats['N']}, "
                      f"real={stats['real_tokens']}, pad%={stats['pad_pct']:.1f}%, "
                      f"time={elapsed*1000:.1f}ms")

        step_count += 1

    if not all_stats:
        return None

    # Aggregate
    import numpy as np

    total_pos = sum(s["total_positions"] for s in all_stats)
    total_real = sum(s["real_tokens"] for s in all_stats)
    total_loss = sum(s["loss_tokens"] for s in all_stats)
    total_time = sum(step_times)

    return {
        "steps": len(all_stats),
        "avg_B": np.mean([s["B"] for s in all_stats]),
        "avg_N": np.mean([s["N"] for s in all_stats]),
        "avg_step_ms": np.mean([s["step_ms"] for s in all_stats]),
        "pos_per_sec": total_pos / total_time,
        "real_tok_per_sec": total_real / total_time,
        "loss_tok_per_sec": total_loss / total_time,
        "pad_pct": (1 - total_real / total_pos) * 100,
        "loss_pct": total_loss / total_real * 100 if total_real > 0 else 0,
        "vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_bucketing", action="store_true")
    parser.add_argument("--use_packing", action="store_true")
    parser.add_argument("--pack_max_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)
    parser.add_argument("--use_tilegym", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="lmms-lab/LLaVA-OneVision-Data")
    parser.add_argument("--sub_set", type=str, default="sharegpt4v(coco)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.use_tilegym:
        from tilegym.transformers import apply_tilegym_kernel_to_qwen2

        from siq_vl.kernels.fused_linear_ce import patch_qwen2_fused_linear_ce

        apply_tilegym_kernel_to_qwen2(use_cutile=True)
        patch_qwen2_fused_linear_ce()

    # Load model
    model, processor = get_stage1_model_and_processor(
        pretrained_vision_model_path="google/siglip2-base-patch16-224",
        pretrained_text_model_path="Qwen/Qwen2.5-0.5B-Instruct",
        use_tilegym=args.use_tilegym,
    )
    model = model.to("cuda")

    # Load dataset
    print(f"Loading dataset: {args.data_path} / {args.sub_set}")
    hf_ds = load_dataset(args.data_path, args.sub_set, split="train")
    dataset = RealVQADataset(hf_ds)
    print(f"Dataset size: {len(dataset)} samples")

    # Setup collator
    if args.use_packing:
        collator = PackingCollator(
            processor=processor,
            pack_max_length=args.pack_max_length,
            max_length=args.max_length,
        )
    else:
        collator = SiQ_VLDataCollator(
            processor=processor,
            max_length=args.max_length,
            pad_to_multiple_of=args.pad_to_multiple_of,
        )

    # Wrap collator to skip all-None batches gracefully
    def safe_collate(features):
        features = [f for f in features if f is not None]
        if not features:
            return None
        return collator(features)

    # Setup sampler (bucketing = sort by length)
    if args.use_bucketing:
        lengths = dataset.lengths
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        # Take middle section (avoid extremes)
        mid = len(sorted_indices) // 3
        sampler = sorted_indices[mid : mid + args.batch_size * (args.steps + args.warmup) * 2]

        class IndexSampler:
            def __init__(self, indices):
                self.indices = indices

            def __iter__(self):
                return iter(self.indices)

            def __len__(self):
                return len(self.indices)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=IndexSampler(sampler),
            collate_fn=safe_collate,
            num_workers=0,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=safe_collate,
            num_workers=0,
            pin_memory=True,
        )

    # Print config
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"Stage: {args.stage} | B={args.batch_size} | TileGym={args.use_tilegym}")
    print(f"Bucketing: {args.use_bucketing} | Packing: {args.use_packing}")
    if args.use_packing:
        print(f"Pack max_length: {args.pack_max_length}")
    print(f"Steps: {args.steps} (warmup: {args.warmup})")
    print()

    # Run benchmark
    torch.cuda.reset_peak_memory_stats()
    results = run_benchmark(model, dataloader, steps=args.steps, warmup=args.warmup)

    if results is None:
        print("ERROR: No steps completed!")
        return

    # Print results
    print("=" * 70)
    print("RESULTS (ground truth, measured on real data)")
    print("=" * 70)
    print(f"  Avg batch:      B={results['avg_B']:.0f}, N={results['avg_N']:.0f}")
    print(f"  Avg step time:  {results['avg_step_ms']:.1f} ms")
    print(f"  VRAM peak:      {results['vram_gb']:.2f} GB")
    print()
    print(f"  Pos/s (hw):     {results['pos_per_sec']:,.0f}  (total positions incl. padding)")
    print(f"  Real tok/s:     {results['real_tok_per_sec']:,.0f}  (non-padding tokens)")
    print(f"  Loss tok/s:     {results['loss_tok_per_sec']:,.0f}  (tokens with gradient)")
    print()
    print(f"  Padding waste:  {results['pad_pct']:.1f}%")
    print(f"  Loss ratio:     {results['loss_pct']:.1f}% (of real tokens)")
    print(f"  Overall eff:    {results['loss_tok_per_sec'] / results['pos_per_sec'] * 100:.1f}% (loss/total positions)")
    print()


if __name__ == "__main__":
    main()
