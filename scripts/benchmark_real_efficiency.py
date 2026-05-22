"""Ground-truth efficiency measurement on REAL data.

Measures ALL metrics precisely by running actual training steps with the real
data pipeline. Supports both model scales and all optimization toggles for
systematic apple-to-apple comparison.

Metrics:
  - Pos/s (hw): Total positions processed per second (B * N_padded / time)
  - Real tok/s: Non-padding tokens per second (attention_mask.sum() / time)
  - Loss tok/s: Tokens contributing to loss per second ((labels != -100).sum() / time)
  - Pad%: Padding waste = 1 - Real/Pos
  - Loss%: Fraction of real tokens that produce loss

Usage:
    # Small model baseline
    python scripts/benchmark_real_efficiency.py --model small --batch_size 4

    # Small model with all optimizations
    python scripts/benchmark_real_efficiency.py --model small --batch_size 64 --use_bucketing --use_tilegym --pad_to_multiple_of 64

    # Large model
    python scripts/benchmark_real_efficiency.py --model large --batch_size 16 --use_bucketing

    # Stage 2 (unfrozen LLM)
    python scripts/benchmark_real_efficiency.py --model small --stage 2 --batch_size 32 --use_tilegym
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

from siq_vl.collator import PackingCollator, SiQ_VLDataCollator
from siq_vl.model.modeling import get_stage1_model_and_processor

MODEL_CONFIGS = {
    "small": {
        "vision": "google/siglip2-base-patch16-224",
        "text": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "SigLIP2-base-224 + Qwen2.5-0.5B",
        "pixel_shuffle_factor": 2,  # 224/16=14, 14/2=7 ✓
    },
    "large": {
        "vision": "google/siglip2-so400m-patch14-384",
        "text": "Qwen/Qwen2.5-1.5B-Instruct",
        "label": "SigLIP2-so400m-384 + Qwen2.5-1.5B",
        "pixel_shuffle_factor": 3,  # 384/14=27, 27/3=9 ✓
    },
}


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

    all_stats = []
    step_times = []

    step_count = 0
    for batch in dataloader:
        if step_count >= warmup + steps:
            break

        if batch is None:
            continue

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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

    import numpy as np

    total_pos = sum(s["total_positions"] for s in all_stats)
    total_real = sum(s["real_tokens"] for s in all_stats)
    total_loss = sum(s["loss_tokens"] for s in all_stats)
    total_time = sum(step_times)

    return {
        "steps": len(all_stats),
        "avg_B": float(np.mean([s["B"] for s in all_stats])),
        "avg_N": float(np.mean([s["N"] for s in all_stats])),
        "avg_step_ms": float(np.mean([s["step_ms"] for s in all_stats])),
        "pos_per_sec": total_pos / total_time,
        "real_tok_per_sec": total_real / total_time,
        "loss_tok_per_sec": total_loss / total_time,
        "pad_pct": (1 - total_real / total_pos) * 100,
        "loss_pct": total_loss / total_real * 100 if total_real > 0 else 0,
        "vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main():
    parser = argparse.ArgumentParser(description="Ground-truth training efficiency benchmark")
    # Model selection
    parser.add_argument("--model", type=str, default="small", choices=["small", "large"],
                        help="Model scale: small (0.5B) or large (1.5B)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1 (frozen LLM) or 2 (unfrozen)")
    # Batch & data
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_bucketing", action="store_true")
    parser.add_argument("--use_packing", action="store_true")
    parser.add_argument("--pack_max_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)
    # Kernel optimization
    parser.add_argument("--use_tilegym", action="store_true", help="TileGym full stack (cuTile)")
    parser.add_argument("--use_liger", action="store_true", help="Liger-Kernel fused ops")
    parser.add_argument("--no_fused_ce", action="store_true",
                        help="Disable Liger fused_linear_cross_entropy (faster on high-VRAM GPUs)")
    parser.add_argument("--use_compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--force_fp32", action="store_true", help="Reproduce FP32 baseline bug")
    # Stage 2 options
    parser.add_argument("--use_grad_ckpt", action="store_true", help="Gradient checkpointing (Stage 2)")
    parser.add_argument("--use_lora", action="store_true", help="LoRA fine-tuning (Stage 2)")
    parser.add_argument("--lora_r", type=int, default=16)
    # Benchmark params
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data_path", type=str, default="lmms-lab/LLaVA-OneVision-Data")
    parser.add_argument("--sub_set", type=str, default="sharegpt4v(coco)")
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    set_seed(args.seed)

    config = MODEL_CONFIGS[args.model]
    print(f"{'='*70}")
    print(f"  Model: {config['label']}")
    print(f"  Stage: {args.stage} | B={args.batch_size}")
    print(f"  Kernels: tilegym={args.use_tilegym}, liger={args.use_liger}, compile={args.use_compile}")
    print(f"  Data: bucketing={args.use_bucketing}, packing={args.use_packing}")
    if args.force_fp32:
        print(f"  ** FORCE FP32 (reproducing baseline bug) **")
    print(f"{'='*70}\n")

    # Apply kernel patches BEFORE model loading
    if args.use_tilegym:
        from tilegym.transformers import apply_tilegym_kernel_to_qwen2
        from siq_vl.kernels.fused_linear_ce import patch_qwen2_fused_linear_ce
        apply_tilegym_kernel_to_qwen2(use_cutile=True)
        patch_qwen2_fused_linear_ce()
        print("[kernel] TileGym full stack applied")
    elif args.use_liger:
        print("[kernel] Liger-Kernel will be applied via model loader")

    # Load model
    dtype = torch.float32 if args.force_fp32 else torch.bfloat16

    model, processor = get_stage1_model_and_processor(
        pretrained_vision_model_path=config["vision"],
        pretrained_text_model_path=config["text"],
        vision_pixel_shuffle_factor=config["pixel_shuffle_factor"],
        use_tilegym=args.use_tilegym,
        use_liger=args.use_liger,
        use_fused_ce=not args.no_fused_ce,
    )

    if args.force_fp32:
        model = model.to(device="cuda", dtype=torch.float32)
    else:
        model = model.to(device="cuda", dtype=torch.bfloat16)

    # Stage 2: unfreeze text model
    if args.stage == 2:
        for param in model.text_model.parameters():
            param.requires_grad = True
        print("[stage 2] Text model unfrozen")

        if args.use_grad_ckpt:
            model.text_model.gradient_checkpointing_enable()
            print("[stage 2] Gradient checkpointing enabled")

        if args.use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            )
            model.text_model = get_peft_model(model.text_model, lora_config)
            print(f"[stage 2] LoRA r={args.lora_r} applied")

    if args.use_compile and not args.use_tilegym:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
        print("[kernel] torch.compile applied")

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable: {trainable/1e6:.1f}M / Total: {total/1e6:.1f}M")

    # Load dataset
    print(f"\nLoading dataset: {args.data_path} / {args.sub_set}")
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

    def safe_collate(features):
        features = [f for f in features if f is not None]
        if not features:
            return None
        return collator(features)

    # Setup sampler
    if args.use_bucketing:
        lengths = dataset.lengths
        sorted_indices = sorted(range(len(dataset)), key=lambda i: lengths[i])
        mid = len(sorted_indices) // 3
        needed = args.batch_size * (args.steps + args.warmup) * 2
        sampler_indices = sorted_indices[mid: mid + needed]

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
            sampler=IndexSampler(sampler_indices),
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

    # Run benchmark
    print(f"\nRunning {args.steps} steps (warmup={args.warmup})...")
    torch.cuda.reset_peak_memory_stats()
    results = run_benchmark(model, dataloader, steps=args.steps, warmup=args.warmup)

    if results is None:
        print("ERROR: No steps completed!")
        return

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS (ground truth, measured on real data)")
    print(f"{'='*70}")
    print(f"  Model:          {config['label']}")
    print(f"  Stage:          {args.stage}")
    print(f"  Avg batch:      B={results['avg_B']:.0f}, N={results['avg_N']:.0f}")
    print(f"  Avg step time:  {results['avg_step_ms']:.1f} ms")
    print(f"  VRAM peak:      {results['vram_gb']:.2f} GB")
    print()
    print(f"  Pos/s (hw):     {results['pos_per_sec']:,.0f}")
    print(f"  Real tok/s:     {results['real_tok_per_sec']:,.0f}")
    print(f"  Loss tok/s:     {results['loss_tok_per_sec']:,.0f}")
    print()
    print(f"  Padding waste:  {results['pad_pct']:.1f}%")
    print(f"  Loss ratio:     {results['loss_pct']:.1f}%")
    print()

    # Save to JSON
    if args.output_json:
        output = {
            "model": args.model,
            "model_label": config["label"],
            "stage": args.stage,
            "config": {
                "batch_size": args.batch_size,
                "use_bucketing": args.use_bucketing,
                "use_packing": args.use_packing,
                "pack_max_length": args.pack_max_length,
                "pad_to_multiple_of": args.pad_to_multiple_of,
                "use_tilegym": args.use_tilegym,
                "use_liger": args.use_liger,
                "no_fused_ce": args.no_fused_ce,
                "use_compile": args.use_compile,
                "force_fp32": args.force_fp32,
                "use_grad_ckpt": args.use_grad_ckpt,
                "use_lora": args.use_lora,
            },
            "results": results,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
