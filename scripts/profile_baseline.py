"""
Baseline profiling script for SiQ-VL training.

Runs a short training loop (N steps) with torch.profiler enabled,
producing Chrome trace files and a summary table. Also records peak VRAM,
step times, and tokens/s for the iteration log.

Usage:
    python scripts/profile_baseline.py [--args...]

For nsys profiling (wraps this script externally):
    bash scripts/profile_baseline.sh
"""

import argparse
import json
import os
import time

import torch
import torch.cuda
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torchmetrics.utilities.prints import rank_zero_info
from transformers import Trainer, TrainingArguments, set_seed

from siq_vl.collator import CachedVisionDataCollator, SiQ_VLDataCollator
from siq_vl.dataset import CachedVQADataset, VQADataset
from siq_vl.model.modeling import get_stage1_model_and_processor, get_stage2_model_and_processor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile SiQ-VL training baseline")

    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--vision_model_name_or_path",
        type=str,
        default="google/siglip2-so400m-patch16-512",
    )
    parser.add_argument(
        "--text_model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--pixel_shuffle_factor", type=int, default=4)
    parser.add_argument("--stage_1_checkpoint_path", type=str, default=None)

    parser.add_argument("--data_path", type=str, default="HuggingFaceM4/FineVision")
    parser.add_argument("--sub_sets", type=str, default="sharegpt4v(coco)")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--num_proc", type=int, default=8)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--cached_features_dir", type=str, default=None,
                        help="Directory with pre-extracted vision features (skips vision encoder)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (trades memory for speed)")

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="docs/traces",
        help="Directory to save profiler traces and summary",
    )
    parser.add_argument(
        "--trace_name",
        type=str,
        default="iter_0_baseline",
        help="Base name for trace files",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    trace_path = os.path.join(args.output_dir, f"{args.trace_name}.json")
    summary_path = os.path.join(args.output_dir, f"{args.trace_name}_summary.json")

    rank_zero_info("=" * 80)
    rank_zero_info("SiQ-VL BASELINE PROFILING")
    rank_zero_info("=" * 80)
    rank_zero_info(f"Stage: {args.stage}")
    rank_zero_info(f"Vision: {args.vision_model_name_or_path}")
    rank_zero_info(f"LLM: {args.text_model_name_or_path}")
    rank_zero_info(f"pixel_shuffle_factor: {args.pixel_shuffle_factor}")
    rank_zero_info(f"Batch size: {args.per_device_train_batch_size}")
    rank_zero_info(f"Grad accum: {args.gradient_accumulation_steps}")
    rank_zero_info(f"Warmup steps: {args.warmup_steps}")
    rank_zero_info(f"Profile steps: {args.profile_steps}")
    rank_zero_info(f"Trace output: {trace_path}")
    rank_zero_info("=" * 80)

    # ================================================================
    # 1. Load model & processor
    # ================================================================
    rank_zero_info(">>> Loading model and processor...")

    if args.stage == 1:
        model, processor = get_stage1_model_and_processor(
            pretrained_vision_model_path=args.vision_model_name_or_path,
            pretrained_text_model_path=args.text_model_name_or_path,
            vision_pixel_shuffle_factor=args.pixel_shuffle_factor,
        )
    else:
        checkpoint = args.stage_1_checkpoint_path
        if checkpoint is None:
            raise ValueError("--stage_1_checkpoint_path required for stage 2 profiling")
        model, processor = get_stage2_model_and_processor(
            stage_1_checkpoint_path=checkpoint,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # When using cached features, offload the vision model from GPU to save VRAM
    if args.cached_features_dir:
        model.vision_model.to("cpu")
        torch.cuda.empty_cache()
        rank_zero_info(">>> Vision model offloaded to CPU (using cached features)")

    rank_zero_info(f">>> Model on {device}")

    # ================================================================
    # 2. Load dataset
    # ================================================================
    rank_zero_info(">>> Loading dataset...")
    from datasets import load_dataset

    sub_sets = [s.strip() for s in args.sub_sets.split(",")]
    raw_datasets = []
    for subset in sub_sets:
        ds = load_dataset(args.data_path, name=subset, split="train", num_proc=args.num_proc)
        raw_datasets.append(ds)

    if len(raw_datasets) > 1:
        from datasets.combine import interleave_datasets
        raw_dataset = interleave_datasets(raw_datasets, seed=args.seed, stopping_strategy="first_exhausted")
    else:
        raw_dataset = raw_datasets[0]

    if args.max_samples:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    if args.cached_features_dir:
        train_dataset = CachedVQADataset(raw_dataset, cache_dir=args.cached_features_dir)
        data_collator = CachedVisionDataCollator(processor=processor, max_length=args.max_length)
        rank_zero_info(f">>> Using CACHED vision features from: {args.cached_features_dir}")
    else:
        train_dataset = VQADataset(raw_dataset)
        data_collator = SiQ_VLDataCollator(processor=processor, max_length=args.max_length)
    rank_zero_info(f">>> Dataset size: {len(train_dataset)}")

    total_steps = args.warmup_steps + args.profile_steps
    training_args = TrainingArguments(
        output_dir="/tmp/siq_vl_profile",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=total_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=True,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ================================================================
    # 4. Warmup run (no profiler, stabilize CUDA context)
    # ================================================================
    rank_zero_info(f">>> Running {args.warmup_steps} warmup steps...")
    torch.cuda.reset_peak_memory_stats()

    train_dataloader = trainer.get_train_dataloader()
    model.train()
    optimizer = trainer.create_optimizer()
    lr_scheduler = trainer.create_scheduler(num_training_steps=total_steps, optimizer=optimizer)

    model_input_keys = [
        "input_ids", "pixel_values", "vision_features", "attention_mask", "labels", "num_image_tokens"
    ]

    step_times_warmup = []
    data_iter = iter(train_dataloader)

    for step in range(args.warmup_steps):
        batch = next(data_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with record_function("forward"):
            outputs = model(**{k: v for k, v in batch.items() if k in model_input_keys})
            loss = outputs.loss / args.gradient_accumulation_steps

        with record_function("backward"):
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            with record_function("optimizer_step"):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times_warmup.append((t1 - t0) * 1000)

    warmup_vram = torch.cuda.max_memory_allocated() / (1024**3)
    rank_zero_info(f">>> Warmup done. Peak VRAM after warmup: {warmup_vram:.2f} GB")
    rank_zero_info(f">>> Warmup step times (ms): {[f'{t:.1f}' for t in step_times_warmup]}")

    # ================================================================
    # 5. Profiled run (torch.profiler)
    # ================================================================
    rank_zero_info(f">>> Running {args.profile_steps} profiled steps with torch.profiler...")
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    step_tokens = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        schedule=schedule(
            wait=0,
            warmup=2,
            active=args.profile_steps - 2,
            repeat=1,
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
    ) as prof:
        for step in range(args.profile_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Count tokens in this step
            if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
                n_tokens = int((batch["attention_mask"].sum()).item()) if "attention_mask" in batch else batch["input_ids"].numel()
            else:
                n_tokens = 0

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with record_function("forward"):
                outputs = model(**{k: v for k, v in batch.items() if k in model_input_keys})
                loss = outputs.loss / args.gradient_accumulation_steps

            with record_function("backward"):
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                with record_function("optimizer_step"):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            step_times.append(elapsed_ms)
            step_tokens.append(n_tokens)

            prof.step()

            if (step + 1) % 5 == 0:
                rank_zero_info(
                    f"  Step {step + 1}/{args.profile_steps}: "
                    f"{elapsed_ms:.1f} ms, loss={loss.item() * args.gradient_accumulation_steps:.4f}, "
                    f"tokens={n_tokens}"
                )

    # ================================================================
    # 6. Collect and print results
    # ================================================================
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    allocated_vram_gb = torch.cuda.memory_allocated() / (1024**3)
    reserved_vram_gb = torch.cuda.memory_reserved() / (1024**3)

    avg_step_time = sum(step_times) / len(step_times)
    p50_step_time = sorted(step_times)[len(step_times) // 2]
    p95_step_time = sorted(step_times)[int(len(step_times) * 0.95)]
    total_tokens = sum(step_tokens)
    tokens_per_sec = total_tokens / (sum(step_times) / 1000)

    rank_zero_info("\n" + "=" * 80)
    rank_zero_info("BASELINE PROFILING RESULTS")
    rank_zero_info("=" * 80)
    rank_zero_info(f"Steps profiled:          {args.profile_steps}")
    rank_zero_info(f"Avg step time (ms):      {avg_step_time:.1f}")
    rank_zero_info(f"P50 step time (ms):      {p50_step_time:.1f}")
    rank_zero_info(f"P95 step time (ms):      {p95_step_time:.1f}")
    rank_zero_info(f"Min step time (ms):      {min(step_times):.1f}")
    rank_zero_info(f"Max step time (ms):      {max(step_times):.1f}")
    rank_zero_info(f"Tokens / sec:            {tokens_per_sec:.0f}")
    rank_zero_info(f"Peak VRAM (GB):          {peak_vram_gb:.2f}")
    rank_zero_info(f"Allocated VRAM (GB):     {allocated_vram_gb:.2f}")
    rank_zero_info(f"Reserved VRAM (GB):      {reserved_vram_gb:.2f}")
    rank_zero_info(f"Total tokens processed:  {total_tokens}")
    rank_zero_info("=" * 80)

    # Print torch.profiler key averages
    rank_zero_info("\n>>> Top 30 CUDA operations by total time:")
    rank_zero_info(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    )

    rank_zero_info("\n>>> Top 20 operations by CUDA memory:")
    rank_zero_info(
        prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20)
    )

    rank_zero_info("\n>>> Top 20 operations by self CUDA time:")
    rank_zero_info(
        prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    )

    # ================================================================
    # 7. Save summary JSON (for iteration log)
    # ================================================================
    summary = {
        "config": {
            "stage": args.stage,
            "vision_model": args.vision_model_name_or_path,
            "text_model": args.text_model_name_or_path,
            "pixel_shuffle_factor": args.pixel_shuffle_factor,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "dataloader_num_workers": args.dataloader_num_workers,
            "bf16": True,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "results": {
            "warmup_steps": args.warmup_steps,
            "profile_steps": args.profile_steps,
            "avg_step_time_ms": round(avg_step_time, 1),
            "p50_step_time_ms": round(p50_step_time, 1),
            "p95_step_time_ms": round(p95_step_time, 1),
            "min_step_time_ms": round(min(step_times), 1),
            "max_step_time_ms": round(max(step_times), 1),
            "tokens_per_sec": round(tokens_per_sec, 0),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "allocated_vram_gb": round(allocated_vram_gb, 2),
            "reserved_vram_gb": round(reserved_vram_gb, 2),
            "total_tokens": total_tokens,
        },
        "step_times_ms": [round(t, 1) for t in step_times],
        "step_tokens": step_tokens,
        "trace_file": trace_path,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    rank_zero_info(f"\n>>> Chrome trace saved to: {trace_path}")
    rank_zero_info(f">>> Summary JSON saved to: {summary_path}")
    rank_zero_info(">>> Open trace in chrome://tracing or https://ui.perfetto.dev")
    rank_zero_info(">>> Done!")


if __name__ == "__main__":
    main()
