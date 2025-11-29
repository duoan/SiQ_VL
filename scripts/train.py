import argparse
import builtins
import os
import re
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from siq_vl.callbacks import (
    GenerationCallback,
    MetricsCallback,
    SmartGPUCleanCallback,
)
from siq_vl.collator import SiQ_VLDataCollator
from siq_vl.dataset import VQAIterableDataset
from siq_vl.model import SiQ_VLModel
from siq_vl.processing import SiQ_VLProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def extract_model_name(model_path: str) -> str:
    """
    Extract a short model name from a full model path.
    Keeps important model specifications like patch size and resolution.
    Examples:
        "google/siglip-so400m-patch14-384" -> "siglip-so400m-patch14-384"
        "Qwen/Qwen2.5-0.5B-Instruct" -> "qwen2.5-0.5b-instruct"
    """
    # Get the last part after '/'
    name = model_path.split("/")[-1]
    # Convert to lowercase and replace underscores with hyphens for consistency
    name = name.lower().replace("_", "-")
    # Keep all model specifications (patch sizes, resolutions, etc.)
    # These are important for distinguishing model variants
    return name


def generate_run_name(
    vision_model_path: str,
    llm_model_path: str,
    project_name: str,
    output_dir: str | None = None,
) -> str:
    """
    Generate a run name from model names, stage, and datetime.
    Format: {vision_model}_{llm_model}_{stage}_{datetime}
    """
    vision_name = extract_model_name(vision_model_path)
    llm_name = extract_model_name(llm_model_path)

    # --- Robust stage inference ---
    # Priority:
    #   1) Explicit "stageX" pattern in project name (e.g. "siq_vlm_stage1")
    #   2) Explicit "stageX" pattern in output_dir (e.g. "./checkpoints/siq_vlm_stage2/...")
    #   3) Fallback to last token of project name
    stage = None

    def _infer_stage_from_string(s: str) -> str | None:
        s = s.lower()
        # Match: "stage1", "stage_1", "stage-1", "stage 1"
        m = re.search(r"stage[_\s-]?(\d+)", s)
        if m:
            return f"stage_{m.group(1)}"
        return None

    # 1) Try from project name
    stage = _infer_stage_from_string(project_name) or stage

    # 2) Try from output_dir if still unknown
    if stage is None and output_dir is not None:
        stage = _infer_stage_from_string(output_dir)

    # 3) Fallback: use last token of project name
    if stage is None:
        stage = project_name.split("_")[-1] if "_" in project_name else project_name
    
    # Generate datetime string (format: YYYYMMDD_HHMMSS)
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine: vision_llm_stage_datetime
    run_name = f"{vision_name}_{llm_name}_{stage}_{datetime_str}"
    
    return run_name


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process.
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    builtins.print = print


def parse_args():
    parser = argparse.ArgumentParser(description="Train SiQ_VL model")
    
    # Dataset configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="HuggingFaceM4/FineVision",
        help="Path to your FineVision dataset (or the name if on HF Hub)",
    )
    parser.add_argument(
        "--sub_sets",
        type=str,
        default="coco_colors,densefusion_1m,face_emotion,google_landmarks,laion_gpt4v,sharegpt4o,sharegpt4v(coco),sharegpt4v(llava),sharegpt4v(knowledge),sharegpt4v(sam)",
        help="Comma-separated list of dataset subsets",
    )
    
    # Model configuration
    parser.add_argument(
        "--vision_model_name_or_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Path or name of the vision model",
    )
    parser.add_argument(
        "--llm_model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Path or name of the LLM model",
    )
    parser.add_argument(
        "--freeze_llm",
        action="store_true",
        default=True,
        help="Freeze the LLM model (default: True)",
    )
    parser.add_argument(
        "--no_freeze_llm",
        dest="freeze_llm",
        action="store_false",
        help="Do not freeze the LLM model",
    )
    parser.add_argument(
        "--pixel_shuffle_factor",
        type=int,
        default=None,
        help="Pixel shuffle factor for the projector. If None, will auto-calculate based on vision model.",
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/siq_vlm_run1",
        help="Directory to save checkpoints",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help=(
            "Maximum number of training steps. "
            "If <= 0, it will be auto-calculated from max_samples and the global batch size."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from the dataset (for quick testing)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=96,
        help="Number of processes for dataset loading",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    
    # Precision configuration
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bf16 precision",
    )
    parser.add_argument(
        "--no_bf16",
        dest="bf16",
        action="store_false",
        help="Disable bf16 precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use fp16 precision",
    )
    
    # Logging and saving configuration
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between saving checkpoints",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="siq_vl",
        help="Wandb project name",
    )
    parser.add_argument(
        "--gen_eval_samples",
        type=int,
        default=20,
        help="Number of fixed samples to use for generation evaluation (default: 20)",
    )
    parser.add_argument(
        "--gen_eval_interval",
        type=int,
        default=100,
        help="Evaluate generation every N steps (default: 100)",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=0.7,
        help="Temperature for generation sampling (default: 0.7)",
    )
    parser.add_argument(
        "--gen_num_beams",
        type=int,
        default=2,
        help="Number of beams for generation beam search (default: 2)",
    )
    parser.add_argument(
        "--use_distributed",
        action="store_true",
        default=False,
        help="Use distributed training (default: False, auto-detect if multiple GPUs available)",
    )
    parser.add_argument(
        "--no_distributed",
        dest="use_distributed",
        action="store_false",
        help="Disable distributed training",
    )
    
    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()
    
    # Initialize distributed training only if explicitly requested and environment is set up
    # (e.g., by accelerate launcher)
    if args.use_distributed:
        if dist.is_available() and dist.is_initialized():
            # Already initialized by accelerate launcher
            setup_for_distributed(dist.get_rank() == 0)
            print(f">>> Using distributed training (rank {dist.get_rank()}/{dist.get_world_size()})")
        elif dist.is_available():
            # Try to initialize if not already done (requires proper env vars)
            try:
                dist.init_process_group("nccl")
                setup_for_distributed(dist.get_rank() == 0)
                print(f">>> Initialized distributed training (rank {dist.get_rank()}/{dist.get_world_size()})")
            except Exception as e:
                print(f">>> Warning: Failed to initialize distributed training: {e}")
                print(">>> Continuing without distributed training.")
                args.use_distributed = False
        else:
            print(">>> Warning: Distributed training requested but PyTorch distributed not available.")
            args.use_distributed = False
    else:
        # Check if already initialized (e.g., by accelerate launcher)
        if dist.is_available() and dist.is_initialized():
            setup_for_distributed(dist.get_rank() == 0)
            args.use_distributed = True
            print(f">>> Distributed training detected (rank {dist.get_rank()}/{dist.get_world_size()})")
        else:
            print(">>> Running in single-GPU/non-distributed mode.")
            args.use_distributed = False

    # ====================================================
    # 1. Configuration
    # ====================================================
    # Path to your FineVision dataset (or the name if on HF Hub)
    DATA_PATH = args.data_path
    SUB_SETS = [subset.strip() for subset in args.sub_sets.split(",")]

    vision_model_name_or_path = args.vision_model_name_or_path
    llm_model_name_or_path = args.llm_model_name_or_path

    # Backbone identifiers (used both for output_dir and for locating
    # previous-stage checkpoints).
    vision_name = extract_model_name(vision_model_name_or_path)
    llm_name = extract_model_name(llm_model_name_or_path)
    backbone_dir = f"{vision_name}__{llm_name}"

    base_output_dir = args.output_dir
    OUTPUT_DIR = os.path.join(base_output_dir, backbone_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f">>> Using output_dir: {OUTPUT_DIR}")

    # ====================================================
    # 2. Initialize Processor & Model
    # ====================================================
    print(">>> Loading Processor & Tokenizer...")
    # Load base configs from Hugging Face
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)

    # Initialize our Custom Processor
    processor = SiQ_VLProcessor(image_processor, tokenizer)

    # Define training hyperparameters before model initialization
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    print(">>> Loading Model...")
    # Initialize our Custom Model
    # freeze_vision=True: Freezes the Vision Tower, training only Projector + LLM
    model = SiQ_VLModel(
        vision_model_path=vision_model_name_or_path,
        llm_model_path=llm_model_name_or_path,
        freeze_llm=args.freeze_llm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        pixel_shuffle_factor=args.pixel_shuffle_factor,
    )

    # ----------------------------------------------------
    # Optionally load weights from the latest Stage 1 checkpoint
    # ----------------------------------------------------
    # Convention:
    #   - Stage 1 output_root: ./checkpoints/siq_vlm_stage1
    #   - Stage 2 output_root: ./checkpoints/siq_vlm_stage2
    #   - Within each root, we create a backbone-specific subdir:
    #         {vision_name}__{llm_name}
    #   - We detect "stage 2" based on the project name containing "stage_2".
    #     In that case, we auto-load the latest Stage 1 checkpoint that matches
    #     the current backbone configuration.
    if "stage_2" in args.project.lower():
        stage1_root = base_output_dir.replace("siq_vlm_stage2", "siq_vlm_stage1")
        stage1_backbone_dir = os.path.join(stage1_root, backbone_dir)

        def _resolve_latest_stage1_checkpoint(path: str) -> str | None:
            if not os.path.isdir(path):
                return None

            # Prefer a top-level pytorch_model.bin if it exists
            top_level_bin = os.path.join(path, "pytorch_model.bin")
            if os.path.isfile(top_level_bin):
                return top_level_bin

            # Otherwise, look for checkpoint-* subdirectories
            checkpoint_dirs = [
                d
                for d in os.listdir(path)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(path, d))
            ]
            if not checkpoint_dirs:
                return None

            def _step_from_name(name: str) -> int:
                try:
                    return int(name.split("-")[-1])
                except Exception:
                    return -1

            checkpoint_dirs.sort(key=_step_from_name)
            latest_dir = checkpoint_dirs[-1]
            candidate = os.path.join(path, latest_dir, "pytorch_model.bin")
            return candidate if os.path.isfile(candidate) else None

        ckpt_file = _resolve_latest_stage1_checkpoint(stage1_backbone_dir)
        if ckpt_file is not None:
            print(f">>> Initializing model weights from Stage 1 checkpoint: {ckpt_file}")
            state_dict = torch.load(ckpt_file, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f">>> Warning: Missing keys when loading checkpoint ({len(missing_keys)}):")
                print(missing_keys[:20])
            if unexpected_keys:
                print(f">>> Warning: Unexpected keys when loading checkpoint ({len(unexpected_keys)}):")
                print(unexpected_keys[:20])
        else:
            print(
                f">>> Warning: No matching Stage 1 checkpoint found in {stage1_backbone_dir}. "
                f"Stage 2 will start from the base backbones."
            )

    # Enable Gradient Checkpointing (Critical for VRAM efficiency)
    # model.llm.gradient_checkpointing_enable()

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">>> Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # ====================================================
    # 3. Prepare Data
    # ====================================================
    print(">>> Loading Dataset...")
    # Assuming local HF dataset or JSON
    # If JSON: raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    # If saved HF dataset:

    all_raw_datasets = []

    for subset in SUB_SETS:
        raw_dataset = load_dataset(DATA_PATH, name=subset, split="train", num_proc=args.num_proc)
        all_raw_datasets.append(raw_dataset)

    # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets
    train_raw_dataset = concatenate_datasets(all_raw_datasets).shuffle(seed=0)
    
    # Limit dataset size if specified (for quick testing / controlling total samples)
    if args.max_samples is not None:
        print(f">>> Limiting dataset to {args.max_samples} samples")
        train_raw_dataset = train_raw_dataset.select(range(min(args.max_samples, len(train_raw_dataset))))

    # ----------------------------------------------------
    # Auto-calculate max_steps from max_samples & global batch
    # ----------------------------------------------------
    max_steps = args.max_steps
    if max_steps is None or max_steps <= 0:
        # Compute effective global batch size
        global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * get_world_size()
        if global_batch_size <= 0:
            global_batch_size = 1

        effective_samples = len(train_raw_dataset)
        max_steps = (effective_samples + global_batch_size - 1) // global_batch_size
        print(
            f">>> Auto-calculated max_steps={max_steps} "
            f"(effective_samples={effective_samples}, global_batch_size={global_batch_size})"
        )
    else:
        print(f">>> Using user-specified max_steps={max_steps}")

    if is_dist():
        # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_raw_dataset = train_raw_dataset.shard(
            num_shards=get_world_size(), index=get_rank()
        )

    # Wrap with our Lazy Dataset (Handles filtering and turn sampling)
    train_dataset = VQAIterableDataset(
        train_raw_dataset, processor, return_raw_data=True
    )

    # Initialize Collator
    data_collator = SiQ_VLDataCollator(tokenizer=tokenizer)

    # ====================================================
    # 4. Training Arguments
    # ====================================================
    # Generate run name from model names, stage, and datetime
    run_name = generate_run_name(
        vision_model_path=vision_model_name_or_path,
        llm_model_path=llm_model_name_or_path,
        project_name=args.project,
        output_dir=OUTPUT_DIR,
    )
    print(f">>> Generated run name: {run_name}")
    
    # Explicitly set wandb project name via environment variable
    # This ensures wandb uses the correct project name when initialized by Trainer
    os.environ["WANDB_PROJECT"] = args.project
    print(f">>> Setting WANDB_PROJECT to: {args.project}")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=run_name,
        # --- Batch Size & Speed ---
        per_device_train_batch_size=per_device_train_batch_size,  # Adjust based on VRAM (4-8 for 24GB)
        gradient_accumulation_steps=gradient_accumulation_steps,  # Effective batch size = 8 * 4 = 32
        dataloader_num_workers=args.dataloader_num_workers,
        # --- Learning Rate ---
        # Project alignment using 1e-3
        # Recommendation for full finetuning: 1e-5 to 2e-5
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_steps=max_steps,
        max_grad_norm=1.0,  # Clip gradients to prevent instability
        # --- Precision & Memory ---
        bf16=args.bf16,  # REQUIRED for Qwen (Ampere+ GPUs). Do not use fp16.
        fp16=args.fp16,
        gradient_checkpointing=True,
        # --- Logging & Metrics & Saving ---
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,  # Keep only the last 2 checkpoints to save disk space
        report_to="wandb",
        project=args.project,
        # --- CRITICAL FIX FOR LOSS REPORTING ---
        # The issue: Trainer was summing losses across gradient accumulation steps
        # instead of averaging them, causing reported loss to be 4x higher
        # include_for_metrics=["input", "loss"],  # Remove this - it causes loss aggregation issues
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
        # average_tokens_across_devices=True,  # Remove this - may cause loss calculation issues in DDP
        # --- CRITICAL FOR CUSTOM DATASET ---
        # Must be False. If True, Trainer removes columns like 'raw_question'
        # because the model signature doesn't explicitly list them.
        remove_unused_columns=False,
        label_names=["labels"],  # Explicitly tell Trainer which column is the label
        save_safetensors=False,
        # --- accelerator config ---
        accelerator_config={
            "dispatch_batches": False,
            "split_batches": False,
        },
    )

    # ====================================================
    # 5. Start Training
    # ====================================================
    # Initialize generation callback for periodic evaluation
    # Pass processor and the underlying HF dataset so the callback doesn't need the Trainer
    generation_callback = GenerationCallback(
        processor=processor,
        eval_dataset=train_raw_dataset,  # sample visual QA examples from the raw HF dataset
        eval_samples=None,  # or pass a fixed list if you want fully deterministic samples
        num_samples=args.gen_eval_samples,
        eval_interval=args.gen_eval_interval,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        do_sample=True,
        num_beams=args.gen_num_beams,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[
            MetricsCallback(),
            SmartGPUCleanCallback(),
            generation_callback,
        ],
    )

    print(">>> DEBUG: Checking Labels...")
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    for i in range(len(input_ids)):
        if labels[i] != -100:
            print(
                f"Token: {tokenizer.decode([input_ids[i]])} | Label: {labels[i].item()}"
            )
    print("input_ids:\n", input_ids, "\n")
    print("labels:\n", [label.item() for label in labels], "\n")
    print("Question:\n", batch["question"][0], "\n")
    print("Answer:\n", batch["answer"][0], "\n")

    print(">>> Start Training...")
    trainer.train()

    # ====================================================
    # 6. Save Final Model
    # ====================================================
    print(">>> Saving Final Model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(">>> Done!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
