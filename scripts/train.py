import argparse
import builtins
from datetime import datetime
import os
import random
import re
import shutil

from datasets import load_dataset
import numpy as np
import torch
import torch.distributed as dist
from torchmetrics.utilities.prints import rank_zero_info
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
from siq_vl.dataset import VQADataset
from siq_vl.model.configuration import SiQ_VLConfig
from siq_vl.model.modeling import SiQ_VLModel, get_init_vl_model_for_stage_1
from siq_vl.model.processing import SiQ_VLProcessor

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


def extract_vision_config(model_path: str) -> tuple[int, int]:
    """
    Extract image_size and patch_size from vision model path.

    Examples:
        "google/siglip2-so400m-patch16-512" -> (512, 16)
        "google/siglip2-so400m-patch14-224" -> (224, 14)
        "google/siglip-so400m-patch14-384" -> (384, 14)

    Args:
        model_path: Full model path or name (e.g., "google/siglip2-so400m-patch16-512")

    Returns:
        Tuple of (image_size, patch_size)

    Raises:
        ValueError: If image_size or patch_size cannot be extracted from the model path or config.
    """
    model_name = model_path.lower()

    # Extract patch size (e.g., patch14, patch16)
    patch_match = re.search(r"patch(\d+)", model_name)
    patch_size = int(patch_match.group(1)) if patch_match else None

    # Extract image size (usually at the end: -224, -384, -512)
    # Try to match pattern like -224, -384, -512 at the end
    size_match = re.search(r"-(\d+)$", model_name)
    image_size = int(size_match.group(1)) if size_match else None

    # If we can't extract from name, try to get from config
    if patch_size is None or image_size is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path)
            if patch_size is None:
                patch_size = getattr(config, "patch_size", None)
            if image_size is None:
                image_size = getattr(config, "image_size", None)
        except Exception:
            # If config loading fails, we'll raise an error below
            pass

    # Raise error if we still can't determine the values
    if patch_size is None:
        raise ValueError(
            f"Cannot extract patch_size from model path '{model_path}'. "
            f"Expected format: '...patch<number>...' (e.g., 'patch14', 'patch16')"
        )
    if image_size is None:
        raise ValueError(
            f"Cannot extract image_size from model path '{model_path}'. "
            f"Expected format: '...-<number>' at the end (e.g., '-224', '-384', '-512')"
        )

    return image_size, patch_size


def infer_stage_name(output_dir: str | None = None) -> str:
    """
    Infer a concise stage name like 'stage1', 'stage2'.

    Priority:
      1) STAGE env var (e.g. "1" or "2")
      2) output_dir path containing "stage1"/"stage_1"/"stage-1" etc.
      3) fallback to "stage1"
    """

    # 1) Environment variable from shell/launcher
    env_stage = os.environ.get("STAGE")
    if env_stage is not None:
        env_stage = env_stage.strip()
        if env_stage.isdigit():
            return f"stage{env_stage}"

    def _infer_from_string(s: str) -> str | None:
        s = s.lower()
        m = re.search(r"stage[_\\s-]?(\\d+)", s)
        if m:
            return f"stage{m.group(1)}"
        return None

    stage = None
    if output_dir is not None:
        stage = _infer_from_string(output_dir)

    # Fallback to "stage1" if we can't infer anything sensible
    return stage or "stage1"


def generate_run_name(
    vision_model_path: str,
    llm_model_path: str,
    output_dir: str | None = None,
) -> str:
    """
    Generate a run name from model names, stage, and datetime.
    Format: {vision_model}_{llm_model}_{stage}_{datetime}
    """
    vision_name = extract_model_name(vision_model_path)
    llm_name = extract_model_name(llm_model_path)

    return f"{vision_name}_{llm_name}_{infer_stage_name(output_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f">>> Set random seed to {seed} for reproducibility")


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
    parser.add_argument(
        "--sub_sets_weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated sampling weights aligned with sub_sets. "
            "E.g. for 'coco_colors,sharegpt4v(coco),laion_gpt4v,face_emotion' "
            "you can pass '4,4,1,1' to downweight laion_gpt4v."
        ),
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
        default=1,
        help="Pixel shuffle factor for the projector (default: 1, no shuffling).",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help=(
            "Root directory for checkpoints. The final path is computed as "
            "'{output_dir}/siq-vl_{vision_backbone}_{llm_backbone}/{stage}'."
        ),
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
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between evaluation",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=2,
        help="Maximum number of samples to use for evaluation (None = use all). "
        "Set this to limit eval time if eval dataset is large.",
    )
    parser.add_argument(
        "--gen_samples",
        type=int,
        default=20,
        help="Number of fixed samples to use for generation evaluation (default: 20)",
    )
    parser.add_argument(
        "--gen_steps",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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
    # Hugging Face Hub configuration
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="If set, push the final checkpoint to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help=(
            "Optional explicit Hub model id (e.g. 'org/siq_vl-...'). "
            "If not provided, a name of the form "
            "'siq_vl-{vision}__{llm}-{stage}' will be used."
        ),
    )

    return parser.parse_args()


def train(args=None):
    if args is None:
        args = parse_args()

    # Set random seeds for reproducibility
    seed_everything(args.seed)

    # Initialize distributed training only if explicitly requested and environment is set up
    # (e.g., by accelerate launcher)
    if args.use_distributed:
        if dist.is_available() and dist.is_initialized():
            # Already initialized by accelerate launcher
            setup_for_distributed(dist.get_rank() == 0)
            rank_zero_info(f">>> Using distributed training (rank {dist.get_rank()}/{dist.get_world_size()})")
        elif dist.is_available():
            # Try to initialize if not already done (requires proper env vars)
            try:
                dist.init_process_group("nccl")
                setup_for_distributed(dist.get_rank() == 0)
                rank_zero_info(f">>> Initialized distributed training (rank {dist.get_rank()}/{dist.get_world_size()})")
            except Exception as e:
                rank_zero_info(f">>> Warning: Failed to initialize distributed training: {e}")
                rank_zero_info(">>> Continuing without distributed training.")
                args.use_distributed = False
        else:
            rank_zero_info(">>> Warning: Distributed training requested but PyTorch distributed not available.")
            args.use_distributed = False
    else:
        # Check if already initialized (e.g., by accelerate launcher)
        if dist.is_available() and dist.is_initialized():
            setup_for_distributed(dist.get_rank() == 0)
            args.use_distributed = True
            rank_zero_info(f">>> Distributed training detected (rank {dist.get_rank()}/{dist.get_world_size()})")
        else:
            rank_zero_info(">>> Running in single-GPU/non-distributed mode.")
            args.use_distributed = False

    # ====================================================
    # 1. Configuration
    # ====================================================
    # Path to your FineVision dataset (or the name if on HF Hub)
    DATA_PATH = args.data_path
    SUB_SETS = [subset.strip() for subset in args.sub_sets.split(",")]
    if args.sub_sets_weights is not None:
        weight_strs = [w.strip() for w in args.sub_sets_weights.split(",")]
        if len(weight_strs) != len(SUB_SETS):
            raise ValueError(
                f"sub_sets_weights length ({len(weight_strs)}) "
                f"must match sub_sets length ({len(SUB_SETS)}). "
                f"sub_sets={SUB_SETS}, sub_sets_weights={weight_strs}"
            )
        sub_sets_weights = [float(w) for w in weight_strs]
    else:
        sub_sets_weights = [1.0] * len(SUB_SETS)

    rank_zero_info(">>> Using subsets and weights:")
    for name, w in zip(SUB_SETS, sub_sets_weights, strict=False):
        rank_zero_info(f"    - {name}: weight={w}")

    vision_model_name_or_path = args.vision_model_name_or_path
    llm_model_name_or_path = args.llm_model_name_or_path

    # Backbone identifiers (used both for output_dir and for naming).
    vision_name = extract_model_name(vision_model_name_or_path)
    llm_name = extract_model_name(llm_model_name_or_path)

    # Infer stage name once so we can use it consistently for paths and Hub IDs
    stage_name = infer_stage_name(args.output_dir)

    # New local checkpoint layout:
    #   {output_dir}/siq-vl_{vision_backbone}_{llm_backbone}/{stage}
    # e.g.:
    #   ./checkpoints/siq-vl_siglip2-base-patch16-224_qwen2.5-0.5b-instruct/stage1
    base_output_dir = args.output_dir
    run_root = os.path.join(base_output_dir, f"siq-vl_{vision_name}_{llm_name}")
    OUTPUT_DIR = os.path.join(run_root, stage_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rank_zero_info(f">>> Using output_dir: {OUTPUT_DIR}")

    # ====================================================
    # 2. Initialize Processor & Model
    # ====================================================
    rank_zero_info(">>> Loading Processor & Tokenizer...")
    # Load base configs from Hugging Face
    image_processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)

    # Extract image_size and patch_size from vision model path
    image_size, patch_size = extract_vision_config(vision_model_name_or_path)
    rank_zero_info(f">>> Extracted vision config: image_size={image_size}, patch_size={patch_size}")

    # Initialize our Custom Processor
    processor = SiQ_VLProcessor(
        image_processor,
        tokenizer,
        image_size=image_size,
        patch_size=patch_size,
        pixel_shuffle_factor=args.pixel_shuffle_factor,
    )

    # Define training hyperparameters before model initialization
    per_device_train_batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    rank_zero_info(">>> Loading Model...")
    # Initialize our Custom Model using config
    config = SiQ_VLConfig(
        pretrained_vision_model_path=vision_model_name_or_path,
        pretrained_language_model_path=llm_model_name_or_path,
        freeze_language_model=args.freeze_llm,
        vision_pixel_shuffle_factor=args.pixel_shuffle_factor,
    )

    if stage_name == "stage1":
        vl_model = get_init_vl_model_for_stage_1(config)
    elif stage_name == "stage2":
        vl_model = SiQ_VLModel.from_pretrained(
            config.pretrained_language_model_path, config=config, trust_remote_code=True
        )
        if args.freeze_llm:
            for param in vl_model.model.parameters():
                param.requires_grad_(False)
            vl_model.model.eval()
        else:
            for param in vl_model.model.parameters():
                param.requires_grad_(True)
            vl_model.model.train()

    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in vl_model.parameters() if p.requires_grad)
    rank_zero_info(f">>> Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # ====================================================
    # 3. Prepare Data
    # ====================================================
    rank_zero_info(">>> Loading Dataset...")
    # Assuming local HF dataset or JSON
    # If JSON: raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    # If saved HF dataset:

    all_raw_datasets = []
    all_raw_datasets: list = []
    all_weights: list[float] = []

    for subset, weight in zip(SUB_SETS, sub_sets_weights, strict=False):
        rank_zero_info(f">>> Loading subset '{subset}' (weight={weight})...")
        raw_dataset = load_dataset(DATA_PATH, name=subset, split="train", num_proc=args.num_proc)
        all_raw_datasets.append(raw_dataset)
        all_weights.append(weight)

    from datasets.combine import interleave_datasets

    total_weight = sum(all_weights)
    probabilities = [weight / total_weight for weight in all_weights]
    rank_zero_info(">>> Building weighted mixture dataset with probabilities:")
    for name, probability in zip(SUB_SETS, probabilities, strict=False):
        rank_zero_info(f"    - {name}: probability={probability:.4f}")

    concat_raw_dataset = interleave_datasets(
        all_raw_datasets,
        probabilities=probabilities,
        seed=args.seed,
    )

    # Limit dataset size if specified (for quick testing / controlling total samples)
    if args.max_samples is not None:
        rank_zero_info(f">>> Limiting dataset to {args.max_samples} samples")
        concat_raw_dataset = concat_raw_dataset.select(range(min(args.max_samples, len(concat_raw_dataset))))

    splits = concat_raw_dataset.train_test_split(test_size=args.max_eval_samples, shuffle=True)
    train_raw_dataset = splits["train"]
    eval_raw_dataset = splits["test"]

    # Use VQADataset (standard Dataset) instead of VQAIterableDataset
    # This allows Trainer's DataLoader to automatically use DistributedSampler
    # No manual sharding needed!
    rank_zero_info(">>> Creating VQADataset (standard Dataset with DistributedSampler support)...")
    train_dataset = VQADataset(train_raw_dataset)
    eval_dataset = VQADataset(eval_raw_dataset)

    rank_zero_info(f">>> DEBUG: Train Dataset: {train_dataset}")
    rank_zero_info(f">>> DEBUG: Train Dataset Length: {len(train_dataset)}")
    rank_zero_info(f">>> DEBUG: Eval Dataset: {eval_dataset}")
    rank_zero_info(f">>> DEBUG: Eval Dataset Length: {len(eval_dataset)}")
    if len(train_dataset) > 0:
        rank_zero_info(f">>> DEBUG: Train Dataset Sample: {train_dataset[0]}")
    if len(eval_dataset) > 0:
        rank_zero_info(f">>> DEBUG: Eval Dataset Sample: {eval_dataset[0]}")

    # ----------------------------------------------------
    # Auto-calculate max_steps from max_samples & global batch
    # ----------------------------------------------------
    max_steps = args.max_steps
    if max_steps is None or max_steps <= 0:
        # Compute effective global batch size
        global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * get_world_size()
        if global_batch_size <= 0:
            global_batch_size = 1

        # Use the original concat_raw_dataset length for max_steps calculation
        # (before sharding, so we get the total dataset size)
        effective_samples = len(concat_raw_dataset)
        max_steps = (effective_samples + global_batch_size - 1) // global_batch_size
        rank_zero_info(
            f">>> Auto-calculated max_steps={max_steps} (effective_samples={effective_samples}, global_batch_size={global_batch_size})"
        )
    else:
        rank_zero_info(f">>> Using user-specified max_steps={max_steps}")

    # Initialize Collator
    data_collator = SiQ_VLDataCollator(processor=processor, return_raw_data=True)

    # ====================================================
    # 4. Training Arguments
    # ====================================================
    # Generate run name from model names, stage, and datetime
    run_name = generate_run_name(
        vision_model_path=vision_model_name_or_path,
        llm_model_path=llm_model_name_or_path,
        output_dir=OUTPUT_DIR,
    )
    rank_zero_info(f">>> Generated run name: {run_name}")

    # Explicitly set wandb project name via environment variable
    # This ensures wandb uses the correct project name when initialized by Trainer
    os.environ["WANDB_PROJECT"] = "siq-vl"
    rank_zero_info(">>> Setting WANDB_PROJECT to: siq-vl")

    # Prepare Hub model ID if push_to_hub is enabled
    hub_model_id = None
    if getattr(args, "push_to_hub", False):
        # Derive stage name like 'stage1' / 'stage2'
        stage_name = infer_stage_name(base_output_dir)
        # Default repo id (single underscores between logical segments),
        # with a "siq-vl" prefix:
        #   siq-vl_{vision_backbone}_{llm_backbone}_{stage}
        default_repo_name = f"siq-vl_{vision_name}_{llm_name}_{stage_name}"
        hub_model_id = args.hub_model_id or default_repo_name
        rank_zero_info(f">>> Will push to Hub model id: {hub_model_id}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=run_name,
        # --- Batch Size & Speed ---
        per_device_train_batch_size=per_device_train_batch_size,  # Adjust based on VRAM (4-8 for 24GB)
        gradient_accumulation_steps=gradient_accumulation_steps,  # Effective batch size = 8 * 4 = 32
        dataloader_num_workers=args.dataloader_num_workers,
        # --- Evaluation ---
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        # Use same batch size for eval as for training
        per_device_eval_batch_size=per_device_train_batch_size * gradient_accumulation_steps,
        eval_accumulation_steps=1,  # Accumulate eval results in one go
        # Note: max_eval_samples is handled when creating eval_dataset above
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
        project="siq-vl",
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
        # --- Hugging Face Hub ---
        push_to_hub=getattr(args, "push_to_hub", False),
        hub_model_id=hub_model_id,
        hub_strategy="end",  # Only push at the end of training
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
        eval_dataset=eval_dataset,
        num_samples=args.gen_samples,
        eval_interval=args.gen_steps,
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        do_sample=True,
        num_beams=args.gen_num_beams,
    )

    callbacks = [generation_callback, MetricsCallback()]

    if torch.cuda.is_available():
        callbacks.append(SmartGPUCleanCallback())

    trainer = Trainer(
        model=vl_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=callbacks,
    )

    rank_zero_info(">>> DEBUG: Checking Labels...")
    batch = next(iter(trainer.get_train_dataloader()))
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    for i in range(len(input_ids)):
        if labels[i] != -100:
            rank_zero_info(f"Token: {tokenizer.decode([input_ids[i]])} | Label: {labels[i].item()}")
    rank_zero_info(f"input_ids:\n {input_ids} \n")
    rank_zero_info(f"labels:\n {[label.item() for label in labels]} \n")
    rank_zero_info(f"Question:\n {batch['questions'][0]} \n")
    rank_zero_info(f"Answer:\n {batch['answers'][0]} \n")

    rank_zero_info(">>> Start Training...")
    trainer.train()
    # Trainer automatically saves the final model to OUTPUT_DIR and pushes to Hub if enabled

    # ====================================================
    # 6. Save custom model code files for AutoModel.from_pretrained()
    # ====================================================
    # Copy custom modeling.py and configuration.py to output directory
    # so that others can use AutoModel.from_pretrained() to load the model
    rank_zero_info(">>> Saving custom model code files...")
    model_code_dir = os.path.join(OUTPUT_DIR, "siq_vl", "model")
    os.makedirs(model_code_dir, exist_ok=True)

    # Get the source directory of siq_vl.model
    import siq_vl.model as model_module

    source_model_dir = os.path.dirname(model_module.__file__)

    # Copy necessary files
    files_to_copy = ["modeling.py", "configuration.py", "processing.py"]
    for file_name in files_to_copy:
        src_file = os.path.join(source_model_dir, file_name)
        dst_file = os.path.join(model_code_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            rank_zero_info(f">>> Copied {file_name} to {dst_file}")
        else:
            rank_zero_info(f">>> Warning: {src_file} not found, skipping...")

    # Create __init__.py if it doesn't exist
    init_file = os.path.join(model_code_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")

    # Also create parent __init__.py
    siq_vl_dir = os.path.join(OUTPUT_DIR, "siq_vl")
    os.makedirs(siq_vl_dir, exist_ok=True)
    parent_init_file = os.path.join(siq_vl_dir, "__init__.py")
    if not os.path.exists(parent_init_file):
        with open(parent_init_file, "w") as f:
            f.write("")

    rank_zero_info(">>> Done!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
