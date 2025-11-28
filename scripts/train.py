import builtins
import os

import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from siq_vl.callbacks import MetricsCallback, SmartGPUCleanCallback
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


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process.
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    builtins.print = print


def train():
    dist.init_process_group("nccl")
    setup_for_distributed(dist.get_rank() == 0)

    # ====================================================
    # 1. Configuration
    # ====================================================
    # Path to your FineVision dataset (or the name if on HF Hub)
    DATA_PATH = "HuggingFaceM4/FineVision"
    SUB_SETS = [
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
        # "allava_laion",
        # "cocoqa",
        # "LLaVA_Instruct_150K",
        # "llavar_gpt4_20k",
    ]

    vision_model_name_or_path = "google/siglip-so400m-patch14-384"
    llm_model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"

    # Directory to save checkpoints
    OUTPUT_DIR = "./checkpoints/siq_vlm_run1"

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
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 4
    max_steps = 1000

    print(">>> Loading Model...")
    # Initialize our Custom Model
    # freeze_vision=True: Freezes the Vision Tower, training only Projector + LLM
    model = SiQ_VLModel(
        vision_model_path=vision_model_name_or_path,
        llm_model_path=llm_model_name_or_path,
        freeze_llm=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        raw_dataset = load_dataset(DATA_PATH, name=subset, split="train", num_proc=96)
        all_raw_datasets.append(raw_dataset)

    # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets
    train_raw_dataset = concatenate_datasets(all_raw_datasets).shuffle(seed=0)
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
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name="siq_vlm_run1",
        # --- Batch Size & Speed ---
        per_device_train_batch_size=per_device_train_batch_size,  # Adjust based on VRAM (4-8 for 24GB)
        gradient_accumulation_steps=gradient_accumulation_steps,  # Effective batch size = 8 * 4 = 32
        dataloader_num_workers=4,
        # --- Learning Rate ---
        # Project alignment using 1e-3
        # Recommendation for full finetuning: 1e-5 to 2e-5
        learning_rate=1e-3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_steps=max_steps,  # VLMs overfit easily; 1000 steps is usually sufficient for multimodal project alignment
        max_grad_norm=1.0,  # Clip gradients to prevent instability
        # --- Precision & Memory ---
        bf16=True,  # REQUIRED for Qwen (Ampere+ GPUs). Do not use fp16.
        fp16=False,
        gradient_checkpointing=True,
        # --- Logging & Metrics & Saving ---
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,  # Keep only the last 2 checkpoints to save disk space
        report_to="wandb",
        project="siq_vl_stage_1",
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=[MetricsCallback(), SmartGPUCleanCallback()],
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
    train()
