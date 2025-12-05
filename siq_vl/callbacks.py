from contextlib import redirect_stderr, redirect_stdout
import gc
import hashlib
from io import StringIO
import logging
import math
import os
import time
import warnings

from PIL import Image
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.utilities.prints import rank_zero_info
import torchvision.transforms.functional as TF
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb

# Suppress warnings from transformers/accelerate about model sharding
warnings.filterwarnings("ignore", message=".*not sharded.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress accelerate warnings about sharding (these are printed, not logged)
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"


def clean_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def check_cuda_memory_and_clean(force: bool = False, verbose: bool = False):
    if not torch.cuda.is_available():
        return

    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_cached = memory_reserved - memory_allocated
        usage_ratio = memory_allocated / total_memory
        reserved_ratio = memory_reserved / total_memory
        cache_efficiency = memory_allocated / memory_reserved if memory_reserved > 0 else 1.0

        if verbose:
            print(
                f"[GPU {device}] Memory Status:"
                f"  Allocated:{memory_allocated:.2f}GB ({usage_ratio * 100:.1f}% of total),"
                f"  Reserved:{memory_reserved:.2f}GB ({reserved_ratio * 100:.1f}% of total),"
                f"  Cached:{memory_cached:.2f}GB,"
                f"  Efficiency:{cache_efficiency * 100:.1f}%"
            )

        should_clean = False
        reason = ""

        if force:
            should_clean = True
            reason = "Forced cleanup"

        elif usage_ratio > 0.75:
            should_clean = True
            reason = f"High memory usage: {usage_ratio * 100:.1f}%"

        elif reserved_ratio > 0.5 and cache_efficiency < 0.6:
            should_clean = True
            reason = f"Fragmentation detected: eff={cache_efficiency * 100:.1f}%, reserved={reserved_ratio * 100:.1f}%"

        elif memory_cached < 0.5 < usage_ratio and cache_efficiency > 0.95:
            should_clean = True
            reason = f"Low cache headroom ({memory_cached:.2f}GB)"

        if should_clean:
            if verbose:
                print(f"{reason}, cleaning memory...")

            clean_cuda_memory()

            new_allocated = torch.cuda.memory_allocated(device) / 1024**3
            new_reserved = torch.cuda.memory_reserved(device) / 1024**3
            freed = memory_allocated - new_allocated

            if verbose:
                print(
                    f"[GPU {device}] Cleaned: "
                    f"Allocated {memory_allocated:.2f}GB → {new_allocated:.2f}GB "
                    f"(freed {freed:.2f}GB), Reserved {memory_reserved:.2f}GB → {new_reserved:.2f}GB"
                )

    except Exception as e:
        print(f"Error during CUDA memory check: {e}")


class SmartGPUCleanCallback(TrainerCallback):
    """
    A callback that intelligently cleans GPU memory based on usage patterns.

    This callback monitors CUDA memory usage during training and performs
    cleanup when certain thresholds are met to prevent OOM errors.
    """

    def __init__(self, interval=50, verbose=True, force=False):
        """
        interval: every N steps run memory check
        verbose:  print logs on rank0
        force:    always clean regardless of memory threshold
        """
        self.interval = interval
        self.verbose = verbose
        self.force = force

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # only check every N steps
        if state.global_step > 0 and state.global_step % self.interval == 0:
            # rank0 prints logs, but all ranks must clean!
            is_rank0 = (not torch.distributed.is_initialized()) or args.local_rank == 0

            check_cuda_memory_and_clean(
                force=self.force,
                verbose=is_rank0 and self.verbose,
            )

            if torch.mps.is_available():
                torch.mps.empty_cache()


class MetricsCallback(TrainerCallback):
    """
    Callback to compute and log additional training metrics:
    - Perplexity (exp(loss))
    """

    def __init__(self):
        self.start_time = None
        self.total_tokens = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.start_time = time.time()
        self.total_tokens = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Calculate and add additional metrics when logging"""
        if logs is None:
            return

        # Calculate perplexity from loss
        if "loss" in logs:
            try:
                perplexity = math.exp(logs["loss"])
                # Cap perplexity at a reasonable value to avoid inf
                logs["perplexity"] = min(perplexity, 10000.0)
            except (OverflowError, ValueError):
                logs["perplexity"] = float("inf")


class GenerationCallback(TrainerCallback):
    """
    Callback to periodically generate predictions on fixed samples and log to wandb.
    Creates a wandb table with images, questions, ground truth answers, and generated answers.
    Reuses wandb images to avoid duplication.
    """

    def __init__(
        self,
        processor=None,
        eval_dataset=None,
        eval_samples: list[dict] | None = None,
        num_samples: int = 20,
        eval_interval: int = 100,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        do_sample: bool = False,  # greedy generation
        num_beams: int = 1,
    ):
        """
        Args:
            processor: SiQ_VLProcessor instance for processing inputs. If None, will try to get from Trainer.
            eval_dataset: Dataset to sample evaluation examples from. Should support __len__ and __getitem__.
                         If None and eval_samples is also None, generation is skipped.
            eval_samples: Fixed list of samples to evaluate on. Each sample should be a dict with:
                         {"image": PIL.Image, "question": str, "answer": str}
                         If None, will sample from eval_dataset.
            num_samples: Number of samples to use if eval_samples is None (default: 20)
            eval_interval: Evaluate every N steps (default: 100)
            max_new_tokens: Maximum number of tokens to generate (default: 256)
            temperature: Sampling temperature (default: 0.7)
            do_sample: Whether to use sampling (default: True)
            num_beams: Number of beams for beam search (default: 2)
        """
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.eval_samples = eval_samples
        self.num_samples = num_samples
        self.eval_interval = eval_interval
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.num_beams = num_beams

        # Cache for wandb images to avoid re-uploading
        self.image_cache: dict[str, wandb.Image] = {}

        # Lazy-loaded metric instances (created on first use)
        self._bert_score: BERTScore | None = None
        self._clip_score: CLIPScore | None = None
        self._rouge_score: ROUGEScore | None = None

    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate a hash for an image to use as cache key."""
        # Convert PIL image to bytes for hashing
        import io

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return hashlib.md5(img_bytes.read()).hexdigest()

    def _get_wandb_image(self, image: Image.Image) -> wandb.Image:
        """Get or create a wandb.Image, reusing cached version if available."""
        img_hash = self._get_image_hash(image)
        if img_hash not in self.image_cache:
            self.image_cache[img_hash] = wandb.Image(image)
        return self.image_cache[img_hash]

    def _get_bert_score(self) -> BERTScore:
        """Get or create BERTScore instance (lazy loading)."""
        if self._bert_score is None:
            # Suppress warnings and print statements during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr to suppress print statements from accelerate
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    # NOTE:
                    # Some BERTScore backbone models (e.g. RoBERTa-based) only support
                    # sequences up to 512/514 tokens. Without truncation, very long
                    # generated answers can cause shape mismatches inside the model
                    # (e.g. attention_mask length > token_type_ids buffer length).
                    # We explicitly enable truncation at 512 tokens to avoid:
                    #   RuntimeError: The expanded size of the tensor (640) must match
                    #   the existing size (514) at non-singleton dimension 1.
                    self._bert_score = BERTScore(
                        rescale_with_baseline=True,
                        max_length=512,
                        truncation=True,
                    )
        return self._bert_score

    def _get_clip_score(self) -> CLIPScore:
        """Get or create CLIPScore instance (lazy loading)."""
        if self._clip_score is None:
            # Suppress warnings and print statements during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr to suppress print statements from accelerate
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self._clip_score = CLIPScore()
        return self._clip_score

    def _get_rouge_score(self) -> ROUGEScore:
        """Get or create ROUGEScore instance (lazy loading)."""
        if self._rouge_score is None:
            # Suppress warnings and print statements during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr to suppress print statements from accelerate
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self._rouge_score = ROUGEScore()
        return self._rouge_score

    def _sample_from_dataset(self, dataset, num_samples: int) -> list[dict]:
        """Sample fixed examples from the dataset."""
        samples = []
        # Use a fixed seed to get the same samples every time
        import random

        random.seed(42)

        # Get dataset length
        try:
            dataset_len = len(dataset)
        except (TypeError, AttributeError):
            rank_zero_info(">>> [GenerationCallback] Warning: Dataset does not support len(), cannot sample.")
            return []

        if dataset_len == 0:
            rank_zero_info(">>> [GenerationCallback] Warning: Dataset is empty.")
            return []

        indices = random.sample(range(dataset_len), min(num_samples, dataset_len))

        for idx in indices:
            try:
                item = dataset[idx]
            except Exception as e:
                rank_zero_info(f">>> [GenerationCallback] Warning: Failed to get item {idx} from dataset: {e}")
                continue

            # Extract image, question, and answer
            # Handle different dataset formats
            image = None
            question = None
            answer = None

            # Try to get image from different possible keys
            if "images" in item:
                images = item["images"]
                if isinstance(images, list) and len(images) > 0:
                    image = images[0]
                elif not isinstance(images, list):
                    image = images
            elif "image" in item:
                image = item["image"]

            if image is None:
                continue

            # Handle different text formats
            if "texts" in item and len(item["texts"]) > 0:
                # Multi-turn format (like VQAIterableDataset's underlying format)
                turn = item["texts"][0]
                question = turn.get("user", "")
                answer = turn.get("assistant", "")
            elif "question" in item and "answer" in item:
                # Simple Q&A format (like VQAIterableDataset's yielded format)
                question = item["question"]
                answer = item["answer"]
            else:
                # Try to find question/answer in other possible keys
                question = item.get("question") or item.get("q") or item.get("prompt")
                answer = item.get("answer") or item.get("a") or item.get("response")

            if question is None:
                continue

            samples.append(
                {
                    "image": image,
                    "question": question,
                    "answer": answer or "",
                }
            )

        return samples

    def _generate_answer_batch(
        self,
        model,
        processor,
        samples: list[tuple[Image.Image, str]],
        device: torch.device,
        batch_size: int | None = None,
    ) -> list[str]:
        """
        Generate answers for a batch of samples using the model's generate method.
        If batch_size is provided, splits the samples into smaller batches to avoid OOM.

        Args:
            model: The model to use for generation
            processor: The processor to use for tokenization
            samples: List of (image, question) tuples
            device: Device to run generation on
            batch_size: Maximum batch size per generation call. If None, processes entire batch at once.

        Returns:
            List of generated answer strings
        """
        if not samples:
            return []

        # If batch_size is not specified or batch is small enough, process all at once
        if batch_size is None or len(samples) <= batch_size:
            return self._generate_answer_batch_single(model, processor, samples, device)

        # Split into smaller batches and process sequentially
        all_generated_texts = []
        num_batches = math.ceil(len(samples) / batch_size)

        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i : i + batch_size]
            rank_zero_info(
                f">>> [GenerationCallback] Processing batch {i // batch_size + 1}/{num_batches} "
                f"({len(batch_samples)} samples)..."
            )
            batch_texts = self._generate_answer_batch_single(model, processor, batch_samples, device)
            all_generated_texts.extend(batch_texts)

            # Clean memory after each batch to prevent accumulation
            clean_cuda_memory()

        return all_generated_texts

    def _generate_answer_batch_single(
        self,
        model,
        processor,
        samples: list[tuple[Image.Image, str]],
        device: torch.device,
    ) -> list[str]:
        """
        Generate answers for a single batch of samples (internal helper method).

        Args:
            model: The model to use for generation
            processor: The processor to use for tokenization
            samples: List of (image, question) tuples
            device: Device to run generation on

        Returns:
            List of generated answer strings
        """
        if not samples:
            return []

        # Process batch: (image, question, None) for generation mode
        batch = [(image, question, None) for image, question in samples]
        inputs = processor(
            batch=batch,
            return_tensors="pt",
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        pixel_values = inputs["pixel_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Generate for entire batch
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                num_beams=self.num_beams if not self.do_sample else 1,
                repetition_penalty=1.2,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode all generated sequences (assistant responses only)
        generated_texts = processor.batch_decode(
            output_ids,
            assistant_only=True,
            skip_special_tokens=True,
        )

        return [text.strip() for text in generated_texts]

    def _is_rank0(self, args: TrainingArguments) -> bool:
        """Check if current process is rank 0 (main process)."""
        # If distributed training is not initialized, we're on rank 0
        if not torch.distributed.is_initialized():
            return True
        # In distributed training, check local_rank
        return args.local_rank == 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Run generation at the start of training (step 0)."""
        # Only run on main process (rank 0)
        if not self._is_rank0(args):
            return

        # Run generation at step 0
        self._run_generation(args, state, model=model, log_step=0, **kwargs)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Run generation at the end of training."""
        # Only run on main process (rank 0)
        if not self._is_rank0(args):
            return

        # Skip if the final step is already covered by eval_interval
        # (e.g., if eval_interval=200 and final step=1000, on_log already ran at step 1000)
        if state.global_step > 0 and state.global_step % self.eval_interval == 0:
            return

        # Run generation at final step (only if not already done by on_log)
        # Save the step value to ensure consistent logging
        final_step = state.global_step
        self._run_generation(args, state, model=model, log_step=final_step, **kwargs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs=None,
        **kwargs,
    ):
        """Generate predictions and log to wandb at specified intervals."""
        # Only run on main process (rank 0)
        if not self._is_rank0(args):
            return

        # CRITICAL: Skip generation if this is an eval log call
        # After evaluation, Trainer calls on_log with logs containing eval metrics (e.g., 'eval_loss')
        # This would cause wandb step conflicts because eval metrics are logged with a different step
        # We should only run generation during training steps, not after eval
        if logs is not None and any(key.startswith("eval_") for key in logs):
            return

        # Check if we should evaluate at this step
        # Skip step 0 (handled in on_train_begin)
        if state.global_step == 0:
            return

        # Check if we should evaluate at this step based on interval
        if state.global_step % self.eval_interval != 0:
            return

        # CRITICAL: Save the step value at the time of on_log call
        # Generation may take time, and state.global_step may increase during generation
        # We need to use the step value from when on_log was called, not when generation finishes
        log_step = state.global_step
        self._run_generation(args, state, model=model, log_step=log_step, **kwargs)

    def _run_generation(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model=None,
        log_step: int | None = None,
        **kwargs,
    ):
        """Internal method to run generation and log to wandb.

        Args:
            log_step: The step value to use for wandb logging. If None, uses state.global_step.
                      This should be the step value from when on_log was called, not the current step.
        """

        # Skip if wandb is not initialized
        if not wandb.run:
            return

        # Get model from kwargs if not provided
        if model is None:
            model = kwargs.get("model")

        if model is None:
            rank_zero_info(">>> [GenerationCallback] Warning: Model not found, skipping generation.")
            return

        # Get processor - try from self first, then from Trainer
        processor = self.processor
        if processor is None:
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "processing_class"):
                processor = trainer.processing_class
            elif (
                trainer is not None
                and hasattr(trainer, "data_collator")
                and hasattr(trainer.data_collator, "processor")
            ):
                processor = trainer.data_collator.processor

        if processor is None:
            rank_zero_info(">>> [GenerationCallback] Warning: Processor not found, skipping generation.")
            return

        # Use the saved log_step if provided, otherwise fall back to current state.global_step
        step_to_log = log_step if log_step is not None else state.global_step
        rank_zero_info(f">>> [GenerationCallback] Generating predictions at step {step_to_log}...")

        try:
            # Decide where to get eval samples from
            eval_samples = self.eval_samples

            # If no fixed eval_samples, we need an eval_dataset to sample from
            if eval_samples is None:
                eval_dataset = self.eval_dataset
                if eval_dataset is None:
                    # Try to get from Trainer
                    trainer = kwargs.get("trainer")
                    if trainer is not None and hasattr(trainer, "eval_dataset") and trainer.eval_dataset is not None:
                        eval_dataset = trainer.eval_dataset
                        # If it's an IterableDataset, try to get the underlying dataset
                        if hasattr(eval_dataset, "dataset"):
                            eval_dataset = eval_dataset.dataset

                if eval_dataset is None:
                    rank_zero_info(
                        ">>> [GenerationCallback] Warning: No eval_dataset or eval_samples; skipping generation."
                    )
                    return

                # Try to sample from eval_dataset (supports HF Dataset with __len__/__getitem__)
                try:
                    if hasattr(eval_dataset, "__getitem__") and hasattr(eval_dataset, "__len__"):
                        eval_samples = self._sample_from_dataset(eval_dataset, self.num_samples)
                    else:
                        rank_zero_info(
                            ">>> [GenerationCallback] Warning: eval_dataset is not indexable; skipping generation."
                        )
                        return
                except Exception as e:
                    rank_zero_info(f">>> [GenerationCallback] Error sampling eval_dataset: {e}")
                    import traceback

                    traceback.print_exc()
                    return

            if not eval_samples:
                rank_zero_info(">>> [GenerationCallback] Warning: No eval samples found.")
                return

            # Get device and unwrap model - use unwrapped model to avoid DDP issues
            # CRITICAL: Unwrap DDP model to prevent hook conflicts during generation
            unwrapped_model = model.module if hasattr(model, "module") else model
            device = next(unwrapped_model.parameters()).device

            # Generate predictions
            # CRITICAL: Ensure we're in eval mode and using no_grad context
            # to prevent interference with DDP hooks during training
            original_training = unwrapped_model.training

            try:
                unwrapped_model.eval()
                # Prepare all samples for batch processing
                from siq_vl.dataset import _to_pil_rgb

                batch_samples = []
                sample_metadata = []
                for i, sample in enumerate(eval_samples):
                    try:
                        image = _to_pil_rgb(sample["image"])
                        question = sample["question"]
                        ground_truth = sample.get("answer", "")

                        batch_samples.append((image, question))
                        sample_metadata.append(
                            {
                                "image": image,
                                "question": question,
                                "ground_truth": ground_truth,
                                "index": i,
                            }
                        )
                    except Exception as e:
                        rank_zero_info(f">>> [GenerationCallback] Error preparing sample {i}: {e}")
                        continue

                if not batch_samples:
                    rank_zero_info(">>> [GenerationCallback] Warning: No valid samples to generate.")
                    return

                rank_zero_info(
                    f">>> [GenerationCallback] Generating answers for {len(batch_samples)} samples in batch..."
                )

                # Generate answers for entire batch (with batch splitting if needed)
                try:
                    batch_size = (
                        args.per_device_train_batch_size if hasattr(args, "per_device_train_batch_size") else None
                    )
                    generated_texts = self._generate_answer_batch(
                        unwrapped_model, processor, batch_samples, device, batch_size=batch_size
                    )
                except Exception as e:
                    rank_zero_info(f">>> [GenerationCallback] Error during batch generation: {e}")
                    import traceback

                    traceback.print_exc()
                    return

                # Calcalte the metrics for the generated texts
                ground_truths = [metadata["ground_truth"] for metadata in sample_metadata]
                # Convert PIL images to tensors in the format expected by clip_score: [C, H, W] with values in [0, 1]
                images = [metadata["image"] for metadata in sample_metadata]
                images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
                # Convert PIL to tensor: (H, W, C) -> (C, H, W), values in [0, 1]
                images = [TF.pil_to_tensor(image) for image in images]

                # calculate the metrics 1 sample by 1 sample
                generated_metrics = []

                # Use cached metric instances (lazy-loaded, created once)
                bert_score = self._get_bert_score()
                clip_score = self._get_clip_score()
                rouge_score = self._get_rouge_score()

                for generated_text, ground_truth, image in zip(generated_texts, ground_truths, images, strict=False):
                    # BERT scores
                    generated_bert_scores = bert_score(generated_text, ground_truth)
                    generated_bert_f1 = generated_bert_scores["f1"].detach().cpu().numpy().mean().item()
                    generated_bert_precision = generated_bert_scores["precision"].detach().cpu().numpy().mean().item()
                    generated_bert_recall = generated_bert_scores["recall"].detach().cpu().numpy().mean().item()

                    # Rouge scores
                    """
                    {'rouge1_fmeasure': tensor(0.7500),
                    'rouge1_precision': tensor(0.7500),
                    'rouge1_recall': tensor(0.7500),
                    'rouge2_fmeasure': tensor(0.),
                    'rouge2_precision': tensor(0.),
                    'rouge2_recall': tensor(0.),
                    'rougeL_fmeasure': tensor(0.5000),
                    'rougeL_precision': tensor(0.5000),
                    'rougeL_recall': tensor(0.5000),
                    'rougeLsum_fmeasure': tensor(0.5000),
                    'rougeLsum_precision': tensor(0.5000),
                    'rougeLsum_recall': tensor(0.5000)}
                    """
                    generated_rouge_scores = rouge_score(generated_text, ground_truth)
                    for key, value in generated_rouge_scores.items():
                        generated_rouge_scores[key] = value.detach().cpu().numpy().mean().item()

                    # Clip scores
                    generated_img_clip = clip_score(generated_text, image).detach().cpu().numpy().mean().item()
                    generated_ans_clip = clip_score(generated_text, ground_truth).detach().cpu().numpy().mean().item()

                    generated_metrics.append(
                        {
                            "bert_f1": generated_bert_f1,
                            "bert_precision": generated_bert_precision,
                            "bert_recall": generated_bert_recall,
                            "img_clip": generated_img_clip,
                            "ans_clip": generated_ans_clip,
                            "rouge1_fmeasure": generated_rouge_scores["rouge1_fmeasure"],
                            "rouge1_precision": generated_rouge_scores["rouge1_precision"],
                            "rouge1_recall": generated_rouge_scores["rouge1_recall"],
                            "rouge2_fmeasure": generated_rouge_scores["rouge2_fmeasure"],
                            "rouge2_precision": generated_rouge_scores["rouge2_precision"],
                            "rouge2_recall": generated_rouge_scores["rouge2_recall"],
                            "rougeL_fmeasure": generated_rouge_scores["rougeL_fmeasure"],
                            "rougeL_precision": generated_rouge_scores["rougeL_precision"],
                            "rougeL_recall": generated_rouge_scores["rougeL_recall"],
                            "rougeLsum_fmeasure": generated_rouge_scores["rougeLsum_fmeasure"],
                            "rougeLsum_precision": generated_rouge_scores["rougeLsum_precision"],
                            "rougeLsum_recall": generated_rouge_scores["rougeLsum_recall"],
                        }
                    )

                # Note: We don't delete the metric instances anymore since they're cached for reuse
                gc.collect()
                check_cuda_memory_and_clean(force=True, verbose=True)

                # calculate the average metrics
                import numpy as np

                average_metrics = {
                    "gen/bert_f1": np.mean([metrics["bert_f1"] for metrics in generated_metrics]),
                    "gen/bert_precision": np.mean([metrics["bert_precision"] for metrics in generated_metrics]),
                    "gen/bert_recall": np.mean([metrics["bert_recall"] for metrics in generated_metrics]),
                    "gen/img_clip": np.mean([metrics["img_clip"] for metrics in generated_metrics]),
                    "gen/ans_clip": np.mean([metrics["ans_clip"] for metrics in generated_metrics]),
                }

                # Build table data from batch results
                table_data = []

                for metadata, generated, metrics in zip(
                    sample_metadata, generated_texts, generated_metrics, strict=False
                ):
                    # Get wandb image (reused if already uploaded)
                    wandb_img = self._get_wandb_image(metadata["image"])

                    table_data.append(
                        [
                            wandb_img,
                            metadata["question"],
                            metadata["ground_truth"],
                            generated,
                            metrics["bert_f1"],
                            metrics["bert_precision"],
                            metrics["bert_recall"],
                            metrics["img_clip"],
                            metrics["ans_clip"],
                            metrics["rouge1_fmeasure"],
                            metrics["rouge1_precision"],
                            metrics["rouge1_recall"],
                            metrics["rouge2_fmeasure"],
                            metrics["rouge2_precision"],
                            metrics["rouge2_recall"],
                            metrics["rougeL_fmeasure"],
                            metrics["rougeL_precision"],
                            metrics["rougeL_recall"],
                            metrics["rougeLsum_fmeasure"],
                            metrics["rougeLsum_precision"],
                            metrics["rougeLsum_recall"],
                        ]
                    )
            finally:
                # Restore original training mode
                unwrapped_model.train(original_training)

            # Create wandb table
            table = wandb.Table(
                columns=[
                    "Image",
                    "Question",
                    "Ground Truth",
                    "Generated Answer",
                    "BERT F1",
                    "BERT Precision",
                    "BERT Recall",
                    "Image CLIP",
                    "Answer CLIP",
                    "ROUGE1 F1",
                    "ROUGE1 Precision",
                    "ROUGE1 Recall",
                    "ROUGE2 F1",
                    "ROUGE2 Precision",
                    "ROUGE2 Recall",
                    "ROUGE L F1",
                    "ROUGE L Precision",
                    "ROUGE L Recall",
                    "ROUGE LSUM F1",
                    "ROUGE LSUM Precision",
                    "ROUGE LSUM Recall",
                ],
                data=table_data,
            )

            # Log to wandb using the step value from when on_log was called
            # Use the saved log_step if provided, otherwise fall back to current state.global_step
            step_to_log = log_step if log_step is not None else state.global_step
            log_payload = {
                f"generation_samples/step_{step_to_log}": table,
            }
            log_payload.update(average_metrics)
            wandb.log(log_payload, step=step_to_log)

            rank_zero_info(f">>> [GenerationCallback] Logged {len(table_data)} samples to wandb.")

        except Exception as e:
            rank_zero_info(f">>> [GenerationCallback] Error during generation: {e}")
            import traceback

            traceback.print_exc()
