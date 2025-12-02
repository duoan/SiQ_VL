import gc
import hashlib
import math
import time

from PIL import Image
import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

import wandb


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
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_beams: int = 2,
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
            print(">>> [GenerationCallback] Warning: Dataset does not support len(), cannot sample.")
            return []

        if dataset_len == 0:
            print(">>> [GenerationCallback] Warning: Dataset is empty.")
            return []

        indices = random.sample(range(dataset_len), min(num_samples, dataset_len))

        for idx in indices:
            try:
                item = dataset[idx]
            except Exception as e:
                print(f">>> [GenerationCallback] Warning: Failed to get item {idx} from dataset: {e}")
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
    ) -> list[str]:
        """
        Generate answers for a batch of samples using the model's generate method.

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
        self._run_generation(args, state, model=model, **kwargs)

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
        self._run_generation(args, state, model=model, **kwargs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Generate predictions and log to wandb at specified intervals."""
        # Only run on main process (rank 0)
        if not self._is_rank0(args):
            return

        # Check if we should evaluate at this step
        # Skip step 0 (handled in on_train_begin)
        if state.global_step == 0:
            return

        # Check if we should evaluate at this step based on interval
        if state.global_step % self.eval_interval != 0:
            return

        self._run_generation(args, state, model=model, **kwargs)

    def _run_generation(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model=None,
        **kwargs,
    ):
        """Internal method to run generation and log to wandb."""

        # Skip if wandb is not initialized
        if not wandb.run:
            return

        # Get model from kwargs if not provided
        if model is None:
            model = kwargs.get("model")

        if model is None:
            print(">>> [GenerationCallback] Warning: Model not found, skipping generation.")
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
            print(">>> [GenerationCallback] Warning: Processor not found, skipping generation.")
            return

        print(f">>> [GenerationCallback] Generating predictions at step {state.global_step}...")

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
                    print(">>> [GenerationCallback] Warning: No eval_dataset or eval_samples; skipping generation.")
                    return

                # Try to sample from eval_dataset (supports HF Dataset with __len__/__getitem__)
                try:
                    if hasattr(eval_dataset, "__getitem__") and hasattr(eval_dataset, "__len__"):
                        eval_samples = self._sample_from_dataset(eval_dataset, self.num_samples)
                    else:
                        print(">>> [GenerationCallback] Warning: eval_dataset is not indexable; skipping generation.")
                        return
                except Exception as e:
                    print(f">>> [GenerationCallback] Error sampling eval_dataset: {e}")
                    import traceback

                    traceback.print_exc()
                    return

            if not eval_samples:
                print(">>> [GenerationCallback] Warning: No eval samples found.")
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
                        print(f">>> [GenerationCallback] Error preparing sample {i}: {e}")
                        continue

                if not batch_samples:
                    print(">>> [GenerationCallback] Warning: No valid samples to generate.")
                    return

                print(f">>> [GenerationCallback] Generating answers for {len(batch_samples)} samples in batch...")

                # Generate answers for entire batch
                try:
                    generated_texts = self._generate_answer_batch(unwrapped_model, processor, batch_samples, device)
                except Exception as e:
                    print(f">>> [GenerationCallback] Error during batch generation: {e}")
                    import traceback

                    traceback.print_exc()
                    return

                # Build table data from batch results
                table_data = []
                for metadata, generated in zip(sample_metadata, generated_texts, strict=False):
                    # Get wandb image (reused if already uploaded)
                    wandb_img = self._get_wandb_image(metadata["image"])

                    table_data.append(
                        [
                            wandb_img,
                            metadata["question"],
                            metadata["ground_truth"],
                            generated,
                        ]
                    )
            finally:
                # Restore original training mode
                unwrapped_model.train(original_training)

            # Create wandb table
            table = wandb.Table(
                columns=["Image", "Question", "Ground Truth", "Generated Answer"],
                data=table_data,
            )

            # Log to wandb
            wandb.log(
                {
                    f"generation_samples/step_{state.global_step}": table,
                },
                step=state.global_step,
            )

            print(f">>> [GenerationCallback] Logged {len(table_data)} samples to wandb.")

        except Exception as e:
            print(f">>> [GenerationCallback] Error during generation: {e}")
            import traceback

            traceback.print_exc()
