import gc
import hashlib
import math
import time
from typing import Dict, List, Optional

import torch
import wandb
from PIL import Image
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)



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
        cache_efficiency = (
            memory_allocated / memory_reserved if memory_reserved > 0 else 1.0
        )

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
            reason = (
                f"Fragmentation detected: eff={cache_efficiency * 100:.1f}%, "
                f"reserved={reserved_ratio * 100:.1f}%"
            )

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
        processor,
        eval_dataset=None,
        eval_samples: Optional[List[Dict]] = None,
        num_samples: int = 20,
        eval_interval: int = 100,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_beams: int = 2,
    ):
        """
        Args:
            processor: Processor used for text/image preprocessing.
            eval_dataset: Dataset to sample evaluation examples from (e.g. HF dataset).
                         If None and eval_samples is also None, generation is skipped.
            eval_samples: Fixed list of samples to evaluate on. If None, will sample from eval_dataset.
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
        self.image_cache: Dict[str, wandb.Image] = {}

    def _get_image_hash(self, image: Image.Image) -> str:
        """Generate a hash for an image to use as cache key."""
        # Convert PIL image to bytes for hashing
        import io
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return hashlib.md5(img_bytes.read()).hexdigest()
    
    def _get_wandb_image(self, image: Image.Image) -> wandb.Image:
        """Get or create a wandb.Image, reusing cached version if available."""
        img_hash = self._get_image_hash(image)
        if img_hash not in self.image_cache:
            self.image_cache[img_hash] = wandb.Image(image)
        return self.image_cache[img_hash]
    
    def _sample_from_dataset(self, dataset, num_samples: int) -> List[Dict]:
        """Sample fixed examples from the dataset."""
        samples = []
        # Use a fixed seed to get the same samples every time
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        for idx in indices:
            item = dataset[idx]
            # Extract image, question, and answer
            image = item.get("images", [None])[0] if "images" in item else None
            if image is None:
                continue
                
            # Handle different dataset formats
            if "texts" in item and len(item["texts"]) > 0:
                # Multi-turn format
                turn = item["texts"][0]
                question = turn.get("user", "")
                answer = turn.get("assistant", "")
            elif "question" in item and "answer" in item:
                # Simple Q&A format
                question = item["question"]
                answer = item["answer"]
            else:
                continue
                
            samples.append({
                "image": image,
                "question": question,
                "answer": answer,
            })
        
        return samples
    
    def _generate_answer_simple(
        self,
        model,
        processor,
        image: Image.Image,
        question: str,
        device: torch.device,
    ) -> str:
        """Generate answer using the model's generate_answer method."""
        return model.generate_answer(
            processor=processor,
            samples=(image, question),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            device=device,
        )
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Run generation at the start of training (step 0)."""
        # Only run on main process
        if args.local_rank != 0 and args.local_rank != -1:
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
        # Only run on main process
        if args.local_rank != 0 and args.local_rank != -1:
            return
        
        # Run generation at final step
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
        # Only run on main process
        if args.local_rank != 0 and args.local_rank != -1:
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
        
        print(f">>> [GenerationCallback] Generating predictions at step {state.global_step}...")
        
        try:
            processor = self.processor

            # Decide where to get eval samples from
            eval_samples = self.eval_samples

            # If no fixed eval_samples, we need an eval_dataset to sample from
            if eval_samples is None:
                eval_dataset = self.eval_dataset
                if eval_dataset is None:
                    print(">>> [GenerationCallback] Warning: No eval_dataset or eval_samples; skipping generation.")
                    return

            if eval_samples is None:
                # Try to sample from eval_dataset (supports HF Dataset with __len__/__getitem__)
                try:
                    if hasattr(eval_dataset, "__getitem__") and hasattr(eval_dataset, "__len__"):
                        eval_samples = self._sample_from_dataset(
                            eval_dataset, self.num_samples
                        )
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
            
            # Get device
            device = next(model.parameters()).device
            
            # Generate predictions
            table_data = []
            for i, sample in enumerate(eval_samples):
                try:
                    # Convert image to PIL if needed
                    from siq_vl.dataset import _to_pil_rgb
                    image = _to_pil_rgb(sample["image"])
                    question = sample["question"]
                    ground_truth = sample["answer"]
                    
                    # Generate answer
                    # Use a simpler generation approach
                    generated = self._generate_answer_simple(
                        model, processor, image, question, device
                    )
                    
                    # Get wandb image (reused if already uploaded)
                    wandb_img = self._get_wandb_image(image)
                    
                    table_data.append([
                        wandb_img,
                        question,
                        ground_truth,
                        generated,
                    ])
                    
                    if (i + 1) % 5 == 0:
                        print(f">>> [GenerationCallback] Generated {i + 1}/{len(eval_samples)} samples...")
                        
                except Exception as e:
                    print(f">>> [GenerationCallback] Error generating for sample {i}: {e}")
                    continue
            
            # Create wandb table
            table = wandb.Table(
                columns=["Image", "Question", "Ground Truth", "Generated Answer"],
                data=table_data,
            )
            
            # Log to wandb
            wandb.log({
                f"generation_samples/step_{state.global_step}": table,
            }, step=state.global_step)
            
            print(f">>> [GenerationCallback] Logged {len(table_data)} samples to wandb.")
            
        except Exception as e:
            print(f">>> [GenerationCallback] Error during generation: {e}")
            import traceback
            traceback.print_exc()
