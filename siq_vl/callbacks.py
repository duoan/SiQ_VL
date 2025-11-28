import gc
import math
import time

import torch
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
