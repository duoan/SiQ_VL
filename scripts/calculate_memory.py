#!/usr/bin/env python3
"""
Calculate GPU memory usage for SiQ-VL model training.

This script estimates memory consumption for different configurations
and helps identify OOM issues.
"""


def calculate_model_params(
    vision_model: str,
    llm_model: str,
    pixel_shuffle_factor: int = 1,
) -> dict[str, int]:
    """
    Calculate number of parameters for each component.

    Returns:
        Dictionary with parameter counts for each component
    """
    # Vision model parameters (frozen, but still loaded)
    # SigLIP2 SO400M has ~400M parameters
    vision_params = {
        "siglip2-base-patch16-224": 86_000_000,  # ~86M
        "siglip2-so400m-patch16-512": 400_000_000,  # ~400M
        "siglip2-so400m-patch14-384": 400_000_000,  # ~400M
    }

    # LLM parameters
    llm_params = {
        "Qwen/Qwen2.5-0.5B-Instruct": 500_000_000,  # ~500M
        "Qwen/Qwen2.5-1.5B-Instruct": 1_500_000_000,  # ~1.5B
        "Qwen/Qwen2.5-3B-Instruct": 3_000_000_000,  # ~3B
    }

    # Get vision model size
    vision_key = vision_model.split("/")[-1] if "/" in vision_model else vision_model
    vision_size = vision_params.get(vision_key, 400_000_000)

    # Get LLM size
    llm_size = llm_params.get(llm_model, 1_500_000_000)

    # Projector parameters
    # Vision hidden size (SigLIP SO400M: 1152, Base: 768)
    vision_hidden = 1152 if "so400m" in vision_key.lower() else 768

    # LLM hidden size (Qwen 0.5B: 896, 1.5B: 1024, 3B: 2048)
    if "0.5b" in llm_model.lower():
        llm_hidden = 896
    elif "1.5b" in llm_model.lower():
        llm_hidden = 1024
    elif "3b" in llm_model.lower():
        llm_hidden = 2048
    else:
        llm_hidden = 1024  # Default

    # Projector: Linear(vision_hidden * pixel_shuffle_factor^2, llm_hidden)
    projector_input_dim = vision_hidden * (pixel_shuffle_factor**2)
    projector_params = projector_input_dim * llm_hidden  # Linear layer weights

    return {
        "vision_model": vision_size,
        "llm_model": llm_size,
        "projector": projector_params,
        "vision_hidden": vision_hidden,
        "llm_hidden": llm_hidden,
        "pixel_shuffle_factor": pixel_shuffle_factor,
    }


def calculate_sequence_lengths(
    vision_model: str,
    pixel_shuffle_factor: int = 1,
) -> dict[str, int]:
    """
    Calculate sequence lengths for vision tokens.

    Returns:
        Dictionary with sequence length information
    """
    # Extract image size and patch size from model name
    vision_key = vision_model.split("/")[-1] if "/" in vision_model else vision_model

    # Parse image size (e.g., patch16-512 -> 512)
    if "patch16-512" in vision_key:
        image_size = 512
        patch_size = 16
    elif "patch16-224" in vision_key:
        image_size = 224
        patch_size = 16
    elif "patch14-384" in vision_key:
        image_size = 384
        patch_size = 14
    else:
        # Defaults
        image_size = 512
        patch_size = 16

    # Calculate patches
    patches_per_dim = image_size // patch_size
    num_patches_before = patches_per_dim**2

    # After pixel shuffle
    patches_per_dim_after = patches_per_dim // pixel_shuffle_factor
    num_patches_after = patches_per_dim_after**2

    return {
        "image_size": image_size,
        "patch_size": patch_size,
        "patches_per_dim": patches_per_dim,
        "num_patches_before_shuffle": num_patches_before,
        "num_patches_after_shuffle": num_patches_after,
        "reduction_factor": num_patches_before / num_patches_after if num_patches_after > 0 else 1,
    }


def estimate_memory_usage(
    vision_model: str,
    llm_model: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_len: int = 2048,
    pixel_shuffle_factor: int = 1,
    precision: str = "bf16",  # "bf16", "fp16", "fp32"
    freeze_llm: bool = True,
    gradient_checkpointing: bool = True,
) -> dict[str, float]:
    """
    Estimate GPU memory usage in GB.

    Args:
        vision_model: Vision model name
        llm_model: LLM model name
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_seq_len: Maximum sequence length (text + image tokens)
        pixel_shuffle_factor: Pixel shuffle factor
        precision: Training precision
        freeze_llm: Whether LLM is frozen
        gradient_checkpointing: Whether gradient checkpointing is enabled

    Returns:
        Dictionary with memory breakdown in GB
    """
    # Get model parameters
    model_info = calculate_model_params(vision_model, llm_model, pixel_shuffle_factor)
    seq_info = calculate_sequence_lengths(vision_model, pixel_shuffle_factor)

    # Precision multiplier (bytes per parameter)
    precision_bytes = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
    }
    bytes_per_param = precision_bytes.get(precision, 2)

    # 1. Model weights memory
    vision_memory = (model_info["vision_model"] * bytes_per_param) / (1024**3)  # GB
    llm_memory = (model_info["llm_model"] * bytes_per_param) / (1024**3)  # GB
    projector_memory = (model_info["projector"] * bytes_per_param) / (1024**3)  # GB

    # 2. Activation memory (forward pass)
    # Vision features: [batch, num_patches, vision_hidden]
    num_patches = seq_info["num_patches_after_shuffle"]
    vision_hidden = model_info["vision_hidden"]
    llm_hidden = model_info["llm_hidden"]

    # Vision activations
    vision_activations = (batch_size * num_patches * vision_hidden * bytes_per_param) / (1024**3)

    # Projected image embeddings: [batch, num_patches, llm_hidden]
    image_embeds = (batch_size * num_patches * llm_hidden * bytes_per_param) / (1024**3)

    # Text embeddings: [batch, text_seq_len, llm_hidden]
    # max_seq_len is the maximum text sequence length (typically 2048)
    # The actual total sequence length includes both text and image tokens
    text_seq_len = max_seq_len  # This is the text part
    text_embeds = (batch_size * text_seq_len * llm_hidden * bytes_per_param) / (1024**3)

    # Combined input embeddings: [batch, total_seq_len, llm_hidden]
    # This is the actual sequence length that goes into the LLM
    # It includes: text tokens + image tokens (after pixel shuffle)
    # THIS IS THE KEY: pixel_shuffle reduces num_patches, which reduces total_seq_len
    actual_seq_len = text_seq_len + num_patches
    combined_embeds = (batch_size * actual_seq_len * llm_hidden * bytes_per_param) / (1024**3)

    # LLM activations (more accurate estimate)
    # For transformer: activations include:
    # 1. Hidden states per layer: [batch, seq_len, hidden]
    # 2. Attention matrices: [batch, num_heads, seq_len, seq_len] - THIS IS THE BIG ONE!
    # 3. FFN activations: [batch, seq_len, hidden * 4] (intermediate size)

    # Qwen 1.5B has ~24 layers, 16 attention heads
    num_layers = 24 if "1.5b" in llm_model.lower() else 20
    num_heads = 16 if "1.5b" in llm_model.lower() else 12

    # Use actual sequence length (text + image tokens)
    actual_seq_len = text_seq_len + num_patches

    if gradient_checkpointing:
        # Gradient checkpointing: only store activations at checkpoint boundaries
        # Typically reduces by 4-8x, but attention matrices still need to be computed
        checkpoint_interval = 4  # Checkpoint every 4 layers
        stored_layers = max(1, num_layers // checkpoint_interval)
        llm_hidden_states = (batch_size * actual_seq_len * llm_hidden * stored_layers * bytes_per_param) / (1024**3)

        # Attention matrices: O(n^2) - this is the memory killer!
        # Even with checkpointing, we need to compute attention for each layer
        # But we can recompute, so we only store a few at a time
        attention_memory_per_layer = (batch_size * num_heads * actual_seq_len * actual_seq_len * bytes_per_param) / (
            1024**3
        )
        attention_memory = attention_memory_per_layer * min(4, num_layers)  # Store ~4 at a time

        # FFN activations (intermediate size is 4x hidden)
        ffn_memory = (batch_size * actual_seq_len * llm_hidden * 4 * stored_layers * bytes_per_param) / (1024**3)

        llm_activations = llm_hidden_states + attention_memory + ffn_memory
        llm_attention_memory = attention_memory  # Store separately for reporting
    else:
        # Without checkpointing: store all activations
        llm_hidden_states = (batch_size * actual_seq_len * llm_hidden * num_layers * bytes_per_param) / (1024**3)

        # Attention matrices for all layers
        attention_memory = (batch_size * num_heads * actual_seq_len * actual_seq_len * num_layers * bytes_per_param) / (
            1024**3
        )

        # FFN activations
        ffn_memory = (batch_size * actual_seq_len * llm_hidden * 4 * num_layers * bytes_per_param) / (1024**3)

        llm_activations = llm_hidden_states + attention_memory + ffn_memory
        llm_attention_memory = attention_memory  # Store separately for reporting

    # 3. Gradient memory (only for trainable parameters)
    trainable_params = model_info["projector"]
    if not freeze_llm:
        trainable_params += model_info["llm_model"]

    gradient_memory = (trainable_params * bytes_per_param) / (1024**3)

    # 4. Optimizer memory (AdamW: 2x parameters for momentum + variance)
    optimizer_memory = (trainable_params * 2 * bytes_per_param) / (1024**3)

    # 5. Input data memory
    # Pixel values: [batch, 3, image_size, image_size]
    pixel_memory = (
        batch_size * 3 * seq_info["image_size"] * seq_info["image_size"] * 4  # fp32 for input
    ) / (1024**3)

    # Input IDs: [batch, max_seq_len]
    input_ids_memory = (batch_size * max_seq_len * 4) / (1024**3)  # int32

    # 6. Overhead (PyTorch, CUDA, etc.)
    overhead = 2.0  # GB

    # Total memory
    total_memory = (
        vision_memory
        + llm_memory
        + projector_memory
        + vision_activations
        + image_embeds
        + text_embeds
        + combined_embeds
        + llm_activations
        + gradient_memory
        + optimizer_memory
        + pixel_memory
        + input_ids_memory
        + overhead
    )

    return {
        "model_weights": {
            "vision_model": vision_memory,
            "llm_model": llm_memory,
            "projector": projector_memory,
            "total_weights": vision_memory + llm_memory + projector_memory,
        },
        "activations": {
            "vision_activations": vision_activations,
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "combined_embeds": combined_embeds,
            "llm_activations": llm_activations,
            "llm_attention_memory": llm_attention_memory,
            "total_activations": (vision_activations + image_embeds + text_embeds + combined_embeds + llm_activations),
        },
        "training": {
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "total_training": gradient_memory + optimizer_memory,
        },
        "data": {
            "pixel_values": pixel_memory,
            "input_ids": input_ids_memory,
            "total_data": pixel_memory + input_ids_memory,
        },
        "overhead": overhead,
        "total_memory_gb": total_memory,
        "sequence_info": seq_info,
        "model_info": model_info,
    }


def print_memory_report(memory_breakdown: dict, config: dict):
    """Print a formatted memory usage report."""
    print("=" * 80)
    print("GPU MEMORY USAGE ESTIMATION")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Vision Model: {config['vision_model']}")
    print(f"  LLM Model: {config['llm_model']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Precision: {config['precision']}")
    print(f"  Freeze LLM: {config['freeze_llm']}")
    print(f"  Gradient Checkpointing: {config['gradient_checkpointing']}")
    print(f"  Pixel Shuffle Factor: {config['pixel_shuffle_factor']}")

    seq_info = memory_breakdown["sequence_info"]
    print("\nSequence Information:")
    print(f"  Image Size: {seq_info['image_size']}x{seq_info['image_size']}")
    print(f"  Patch Size: {seq_info['patch_size']}x{seq_info['patch_size']}")
    print(
        f"  Patches Before Shuffle: {seq_info['num_patches_before_shuffle']} ({seq_info['patches_per_dim']}x{seq_info['patches_per_dim']})"
    )
    print(
        f"  Patches After Shuffle: {seq_info['num_patches_after_shuffle']} (reduction: {seq_info['reduction_factor']:.1f}x)"
    )

    print("\nMemory Breakdown (GB):")
    print("\n  Model Weights:")
    print(f"    Vision Model:     {memory_breakdown['model_weights']['vision_model']:>8.2f} GB")
    print(f"    LLM Model:        {memory_breakdown['model_weights']['llm_model']:>8.2f} GB")
    print(f"    Projector:        {memory_breakdown['model_weights']['projector']:>8.2f} GB")
    print(f"    Total Weights:    {memory_breakdown['model_weights']['total_weights']:>8.2f} GB")

    print("\n  Activations (Forward Pass):")
    print(f"    Vision Features:  {memory_breakdown['activations']['vision_activations']:>8.2f} GB")
    print(f"    Image Embeds:     {memory_breakdown['activations']['image_embeds']:>8.2f} GB")
    print(f"    Text Embeds:      {memory_breakdown['activations']['text_embeds']:>8.2f} GB")
    print(f"    Combined Embeds:  {memory_breakdown['activations']['combined_embeds']:>8.2f} GB")
    print(f"    LLM Activations:  {memory_breakdown['activations']['llm_activations']:>8.2f} GB")
    if "llm_attention_memory" in memory_breakdown["activations"]:
        print(f"      (Attention:     {memory_breakdown['activations']['llm_attention_memory']:>8.2f} GB)")
    print(f"    Total Activations:{memory_breakdown['activations']['total_activations']:>8.2f} GB")

    print("\n  Training Overhead:")
    print(f"    Gradients:        {memory_breakdown['training']['gradients']:>8.2f} GB")
    print(f"    Optimizer States: {memory_breakdown['training']['optimizer']:>8.2f} GB")
    print(f"    Total Training:   {memory_breakdown['training']['total_training']:>8.2f} GB")

    print("\n  Input Data:")
    print(f"    Pixel Values:     {memory_breakdown['data']['pixel_values']:>8.2f} GB")
    print(f"    Input IDs:        {memory_breakdown['data']['input_ids']:>8.2f} GB")
    print(f"    Total Data:       {memory_breakdown['data']['total_data']:>8.2f} GB")

    print(f"\n  Overhead:          {memory_breakdown['overhead']:>8.2f} GB")
    print(f"\n  {'TOTAL MEMORY:':<20} {memory_breakdown['total_memory_gb']:>8.2f} GB")
    print("=" * 80)


def compare_pixel_shuffle_factors(
    vision_model: str,
    llm_model: str,
    batch_size: int = 4,
    factors: list = None,
):
    """Compare memory usage for different pixel_shuffle factors."""
    if factors is None:
        # For siglip2-so400m-patch16-512: patches_per_dim = 32
        # Valid factors: 1, 2, 4, 8, 16, 32
        factors = [1, 2, 4, 8, 16]

    print("\n" + "=" * 80)
    print("PIXEL SHUFFLE FACTOR COMPARISON")
    print("=" * 80)

    results = []
    for factor in factors:
        try:
            memory = estimate_memory_usage(
                vision_model=vision_model,
                llm_model=llm_model,
                batch_size=batch_size,
                gradient_accumulation_steps=4,
                pixel_shuffle_factor=factor,
                precision="bf16",
                freeze_llm=True,
                gradient_checkpointing=True,
            )
            seq_info = memory["sequence_info"]
            results.append(
                {
                    "factor": factor,
                    "total_memory": memory["total_memory_gb"],
                    "num_patches": seq_info["num_patches_after_shuffle"],
                    "reduction": seq_info["reduction_factor"],
                    "activations": memory["activations"]["total_activations"],
                }
            )
        except Exception as e:
            print(f"  Factor {factor}: Error - {e}")
            continue

    print(f"\n{'Factor':<8} {'Patches':<10} {'Reduction':<12} {'Activations (GB)':<18} {'Total (GB)':<12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['factor']:<8} "
            f"{r['num_patches']:<10} "
            f"{r['reduction']:.1f}x{'':<8} "
            f"{r['activations']:<18.2f} "
            f"{r['total_memory']:<12.2f}"
        )

    if len(results) > 1:
        baseline = results[0]["total_memory"]
        print("\nMemory Savings vs factor=1:")
        for r in results[1:]:
            savings = baseline - r["total_memory"]
            savings_pct = (savings / baseline) * 100
            print(f"  Factor {r['factor']}: {savings:.2f} GB ({savings_pct:.1f}% reduction)")

    print("=" * 80)
    return results


if __name__ == "__main__":
    # Your configuration
    vision_model = "google/siglip2-so400m-patch16-512"
    llm_model = "Qwen/Qwen2.5-1.5B-Instruct"

    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION ANALYSIS")
    print("=" * 80)

    # Calculate optimal pixel_shuffle_factor
    seq_info = calculate_sequence_lengths(vision_model, pixel_shuffle_factor=1)
    patches_per_dim = seq_info["patches_per_dim"]

    # Find valid factors
    valid_factors = [f for f in range(1, patches_per_dim + 1) if patches_per_dim % f == 0]
    print(f"\nValid pixel_shuffle factors for {patches_per_dim}x{patches_per_dim} patches: {valid_factors}")

    # Recommended factor (largest reasonable factor)
    # Using factor=8 gives 16x16=256 patches (good balance)
    # Using factor=16 gives 2x2=4 patches (very aggressive)
    recommended_factor = 8  # Good balance between memory and quality

    # Current config (assuming auto-calculated factor)
    print("\nAnalyzing current configuration...")
    memory_current = estimate_memory_usage(
        vision_model=vision_model,
        llm_model=llm_model,
        batch_size=4,
        gradient_accumulation_steps=4,
        pixel_shuffle_factor=1,  # No shuffle
        precision="bf16",
        freeze_llm=True,
        gradient_checkpointing=True,
    )

    config = {
        "vision_model": vision_model,
        "llm_model": llm_model,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "precision": "bf16",
        "freeze_llm": True,
        "gradient_checkpointing": True,
        "pixel_shuffle_factor": 1,
    }

    print_memory_report(memory_current, config)

    # Compare with pixel shuffle
    print("\n" + "=" * 80)
    print("RECOMMENDATION: Use pixel_shuffle to reduce memory")
    print("=" * 80)

    compare_pixel_shuffle_factors(
        vision_model=vision_model,
        llm_model=llm_model,
        batch_size=4,
        factors=[1, 2, 4, 8, 16],
    )

    # Show recommended configuration
    print("\n\nRecommended Configuration:")
    print("  Use --pixel_shuffle_factor 8")
    print(f"  This reduces patches from {seq_info['num_patches_before_shuffle']} to 256")
    print("  Expected memory reduction: ~15-25%")

    memory_recommended = estimate_memory_usage(
        vision_model=vision_model,
        llm_model=llm_model,
        batch_size=4,
        gradient_accumulation_steps=4,
        pixel_shuffle_factor=8,
        precision="bf16",
        freeze_llm=True,
        gradient_checkpointing=True,
    )

    savings = memory_current["total_memory_gb"] - memory_recommended["total_memory_gb"]
    print(f"\n  Current memory: {memory_current['total_memory_gb']:.2f} GB")
    print(f"  With factor=8: {memory_recommended['total_memory_gb']:.2f} GB")
    print(f"  Savings: {savings:.2f} GB ({(savings / memory_current['total_memory_gb'] * 100):.1f}%)")
