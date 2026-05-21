"""Stage 2 benchmark: unfrozen text model comparison.

Benchmarks:
- Full finetune vs LoRA (r=16)
- Gradient checkpointing impact
- Attention backends (SDPA vs cuTile_training)
- Sequence length scaling
"""

import argparse
import time

import torch
from peft import LoraConfig, get_peft_model

from siq_vl.kernels.attention_backend import register_cutile_attention
from siq_vl.model.configuration import get_siq_vl_config
from siq_vl.model.modeling import (
    SiQ_VLForCausalLM,
    SiQ_VLTextModel,
    SiQ_VLVisionModel,
    _apply_liger_kernel,
)


def build_model(
    attn_impl: str = "sdpa",
    use_lora: bool = False,
    lora_r: int = 16,
    grad_ckpt: bool = False,
):
    _apply_liger_kernel()
    config = get_siq_vl_config(
        text_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        vision_model_name_or_path="google/siglip2-base-patch16-224",
    )
    model = SiQ_VLForCausalLM(config)
    model.text_model = SiQ_VLTextModel.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model.vision_model = SiQ_VLVisionModel.from_pretrained(
        "google/siglip2-base-patch16-224",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.projector = model.projector.to(torch.bfloat16)
    model.freez_vision_model()

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.text_model = get_peft_model(model.text_model, lora_config)
    else:
        pass  # full finetune — all text params trainable

    model = model.to("cuda").train()
    if grad_ckpt:
        if use_lora:
            model.text_model.base_model.model.gradient_checkpointing_enable()
        else:
            model.text_model.gradient_checkpointing_enable()
    return model


def benchmark(model, batch_size=4, seq_len=1024, steps=15, warmup=5):
    image_token_id = model.config.image_token_index
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device="cuda")
    input_ids[:, :49] = image_token_id
    pixel_values = torch.randn(batch_size, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    attention_mask = torch.ones(batch_size, seq_len, device="cuda", dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :49] = -100

    batch = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
    )

    for _ in range(warmup):
        out = model(**batch)
        out.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(**batch)
        out.loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = elapsed / steps * 1000
    tok_per_sec = batch_size * seq_len / (elapsed / steps)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    return ms_per_step, tok_per_sec, peak_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    register_cutile_attention()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    print(f"{'Mode':<40} {'N':<6} {'ms/step':<10} {'tok/s':<10} {'VRAM GB':<8}")
    print("-" * 78)

    configs = [
        ("LoRA r=16, SDPA", dict(attn_impl="sdpa", use_lora=True)),
        ("LoRA r=16, cuTile_training", dict(attn_impl="cutile_training", use_lora=True)),
        ("Full FT, SDPA", dict(attn_impl="sdpa", use_lora=False)),
        ("Full FT + grad_ckpt, SDPA", dict(attn_impl="sdpa", use_lora=False, grad_ckpt=True)),
    ]

    for name, kwargs in configs:
        model = build_model(**kwargs)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for seq_len in args.seq_lens:
            try:
                ms, toks, mem = benchmark(
                    model, args.batch_size, seq_len, args.steps, args.warmup
                )
                print(f"{name:<40} {seq_len:<6} {ms:<10.1f} {toks:<10.0f} {mem:<8.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{name:<40} {seq_len:<6} {'OOM':<10}")
                torch.cuda.empty_cache()
        print(f"  (trainable: {trainable:,})")
        print()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
