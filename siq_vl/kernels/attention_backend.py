"""Register cuTile as a custom attention backend in HuggingFace Transformers.

Usage:
    from siq_vl.kernels.attention_backend import register_cutile_attention

    register_cutile_attention()  # Call once at startup

    # Then in model init:
    model = AutoModel.from_pretrained(..., attn_implementation="cutile")
"""

from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_REGISTERED = False


def cutile_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """cuTile Flash Attention backend for HuggingFace Transformers.

    Uses cuTile forward kernel when gradients are not needed (inference/eval).
    Falls back to SDPA during training (cuTile backward has overhead from
    recomputing the forward pass; SDPA handles both fwd+bwd efficiently).
    """
    from siq_vl.kernels.cutile_attention import CuTileFlashAttention

    # Handle GQA key/value expansion
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Determine causality
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Fall back to SDPA for complex attention masks (4D masks from packing, etc.)
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=False,
        )
    elif not query.requires_grad:
        # Inference path: use cuTile (17% faster on short sequences)
        if query.shape[2] == 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                dropout_p=dropout,
                scale=scaling,
                is_causal=False,
            )
        else:
            attn_output = CuTileFlashAttention.apply(query, key, value, is_causal)
    else:
        # Training path: use SDPA (handles fwd+bwd efficiently in one autograd op)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def register_cutile_attention():
    """Register cuTile as an attention backend. Call once at startup."""
    global _REGISTERED
    if _REGISTERED:
        return

    ALL_ATTENTION_FUNCTIONS["cutile"] = cutile_attention_forward
    _REGISTERED = True
