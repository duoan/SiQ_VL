"""Register TileGym's FA4 as a custom attention backend in HuggingFace Transformers.

TileGym provides the production FlashAttention-4 implementation using cuTile DSL,
optimized for Blackwell GPUs (sm_120) with:
- Native GQA support (no repeat_kv expansion — key architectural win)
- Autotuning tile sizes per sequence length
- K-loop split + ProgramId remapping + fast math (FTZ + APPROX)
- 12-27% faster than SDPA at the kernel level

Two backends registered:
- "cutile": Uses TileGym's forward-only fmha (best for frozen modules / inference)
- "cutile_training": Uses autograd.Function wrapper (FA4 fwd + SDPA bwd for unfrozen modules)

Usage:
    from siq_vl.kernels.attention_backend import register_cutile_attention
    register_cutile_attention()
    model = AutoModel.from_pretrained(..., attn_implementation="cutile")
"""

from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_REGISTERED = False


class TileGymFlashAttentionWithGrad(torch.autograd.Function):
    """Autograd wrapper for TileGym FA4 when gradients ARE needed (Stage 2).

    Forward: TileGym fmha (native GQA, autotuned, K-loop split)
    Backward: PyTorch SDPA (until TileGym ships native backward)
    """

    @staticmethod
    def forward(ctx, query, key, value, is_causal, num_key_value_groups):
        from tilegym.ops import fmha
        output = fmha(query, key, value, is_causal=is_causal)
        ctx.save_for_backward(query, key, value)
        ctx.is_causal = is_causal
        ctx.num_key_value_groups = num_key_value_groups
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        if ctx.num_key_value_groups > 1:
            from tilegym.ops.attn_interface import repeat_kv
            key_expanded = repeat_kv(key, ctx.num_key_value_groups)
            value_expanded = repeat_kv(value, ctx.num_key_value_groups)
        else:
            key_expanded = key
            value_expanded = value

        with torch.enable_grad():
            q = query.detach().requires_grad_(True)
            k = key_expanded.detach().requires_grad_(True)
            v = value_expanded.detach().requires_grad_(True)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=ctx.is_causal
            )
            out.backward(grad_output)

        if ctx.num_key_value_groups > 1:
            B, num_kv_heads, N, D = key.shape
            dk = k.grad.view(B, num_kv_heads, ctx.num_key_value_groups, N, D).sum(dim=2)
            dv = v.grad.view(B, num_kv_heads, ctx.num_key_value_groups, N, D).sum(dim=2)
        else:
            dk = k.grad
            dv = v.grad

        return q.grad, dk, dv, None, None


def _cutile_forward_only(
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
    """TileGym FA4 backend — forward only (for frozen modules / inference).

    Best performance: no autograd overhead, native GQA, autotuned tiles.
    Use when the attention module's parameters don't require gradients.
    """
    from tilegym.ops import fmha

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Fall back to SDPA for 4D masks (packing with FlexAttention)
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
            from tilegym.ops.attn_interface import repeat_kv
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    # Single-token decode
    if query.shape[2] == 1:
        from tilegym.ops import fmha_decode
        sm_scale = scaling if scaling else (1.0 / (query.shape[-1] ** 0.5))
        attn_output = fmha_decode(query, key, value, sm_scale=sm_scale)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    # Main path: TileGym FA4 (native GQA, autotuned)
    attn_output = fmha(query, key, value, is_causal=is_causal)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _cutile_with_backward(
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
    """TileGym FA4 backend with backward support (for unfrozen/trainable attention).

    Uses FA4 forward + SDPA backward via autograd.Function.
    """
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Fall back to SDPA for 4D masks or single-token decode
    if attention_mask is not None or query.shape[2] == 1:
        return _cutile_forward_only(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    num_key_value_groups = getattr(module, "num_key_value_groups", 1)

    if query.requires_grad:
        attn_output = TileGymFlashAttentionWithGrad.apply(query, key, value, is_causal, num_key_value_groups)
    else:
        from tilegym.ops import fmha
        attn_output = fmha(query, key, value, is_causal=is_causal)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def register_cutile_attention():
    """Register TileGym FA4 attention backends. Call once at startup.

    Registers two backends:
    - "cutile": Forward-only (optimal for Stage 1 frozen LLM, inference)
    - "cutile_training": With backward (for Stage 2 unfrozen LLM)
    """
    global _REGISTERED
    if _REGISTERED:
        return

    ALL_ATTENTION_FUNCTIONS["cutile"] = _cutile_forward_only
    ALL_ATTENTION_FUNCTIONS["cutile_training"] = _cutile_with_backward
    _REGISTERED = True

