"""Fused linear cross-entropy using TileGym's cuTile CE kernel.

Replaces the standard `lm_head(hidden) → CE(logits, labels)` pattern with a
chunked computation that never materializes the full [BT, V] logits tensor.

Forward: chunked matmul + TileGym _ce_cutile kernel (online softmax over vocab tiles)
         + computes grad_hidden/grad_weight in the same loop (fused_linear_jsd pattern)
Backward: trivial O(1) — just scales precomputed gradients by grad_output

Key memory optimization: grad_logits (C, V) is NEVER stored across chunks.
Each chunk's (C, V) grad_logits is computed, used for backward GEMMs, then freed
within the same loop iteration. Total stored: grad_hidden (BT, H) + grad_weight (V, H).
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from tilegym.ops.cutile.experimental.fused_linear_cross_entropy import _ce_cutile

_CHUNK_SIZE = 4096


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Fused linear CE: forward computes all grads, backward is O(1) scalar scale."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        target: Tensor,
        ignore_index: int,
    ) -> Tensor:
        bt, h = hidden_states.shape
        vocab_size = weight.shape[0]
        chunk_size = min(_CHUNK_SIZE, bt)
        num_chunks = (bt + chunk_size - 1) // chunk_size

        loss_all = torch.empty(bt, device=hidden_states.device, dtype=torch.float32)
        n_valid = 0

        # Precompute gradients in forward (fused_linear_jsd pattern).
        # Each chunk's grad_logits (C, V) lives only within one iteration.
        grad_hidden = torch.zeros(bt, h, device=hidden_states.device, dtype=hidden_states.dtype)
        grad_weight = torch.zeros(vocab_size, h, device=hidden_states.device, dtype=hidden_states.dtype)

        for i in range(num_chunks):
            s, e = i * chunk_size, min((i + 1) * chunk_size, bt)
            x_chunk = hidden_states[s:e]
            t_chunk = target[s:e]
            loss_chunk = loss_all[s:e]

            valid_mask = t_chunk != ignore_index
            n_valid_chunk = valid_mask.sum().item()

            if n_valid_chunk == 0:
                loss_chunk.zero_()
                continue

            # GEMM 1: logits = x @ W^T (chunk_size, V)
            logits_chunk = x_chunk.detach() @ weight.T

            # cuTile kernel: loss + overwrite logits with softmax probs (in-place)
            _ce_cutile(logits_chunk, t_chunk, loss_chunk, ignore_index)

            # logits_chunk is now softmax probs — compute grad_logits = probs - one_hot
            safe_target = t_chunk.clamp(min=0)
            rows = torch.arange(logits_chunk.shape[0], device=logits_chunk.device)
            logits_chunk[rows, safe_target] -= 1.0
            logits_chunk[~valid_mask] = 0.0
            # logits_chunk is now grad_logits (C, V) — use immediately for backward GEMMs

            # GEMM 2: grad_hidden_chunk = grad_logits @ weight  (C,V) @ (V,H) -> (C,H)
            grad_hidden[s:e] = logits_chunk.to(hidden_states.dtype) @ weight

            # GEMM 3: grad_weight += grad_logits.T @ hidden  (V,C) @ (C,H) -> (V,H)
            grad_weight.addmm_(logits_chunk.to(hidden_states.dtype).T, x_chunk)

            # logits_chunk (C, V) is freed here — never stored!

            n_valid += n_valid_chunk

        # Loss reduction
        if n_valid > 0:
            loss = loss_all.sum() / n_valid
            # Scale grads by 1/n_valid (the loss reduction factor)
            inv_n = 1.0 / n_valid
            grad_hidden.mul_(inv_n)
            grad_weight.mul_(inv_n)
        else:
            loss = loss_all.sum()

        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.has_weight_grad = weight.requires_grad
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        # O(1) backward: just scale by upstream gradient
        return grad_hidden * grad_output, grad_weight * grad_output if ctx.has_weight_grad else None, None, None


def fused_linear_cross_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Fused linear + cross-entropy using TileGym cuTile kernel.

    Never materializes full [BT, V] logits — processes in chunks.
    All gradients computed in forward; backward is O(1).

    Args:
        hidden_states: (BT, H) flattened hidden states
        weight: (V, H) lm_head weight
        target: (BT,) target labels (already shifted)
        ignore_index: label to ignore (default: -100)
    """
    return FusedLinearCrossEntropy.apply(hidden_states, weight, target, ignore_index)


def patch_qwen2_fused_linear_ce():
    """Monkey-patch Qwen2ForCausalLM.forward to use TileGym fused linear cross-entropy.

    This replaces lm_head(hidden) → CE(logits, labels) with a single fused op
    that computes the loss without materializing the full logits tensor.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast

    def _fused_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        loss = None
        if labels is not None and self.training:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            bt = shift_hidden.shape[0] * shift_hidden.shape[1]
            h = shift_hidden.shape[-1]

            loss = fused_linear_cross_entropy(
                shift_hidden.view(bt, h),
                self.lm_head.weight,
                shift_labels.view(bt),
                ignore_index=-100,
            )
            logits = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    Qwen2ForCausalLM.forward = _fused_forward
