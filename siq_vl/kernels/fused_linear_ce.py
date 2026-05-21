"""Fused linear cross-entropy using TileGym's cuTile CE kernel.

Replaces the standard `lm_head(hidden) → CE(logits, labels)` pattern with a
chunked computation that never materializes the full [BT, V] logits tensor.

Forward: chunked matmul + TileGym _ce_cutile kernel (online softmax over vocab tiles)
Backward: reuses in-place softmax probs from forward for grad_hidden = (probs - one_hot) @ W

Monkey-patches Qwen2ForCausalLM.forward to use this fused path during training.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from tilegym.ops.cutile.experimental.fused_linear_cross_entropy import _ce_cutile

_CHUNK_SIZE = 4096


class FusedLinearCrossEntropy(torch.autograd.Function):
    """Autograd wrapper using TileGym's cuTile CE kernel for forward."""

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

        # After _ce_cutile, logits_chunk is overwritten with softmax probs in-place.
        # We store these for backward.
        grad_logits_chunks = []

        for i in range(num_chunks):
            s, e = i * chunk_size, min((i + 1) * chunk_size, bt)
            x_chunk = hidden_states[s:e].detach()
            t_chunk = target[s:e]

            # GEMM: logits = x @ W^T (detached buffer for in-place CE kernel)
            logits_chunk = x_chunk @ weight.T
            loss_chunk = loss_all[s:e]

            valid_mask = t_chunk != ignore_index
            n_valid_chunk = valid_mask.sum().item()

            if n_valid_chunk > 0:
                # TileGym cuTile kernel: computes loss AND overwrites logits with softmax probs
                _ce_cutile(logits_chunk, t_chunk, loss_chunk, ignore_index)

                # logits_chunk now contains softmax probs — compute grad_logits = probs - one_hot
                safe_target = t_chunk.clamp(min=0)
                logits_chunk[torch.arange(logits_chunk.shape[0], device=logits_chunk.device), safe_target] -= 1.0
                logits_chunk[~valid_mask] = 0.0
                grad_logits_chunks.append(logits_chunk.to(hidden_states.dtype))
                n_valid += n_valid_chunk
            else:
                loss_chunk.zero_()
                grad_logits_chunks.append(None)

        # Reduction: mean over valid tokens
        if n_valid > 0:
            loss = loss_all.sum() / n_valid
        else:
            loss = loss_all.sum()

        ctx.save_for_backward(hidden_states, weight)
        ctx.grad_logits_chunks = grad_logits_chunks
        ctx.chunk_size = chunk_size
        ctx.n_valid = n_valid
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight = ctx.saved_tensors
        grad_logits_chunks = ctx.grad_logits_chunks
        chunk_size = ctx.chunk_size
        n_valid = ctx.n_valid
        bt = hidden_states.shape[0]

        scale = grad_output / max(n_valid, 1)

        grad_hidden = torch.empty_like(hidden_states)
        grad_weight: Tensor | None = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)

        for i, grad_logits in enumerate(grad_logits_chunks):
            s = i * chunk_size
            e = min(s + chunk_size, bt)

            if grad_logits is None:
                grad_hidden[s:e] = 0.0
                continue

            # grad_logits already contains (probs - one_hot) from forward
            scaled_grad = grad_logits * scale
            grad_hidden[s:e] = scaled_grad @ weight

            if grad_weight is not None:
                grad_weight.addmm_(scaled_grad.T, hidden_states[s:e])

        ctx.grad_logits_chunks = None
        return grad_hidden, grad_weight, None, None


def fused_linear_cross_entropy(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Fused linear + cross-entropy using TileGym cuTile kernel.

    Never materializes full [BT, V] logits — processes in chunks.
    The cuTile kernel computes loss + softmax probs in a single pass.

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
