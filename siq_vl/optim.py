"""Optimizer utilities for SiQ-VL training.

Provides:
- CompositeOptimizer: wraps multiple optimizers as a single one for HF Trainer
- create_muon_optimizer: builds Muon (2D hidden weights) + AdamW (rest)
- create_adamw_optimizer: builds fused AdamW
"""

from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW, Muon


class CompositeOptimizer(torch.optim.Optimizer):
    """Wraps multiple optimizers so HuggingFace Trainer sees a single optimizer.

    Used to combine Muon (for >=2D hidden weights) with AdamW (for embeddings,
    heads, norms, biases) since Muon only supports 2D parameters.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self._optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
        self.defaults = optimizers[0].defaults
        self.state = {}

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self._optimizers:
            opt.step(closure=closure)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self._optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self._optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


def _partition_params(model: nn.Module):
    """Split trainable parameters into muon-eligible (>=2D hidden) and adamw (rest).

    Returns:
        (muon_params, adamw_params) - two lists of parameter tensors.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "embed" in name or "wte" in name or "wpe" in name
        is_head = "lm_head" in name or "score" in name
        is_norm = "norm" in name or "ln_" in name or "layernorm" in name

        if p.ndim >= 2 and not is_embedding and not is_head and not is_norm:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params


def create_muon_optimizer(
    model: nn.Module,
    muon_lr: float = 5e-4,
    muon_momentum: float = 0.95,
    adamw_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> CompositeOptimizer:
    """Create a composite Muon + AdamW optimizer.

    Muon handles >=2D hidden layer weights via Newton-Schulz orthogonalization.
    AdamW handles embeddings, heads, norms, and biases.

    Args:
        model: The model to optimize.
        muon_lr: Learning rate for Muon (2D hidden weights).
        muon_momentum: Momentum for Muon.
        adamw_lr: Learning rate for AdamW (embeddings, norms, etc).
        weight_decay: Weight decay for both optimizers.

    Returns:
        CompositeOptimizer wrapping [Muon, AdamW].
    """
    muon_params, adamw_params = _partition_params(model)

    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
    )
    adamw_opt = AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    return CompositeOptimizer([muon_opt, adamw_opt]), len(muon_params), len(adamw_params)


def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
) -> AdamW:
    """Create a fused AdamW optimizer for all trainable parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
