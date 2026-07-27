
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .moe_lora import CommunityMoELoraLinear

_MLP_PROJ_NAMES: Dict[str, List[str]] = {
    "Qwen2ForCausalLM": ["gate_proj", "up_proj", "down_proj"],
    "Qwen2_5ForCausalLM": ["gate_proj", "up_proj", "down_proj"],
    "LlamaForCausalLM": ["gate_proj", "up_proj", "down_proj"],
    "MistralForCausalLM": ["gate_proj", "up_proj", "down_proj"],
    "_default": ["gate_proj", "up_proj", "down_proj"],
}

def inject_moe_lora(
    model: nn.Module,
    rank: int = 16,
    lora_alpha: float = 32.0,
    num_communities: int = 8,
    use_global_expert: bool = True,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    arch = type(model).__name__
    if target_modules is None:
        target_modules = _MLP_PROJ_NAMES.get(arch, _MLP_PROJ_NAMES["_default"])

    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for name, module in list(model.named_modules()):
        parent, child_name = _get_parent_and_child(model, name)
        if parent is None or not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(f".{proj}") or name == proj for proj in target_modules):
            continue

        moe_layer = CommunityMoELoraLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            base_weight=module.weight.data,
            base_bias=module.bias.data if module.bias is not None else None,
            rank=rank,
            lora_alpha=lora_alpha,
            num_communities=num_communities,
            use_global_expert=use_global_expert,
        )
        moe_layer = moe_layer.to(module.weight.device)
        setattr(parent, child_name, moe_layer)
        replaced += 1

    print(
        f"[injection] Replaced {replaced} Linear layers -> CommunityMoELoraLinear "
        f"(arch={arch}, use_global={use_global_expert}, num_communities={num_communities}, rank={rank})"
    )
    _print_trainable_params(model)
    return model

def get_moe_layers(model: nn.Module) -> List[CommunityMoELoraLinear]:
    return [m for m in model.modules() if isinstance(m, CommunityMoELoraLinear)]

class RouterDecisionContext:

    def __init__(
        self,
        moe_layers: List[CommunityMoELoraLinear],
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        self.moe_layers = moe_layers
        self.weights = weights
        self.indices = indices
        self._orig_forwards: Dict[int, callable] = {}

    def __enter__(self) -> "RouterDecisionContext":
        for layer in self.moe_layers:
            self._orig_forwards[id(layer)] = layer.forward
            layer.forward = self._make_forward(layer)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for layer in self.moe_layers:
            layer.forward = self._orig_forwards[id(layer)]
        self._orig_forwards.clear()

    def _make_forward(self, layer: CommunityMoELoraLinear):
        orig = self._orig_forwards[id(layer)]
        weights, indices = self.weights, self.indices

        def _patched(x: torch.Tensor) -> torch.Tensor:
            return orig(x, router_weights=weights, router_indices=indices)

        return _patched

def _get_parent_and_child(model: nn.Module, full_name: str) -> Tuple[Optional[nn.Module], str]:
    parts = full_name.split(".")
    if len(parts) == 1:
        return model, full_name
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None, ""
    return parent, parts[-1]

def get_hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, attr):
                return getattr(cfg, attr)
    raise ValueError(
        "Cannot determine hidden_size from model.config. Pass hidden_size explicitly."
    )

def _print_trainable_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[injection] Total params: {total:,} | Trainable (LoRA): {trainable:,}")
