
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..model.moe_lora import CommunityMoELoraLinear
from .dataset import NNPJsonlDataset, collate_fn
from .trainer import build_optimizer_and_scheduler, train_epoch_forced

def _set_global_only_trainable(moe_layers: List[CommunityMoELoraLinear]) -> None:
    for layer in moe_layers:
        if layer.use_global_expert:
            layer.lora_A_global.requires_grad_(True)
            layer.lora_B_global.requires_grad_(True)
        layer.lora_A_local.requires_grad_(False)
        layer.lora_B_local.requires_grad_(False)

def _set_single_expert_trainable(moe_layers: List[CommunityMoELoraLinear], expert_index: int):
    handles = []
    for layer in moe_layers:
        if layer.use_global_expert:
            layer.lora_A_global.requires_grad_(False)
            layer.lora_B_global.requires_grad_(False)
        layer.lora_A_local.requires_grad_(True)
        layer.lora_B_local.requires_grad_(True)

        def _mask(grad: torch.Tensor, idx: int = expert_index) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask[idx] = 1.0
            return grad * mask

        handles.append(layer.lora_A_local.register_hook(_mask))
        handles.append(layer.lora_B_local.register_hook(_mask))
    return handles

def run_stage2_warmup(
    model,
    tokenizer,
    moe_layers: List[CommunityMoELoraLinear],
    nnp_instances: List,
    nnp_instances_per_community: Dict[int, List],
    device: str = "cpu",
    global_warmup_epochs: int = 1,
    expert_warmup_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_length: int = 256,
    grad_clip: float = 1.0,
) -> None:

    def _make_loader(records, bs):
        recs = [r.as_dict() if hasattr(r, "as_dict") else r for r in records]
        if not recs:
            return None
        ds = NNPJsonlDataset(recs, tokenizer, max_length=max_length)
        return DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True, collate_fn=collate_fn)

    any_global = any(layer.use_global_expert for layer in moe_layers)
    if any_global:
        _set_global_only_trainable(moe_layers)
        loader = _make_loader(nnp_instances, batch_size)
        if loader is not None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer, scheduler = build_optimizer_and_scheduler(
                trainable, lr, len(loader) * global_warmup_epochs
            )
            for epoch in range(global_warmup_epochs):
                avg = train_epoch_forced(
                    model, moe_layers, loader, optimizer, scheduler, device, expert_index=None, grad_clip=grad_clip
                )
                print(f"[warmup/global] epoch {epoch + 1}/{global_warmup_epochs} avg_loss={avg:.4f}")

    num_communities = moe_layers[0].num_communities if moe_layers else 0
    for k in range(num_communities):
        records = nnp_instances_per_community.get(k, [])
        loader = _make_loader(records, batch_size)
        if loader is None:
            print(f"[warmup/community_{k}] no NNP instances, skipping")
            continue

        handles = _set_single_expert_trainable(moe_layers, k)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer, scheduler = build_optimizer_and_scheduler(
            trainable, lr, len(loader) * expert_warmup_epochs
        )
        for epoch in range(expert_warmup_epochs):
            avg = train_epoch_forced(
                model, moe_layers, loader, optimizer, scheduler, device, expert_index=k, grad_clip=grad_clip
            )
            print(f"[warmup/community_{k}] epoch {epoch + 1}/{expert_warmup_epochs} avg_loss={avg:.4f}")

        for h in handles:
            h.remove()

    for layer in moe_layers:
        if layer.use_global_expert:
            layer.lora_A_global.requires_grad_(True)
            layer.lora_B_global.requires_grad_(True)
        layer.lora_A_local.requires_grad_(True)
        layer.lora_B_local.requires_grad_(True)
