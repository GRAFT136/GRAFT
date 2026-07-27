
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..model.injection import RouterDecisionContext
from ..model.moe_lora import CommunityMoELoraLinear

def build_optimizer_and_scheduler(
    trainable_params: List[nn.Parameter], lr: float, num_steps: int, weight_decay: float = 0.01
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup_steps = max(1, num_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, max(num_steps, 1))
    return optimizer, scheduler

def forced_route(batch_size: int, expert_index: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.ones(batch_size, 1, device=device)
    indices = torch.full((batch_size, 1), expert_index, dtype=torch.long, device=device)
    return weights, indices

@torch.no_grad()
def compute_query_hidden_state(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_index: Optional[int] = None,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states
    idx = layer_index if layer_index is not None else len(hidden_states) // 2
    h = hidden_states[idx]
    seq_lens = attention_mask.sum(dim=1) - 1
    u = h[torch.arange(h.shape[0], device=h.device), seq_lens]
    return u

def train_epoch_forced(
    model,
    moe_layers: List[CommunityMoELoraLinear],
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    expert_index: Optional[int],
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        B = input_ids.shape[0]

        if expert_index is not None:
            weights, indices = forced_route(B, expert_index, device)
            ctx = RouterDecisionContext(moe_layers, weights, indices)
        else:
            ctx = _NullContext()

        with ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

def train_epoch_joint(
    model,
    router,
    moe_layers: List[CommunityMoELoraLinear],
    community_embeddings_fn,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    num_communities: int,
    top_kappa: int,
    alpha: float = 1.0,
    beta: float = 0.01,
    grad_clip: float = 1.0,
    monitor=None,
    routing_layer_index: Optional[int] = None,
) -> float:
    from ..model.losses import compute_total_loss

    model.train()
    router.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        community_labels = batch.get("community")
        if community_labels is not None:
            community_labels = community_labels.to(device)

        u = compute_query_hidden_state(model, input_ids, attention_mask, routing_layer_index)

        community_embeddings = community_embeddings_fn()
        weights, indices, scores = router(u, community_embeddings.to(device), top_kappa)

        with RouterDecisionContext(moe_layers, weights, indices):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nnp_loss = outputs.loss

        total = compute_total_loss(
            nnp_loss, scores, community_labels, num_communities, top_kappa, alpha, beta
        )

        optimizer.zero_grad()
        total.backward()
        trainable = [p for p in model.parameters() if p.requires_grad] + list(router.parameters())
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += total.item()
        if monitor is not None:
            monitor.update(indices.detach(), scores.detach())

    return total_loss / max(1, len(dataloader))

class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
