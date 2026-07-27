
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..model.gnn_encoder import GNNEncoder
from ..model.moe_lora import CommunityMoELoraLinear
from ..model.router import StructureGroundedRouter
from ..eval.monitor import RouterMonitor
from .dataset import NNPJsonlDataset, collate_fn
from .trainer import build_optimizer_and_scheduler, train_epoch_joint

def run_stage3_joint(
    model,
    router: StructureGroundedRouter,
    moe_layers: List[CommunityMoELoraLinear],
    gnn: GNNEncoder,
    h0: torch.Tensor,
    a_hat: torch.Tensor,
    communities: List[List[int]],
    node_index: Dict[int, int],
    tokenizer,
    nnp_instances: List,
    device: str = "cpu",
    num_epochs: int = 1,
    batch_size: int = 4,
    lr: float = 1e-3,
    max_length: int = 256,
    alpha: float = 1.0,
    beta: float = 0.01,
    top_kappa: int = 2,
    grad_clip: float = 1.0,
    monitor: Optional[RouterMonitor] = None,
) -> None:
    for layer in moe_layers:
        if layer.use_global_expert:
            layer.lora_A_global.requires_grad_(True)
            layer.lora_B_global.requires_grad_(True)
        layer.lora_A_local.requires_grad_(True)
        layer.lora_B_local.requires_grad_(True)
    for p in router.parameters():
        p.requires_grad_(True)
    for p in gnn.parameters():
        p.requires_grad_(True)

    def _community_of(r):
        return r.community_id if hasattr(r, "community_id") else r.get("community")

    def _as_record(r):
        return r.as_dict() if hasattr(r, "as_dict") else r

    records = [_as_record(r) for r in nnp_instances if _community_of(r) is not None]
    if not records:
        print("[joint] no community-labeled NNP instances available; skipping Stage 3.")
        return

    ds = NNPJsonlDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, collate_fn=collate_fn)

    h0 = h0.to(device)
    a_hat = a_hat.to(device)
    gnn = gnn.to(device)

    def community_embeddings_fn() -> torch.Tensor:
        z = gnn(h0, a_hat)
        return GNNEncoder.community_embeddings(z, communities, node_index)

    num_communities = len(communities)
    trainable = (
        [p for p in model.parameters() if p.requires_grad]
        + list(router.parameters())
        + list(gnn.parameters())
    )
    optimizer, scheduler = build_optimizer_and_scheduler(trainable, lr, len(loader) * num_epochs)

    for epoch in range(num_epochs):
        avg = train_epoch_joint(
            model,
            router,
            moe_layers,
            community_embeddings_fn,
            loader,
            optimizer,
            scheduler,
            device,
            num_communities,
            top_kappa,
            alpha=alpha,
            beta=beta,
            grad_clip=grad_clip,
            monitor=monitor,
        )
        print(f"[joint] epoch {epoch + 1}/{num_epochs} avg_loss={avg:.4f}")
        if monitor is not None and monitor.collapse_detected():
            print(f"[joint] WARNING: router collapse detected at epoch {epoch + 1}")
