
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

def routing_supervision_loss(
    scores: torch.Tensor,
    community_labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(scores, community_labels)

def load_balancing_loss(
    scores: torch.Tensor,
    num_communities: int,
    top_kappa: int,
) -> torch.Tensor:
    probs = F.softmax(scores, dim=-1)
    p = probs.mean(dim=0)

    top_kappa = min(top_kappa, num_communities)
    topk_idx = scores.topk(top_kappa, dim=-1).indices
    one_hot = F.one_hot(topk_idx, num_classes=num_communities).float()
    f = one_hot.sum(dim=1).mean(dim=0)

    return num_communities * (f * p).sum()

def compute_total_loss(
    nnp_loss: torch.Tensor,
    scores: Optional[torch.Tensor],
    community_labels: Optional[torch.Tensor],
    num_communities: int,
    top_kappa: int,
    alpha: float = 1.0,
    beta: float = 0.01,
) -> torch.Tensor:
    total = nnp_loss
    if scores is not None:
        bal = load_balancing_loss(scores, num_communities, top_kappa)
        total = total + beta * bal
        if community_labels is not None:
            route = routing_supervision_loss(scores, community_labels)
            total = total + alpha * route
    return total
