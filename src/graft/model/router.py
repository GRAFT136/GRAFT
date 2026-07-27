
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureGroundedRouter(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        community_dim: int,
        routing_dim: int = 128,
        top_kappa: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.community_dim = community_dim
        self.routing_dim = routing_dim
        self.top_kappa = top_kappa

        self.W_q = nn.Linear(hidden_size, routing_dim, bias=False)
        self.W_c = nn.Linear(community_dim, routing_dim, bias=False)

    def scores(self, u: torch.Tensor, community_embeddings: torch.Tensor) -> torch.Tensor:
        q = self.W_q(u)
        c = self.W_c(community_embeddings)
        s = (q @ c.T) / math.sqrt(self.routing_dim)
        return s

    def forward(
        self,
        u: torch.Tensor,
        community_embeddings: torch.Tensor,
        top_kappa: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kappa = top_kappa or self.top_kappa
        s = self.scores(u, community_embeddings)
        K = s.shape[-1]
        kappa = min(kappa, K)

        topk_scores, topk_idx = s.topk(kappa, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)
        return weights, topk_idx, s
