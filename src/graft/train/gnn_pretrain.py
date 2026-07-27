
from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..model.gnn_encoder import GNNEncoder

def pretrain_gnn(
    gnn: GNNEncoder,
    h0: torch.Tensor,
    a_hat: torch.Tensor,
    edges: List[Tuple[int, int]],
    num_nodes: int,
    epochs: int = 20,
    lr: float = 1e-2,
    num_neg_per_pos: int = 1,
    seed: int = 42,
) -> None:
    if not edges or num_nodes < 2:
        return

    rng = random.Random(seed)
    edge_set = set(edges)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    for epoch in range(epochs):
        z = gnn(h0, a_hat)

        pos_u = torch.tensor([e[0] for e in edges], dtype=torch.long)
        pos_v = torch.tensor([e[1] for e in edges], dtype=torch.long)

        neg_pairs = []
        while len(neg_pairs) < len(edges) * num_neg_per_pos:
            u = rng.randrange(num_nodes)
            v = rng.randrange(num_nodes)
            if u != v and (u, v) not in edge_set:
                neg_pairs.append((u, v))
        neg_u = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long)
        neg_v = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long)

        pos_scores = (z[pos_u] * z[pos_v]).sum(dim=-1)
        neg_scores = (z[neg_u] * z[neg_v]).sum(dim=-1)

        logits = torch.cat([pos_scores, neg_scores])
        targets = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[gnn_pretrain] finished {epochs} epochs, final link-prediction loss={loss.item():.4f}")
