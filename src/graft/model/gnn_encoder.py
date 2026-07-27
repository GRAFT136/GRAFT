
from __future__ import annotations

from typing import List, Optional

import networkx as nx
import torch
import torch.nn as nn

def build_normalized_adjacency(G: nx.Graph, num_nodes: int, node_index: dict) -> torch.Tensor:
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    G_und = G.to_undirected() if G.is_directed() else G
    for u, v in G_und.edges():
        if u in node_index and v in node_index:
            i, j = node_index[u], node_index[v]
            A[i, j] = 1.0
            A[j, i] = 1.0
    A = A + torch.eye(num_nodes)
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

@torch.no_grad()
def init_node_features_from_llm(
    texts: List[str],
    embedding_layer: nn.Embedding,
    tokenizer,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    max_length: int = 64,
) -> torch.Tensor:
    device = device or next(embedding_layer.parameters()).device
    rows = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        emb = embedding_layer(enc["input_ids"])
        mask = enc["attention_mask"].unsqueeze(-1).to(emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        rows.append(pooled.float().cpu())
    return torch.cat(rows, dim=0)

class GNNLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.activation = nn.ReLU() if activation else nn.Identity()

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        agg = a_hat @ h
        return self.activation(self.linear(agg))

class GNNEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        assert num_layers >= 1
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [
                GNNLayer(dims[i], dims[i + 1], activation=(i < num_layers - 1))
                for i in range(num_layers)
            ]
        )
        self.out_dim = out_dim

    def forward(self, h0: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = h0
        for layer in self.layers:
            h = layer(h, a_hat.to(h.dtype))
        return h

    @staticmethod
    def community_embeddings(
        node_embeddings: torch.Tensor,
        communities: List[List[int]],
        node_index: dict,
    ) -> torch.Tensor:
        K = len(communities)
        d_L = node_embeddings.shape[-1]
        out = torch.zeros(K, d_L, dtype=node_embeddings.dtype)
        for k, members in enumerate(communities):
            rows = [node_index[m] for m in members if m in node_index]
            if rows:
                out[k] = node_embeddings[rows].mean(dim=0)
        return out
