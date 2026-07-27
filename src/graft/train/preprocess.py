
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import torch
import torch.nn.functional as F

from ..data.nnp_dataset import NNPInstance, build_nnp_instances, build_nnp_instances_for_communities
from ..data.partition import detect_communities
from .gnn_pretrain import pretrain_gnn
from ..model.gnn_encoder import (
    GNNEncoder,
    build_normalized_adjacency,
    init_node_features_from_llm,
)

@dataclass
class Stage1Output:
    node_to_community: Dict[int, int]
    communities: List[List[int]]
    node_index: Dict[int, int]
    nnp_instances: List[NNPInstance]
    nnp_instances_per_community: Dict[int, List[NNPInstance]]
    gnn: GNNEncoder
    h0: torch.Tensor
    a_hat: torch.Tensor
    community_embeddings: torch.Tensor

    @property
    def num_communities(self) -> int:
        return len(self.communities)

def run_stage1_preprocessing(
    graph_data: Dict,
    model,
    tokenizer,
    partition_method: str = "louvain",
    min_community_size: int = 10,
    node_class_map: Optional[Dict[int, str]] = None,
    ground_truth_map: Optional[Dict[int, int]] = None,
    d_max: int = 32,
    gnn_hidden_dim: int = 128,
    gnn_out_dim: int = 128,
    gnn_num_layers: int = 2,
    gnn_pretrain_epochs: int = 20,
    gnn_lr: float = 1e-2,
    seed: int = 42,
    device: str = "cpu",
) -> Stage1Output:
    nodes = graph_data["nodes"]
    G: nx.DiGraph = graph_data["nx_graph"]

    node_to_community, communities = detect_communities(
        G,
        method=partition_method,
        seed=seed,
        min_community_size=min_community_size,
        node_class_map=node_class_map,
        ground_truth_map=ground_truth_map,
    )

    nnp_instances = build_nnp_instances(nodes, G, d_max=d_max, seed=seed, node_to_community=node_to_community)
    nnp_per_community = build_nnp_instances_for_communities(
        nodes, G, node_to_community, communities, d_max=d_max, seed=seed
    )

    node_ids = list(nodes.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    texts = [nodes[nid]["text"] for nid in node_ids]

    embedding_layer = model.get_input_embeddings()
    h0 = init_node_features_from_llm(texts, embedding_layer, tokenizer, device=device)
    a_hat = build_normalized_adjacency(G, num_nodes=len(node_ids), node_index=node_index)

    gnn = GNNEncoder(
        in_dim=h0.shape[-1], hidden_dim=gnn_hidden_dim, out_dim=gnn_out_dim, num_layers=gnn_num_layers
    )
    edges = [(node_index[e["src"]], node_index[e["tgt"]]) for e in graph_data["edges"] if e["src"] in node_index and e["tgt"] in node_index]
    pretrain_gnn(gnn, h0, a_hat, edges, num_nodes=len(node_ids), epochs=gnn_pretrain_epochs, lr=gnn_lr, seed=seed)

    with torch.no_grad():
        z = gnn(h0, a_hat)
        community_embeddings = GNNEncoder.community_embeddings(z, communities, node_index)

    return Stage1Output(
        node_to_community=node_to_community,
        communities=communities,
        node_index=node_index,
        nnp_instances=nnp_instances,
        nnp_instances_per_community=nnp_per_community,
        gnn=gnn,
        h0=h0,
        a_hat=a_hat,
        community_embeddings=community_embeddings,
    )
