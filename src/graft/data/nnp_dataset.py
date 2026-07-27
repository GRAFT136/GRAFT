
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import networkx as nx

DEFAULT_DELIMITER = " | "
PROMPT_TEMPLATE = "Given the entity [{text}], list its directly connected neighbors."

def nnp_prompt(node_text: str) -> str:
    return PROMPT_TEMPLATE.format(text=node_text)

@dataclass
class NNPInstance:

    node_id: int
    prompt: str
    target: str
    neighbor_ids: List[int] = field(default_factory=list)
    community_id: Optional[int] = None
    subsample_index: int = 0
    num_subsamples: int = 1

    def as_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "query": self.prompt,
            "answer": self.target,
            "neighbor_ids": self.neighbor_ids,
            "community": self.community_id,
            "subsample_index": self.subsample_index,
            "num_subsamples": self.num_subsamples,
        }

def _neighbors(G: nx.DiGraph, node_id: int, undirected: bool = True) -> List[int]:
    if undirected:
        succ = set(G.successors(node_id)) if G.has_node(node_id) else set()
        pred = set(G.predecessors(node_id)) if G.has_node(node_id) else set()
        return sorted(succ | pred)
    return sorted(G.successors(node_id)) if G.has_node(node_id) else []

def build_nnp_instances(
    nodes: Dict[int, Dict],
    G: nx.DiGraph,
    d_max: int = 32,
    delimiter: str = DEFAULT_DELIMITER,
    seed: int = 42,
    node_to_community: Optional[Dict[int, int]] = None,
    node_ids: Optional[Iterable[int]] = None,
    undirected_neighbors: bool = True,
) -> List[NNPInstance]:
    rng = random.Random(seed)
    ids = list(node_ids) if node_ids is not None else list(nodes.keys())
    instances: List[NNPInstance] = []

    for vid in ids:
        if vid not in nodes:
            continue
        text_i = nodes[vid]["text"]
        neighbors = _neighbors(G, vid, undirected=undirected_neighbors)
        d_i = len(neighbors)
        community_id = node_to_community.get(vid) if node_to_community else None
        prompt = nnp_prompt(text_i)

        if d_i == 0:
            instances.append(
                NNPInstance(
                    node_id=vid,
                    prompt=prompt,
                    target="(no directly connected neighbors)",
                    neighbor_ids=[],
                    community_id=community_id,
                    subsample_index=0,
                    num_subsamples=1,
                )
            )
            continue

        if d_i <= d_max:
            perm = neighbors[:]
            rng.shuffle(perm)
            target = delimiter.join(nodes[n]["text"] for n in perm if n in nodes)
            instances.append(
                NNPInstance(
                    node_id=vid,
                    prompt=prompt,
                    target=target,
                    neighbor_ids=perm,
                    community_id=community_id,
                    subsample_index=0,
                    num_subsamples=1,
                )
            )
        else:
            num_subsamples = math.ceil(d_i / d_max)
            for m in range(num_subsamples):
                sample = rng.sample(neighbors, min(d_i, d_max))
                rng.shuffle(sample)
                target = delimiter.join(nodes[n]["text"] for n in sample if n in nodes)
                instances.append(
                    NNPInstance(
                        node_id=vid,
                        prompt=prompt,
                        target=target,
                        neighbor_ids=sample,
                        community_id=community_id,
                        subsample_index=m,
                        num_subsamples=num_subsamples,
                    )
                )

    return instances

def build_nnp_instances_for_communities(
    nodes: Dict[int, Dict],
    G: nx.DiGraph,
    node_to_community: Dict[int, int],
    communities: List[List[int]],
    d_max: int = 32,
    delimiter: str = DEFAULT_DELIMITER,
    seed: int = 42,
    undirected_neighbors: bool = True,
) -> Dict[int, List[NNPInstance]]:
    per_community: Dict[int, List[NNPInstance]] = {}
    for k, member_ids in enumerate(communities):
        per_community[k] = build_nnp_instances(
            nodes,
            G,
            d_max=d_max,
            delimiter=delimiter,
            seed=seed + k,
            node_to_community=node_to_community,
            node_ids=member_ids,
            undirected_neighbors=undirected_neighbors,
        )
    return per_community
