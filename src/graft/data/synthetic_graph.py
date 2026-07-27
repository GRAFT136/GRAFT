
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

_COMM_VOCAB: List[List[str]] = [
    ["neural", "perceptron", "backprop", "gradient", "activation"],
    ["bayesian", "prior", "posterior", "inference", "sampling"],
    ["genetic", "chromosome", "mutation", "crossover", "fitness"],
    ["temporal", "reward", "policy", "value", "reinforcement"],
    ["symbolic", "rule", "logic", "predicate", "theorem"],
    ["kernel", "support", "margin", "hyperplane", "dual"],
    ["cluster", "centroid", "partition", "linkage", "dendrogram"],
    ["attention", "transformer", "embedding", "token", "head"],
    ["causal", "intervention", "counterfactual", "effect", "instrument"],
    ["sparse", "regularize", "lasso", "ridge", "penalty"],
    ["graph", "node", "edge", "spectral", "laplacian"],
    ["recurrent", "hidden", "gate", "sequence", "memory"],
]

def _community_text(community_id: int, node_index: int, vocab: List[str]) -> str:
    w1 = vocab[node_index % len(vocab)]
    w2 = vocab[(node_index + 2) % len(vocab)]
    return f"Paper on {w1} and {w2} [community {community_id}, node {node_index}]"

def generate_sbm(
    num_communities: int = 8,
    nodes_per_community: int = 200,
    p_in: float = 0.15,
    p_out: float = 0.005,
    bridge_edges: int = 100,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    C = num_communities
    N_per = nodes_per_community
    vocab_pool = (_COMM_VOCAB * ((C // len(_COMM_VOCAB)) + 1))[:C]

    community_map: Dict[int, int] = {}
    node_texts: Dict[int, Dict] = {}
    nid = 0
    for c in range(C):
        vocab = vocab_pool[c]
        for i in range(N_per):
            text = _community_text(c, i, vocab)
            node_texts[nid] = {"text": text, "community": c}
            community_map[nid] = c
            nid += 1

    total_nodes = nid
    nodes_by_comm: List[List[int]] = [
        list(range(c * N_per, (c + 1) * N_per)) for c in range(C)
    ]

    edges: List[Tuple[int, int]] = []
    edge_set: set = set()

    def add_edge(u: int, v: int) -> None:
        if u != v and (u, v) not in edge_set:
            edges.append((u, v))
            edge_set.add((u, v))

    for c in range(C):
        comm_nodes = nodes_by_comm[c]
        for u in comm_nodes:
            for v in comm_nodes:
                if u != v and np_rng.random() < p_in:
                    add_edge(u, v)

    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            for u in nodes_by_comm[c1]:
                for v in nodes_by_comm[c2]:
                    if np_rng.random() < p_out:
                        add_edge(u, v)
                    if np_rng.random() < p_out:
                        add_edge(v, u)

    added_bridges = 0
    tries = 0
    all_nodes = list(range(total_nodes))
    while added_bridges < bridge_edges and tries < bridge_edges * 50:
        u = rng.choice(all_nodes)
        v = rng.choice(all_nodes)
        if community_map[u] != community_map[v] and (u, v) not in edge_set:
            add_edge(u, v)
            added_bridges += 1
        tries += 1

    G = nx.DiGraph()
    for nid_, attrs in node_texts.items():
        G.add_node(nid_, **attrs)
    G.add_edges_from(edges)

    edges_repr = [{"src": u, "rel": "linked", "tgt": v} for u, v in edges]
    result = {
        "nodes": node_texts,
        "edges": edges_repr,
        "nx_graph": G,
        "community_map": community_map,
        "config": {
            "num_communities": C,
            "nodes_per_community": N_per,
            "p_in": p_in,
            "p_out": p_out,
            "bridge_edges": bridge_edges,
            "seed": seed,
        },
    }

    if output_path:
        _save_json(result, output_path)

    return result

def _save_json(graph_data: Dict, path: str) -> None:
    save_data = {
        "nodes": {str(k): v for k, v in graph_data["nodes"].items()},
        "edges": graph_data["edges"],
        "community_map": {str(k): v for k, v in graph_data["community_map"].items()},
        "config": graph_data.get("config", {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(
        f"[synthetic_graph] Saved to {path}  "
        f"(nodes={len(save_data['nodes'])}, edges={len(save_data['edges'])})"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_communities", type=int, default=8)
    parser.add_argument("--nodes_per_community", type=int, default=200)
    parser.add_argument("--p_in", type=float, default=0.15)
    parser.add_argument("--p_out", type=float, default=0.005)
    parser.add_argument("--bridge_edges", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/synthetic_sbm.json")
    args = parser.parse_args()

    g = generate_sbm(
        num_communities=args.num_communities,
        nodes_per_community=args.nodes_per_community,
        p_in=args.p_in,
        p_out=args.p_out,
        bridge_edges=args.bridge_edges,
        seed=args.seed,
        output_path=args.output,
    )
    G = g["nx_graph"]
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    comm_sizes: Dict[int, int] = {}
    for _, c in g["community_map"].items():
        comm_sizes[c] = comm_sizes.get(c, 0) + 1
    print(f"Communities: {sorted(comm_sizes.values(), reverse=True)}")
