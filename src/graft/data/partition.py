
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import networkx as nx

def detect_communities(
    G: nx.DiGraph,
    method: str = "louvain",
    seed: int = 42,
    min_community_size: int = 10,
    node_class_map: Optional[Dict[int, str]] = None,
    ground_truth_map: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, int], List[List[int]]]:
    G_und = G.to_undirected()

    if method == "louvain":
        raw = _louvain(G_und, seed)
    elif method == "leiden":
        raw = _leiden(G_und, seed)
    elif method == "topic_class":
        assert node_class_map is not None, "topic_class requires node_class_map"
        raw = _topic_class(G, node_class_map)
    elif method == "ground_truth":
        assert ground_truth_map is not None, "ground_truth requires ground_truth_map"
        raw = dict(ground_truth_map)
    else:
        raise ValueError(f"Unknown partition method: {method!r}")

    raw = _merge_small(G_und, raw, min_community_size)
    node_to_community, communities = _reindex(raw, list(G.nodes()))
    _print_summary(communities, method)
    return node_to_community, communities

def modularity(G: nx.Graph, node_to_community: Dict[int, int]) -> float:
    communities: Dict[int, set] = {}
    for n, c in node_to_community.items():
        communities.setdefault(c, set()).add(n)
    return nx.algorithms.community.quality.modularity(
        G.to_undirected(), communities.values()
    )

def _louvain(G_und: nx.Graph, seed: int) -> Dict[int, int]:
    from networkx.algorithms.community import louvain_communities

    parts = louvain_communities(G_und, seed=seed)
    n2c: Dict[int, int] = {}
    for i, part in enumerate(parts):
        for n in part:
            n2c[n] = i
    return n2c

def _leiden(G_und: nx.Graph, seed: int) -> Dict[int, int]:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "leidenalg and igraph are required for Leiden partitioning. "
            "Install with: pip install leidenalg igraph"
        ) from exc
    nodes = list(G_und.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    ig_edges = [(node_idx[u], node_idx[v]) for u, v in G_und.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=ig_edges, directed=False)
    partition = leidenalg.find_partition(
        ig_graph, leidenalg.ModularityVertexPartition, seed=seed
    )
    n2c: Dict[int, int] = {}
    for i, part in enumerate(partition):
        for idx in part:
            n2c[nodes[idx]] = i
    return n2c

def _topic_class(G: nx.DiGraph, node_class_map: Dict[int, str]) -> Dict[int, int]:
    classes = sorted(set(node_class_map.values()))
    class_to_id = {c: i for i, c in enumerate(classes)}
    n2c: Dict[int, int] = {}
    for nid in G.nodes():
        cls = node_class_map.get(nid, "unknown")
        n2c[nid] = class_to_id.get(cls, len(class_to_id))
    return n2c

def _merge_small(G_und: nx.Graph, n2c: Dict[int, int], min_size: int) -> Dict[int, int]:
    if min_size <= 1:
        return n2c

    n2c = dict(n2c)
    changed = True
    while changed:
        changed = False
        size_of = Counter(n2c.values())
        small_ids = {cid for cid, sz in size_of.items() if sz < min_size}
        if not small_ids:
            break
        for node, cid in list(n2c.items()):
            if cid not in small_ids:
                continue
            neighbor_comms: Counter = Counter()
            for nbr in G_und.neighbors(node):
                nbr_c = n2c[nbr]
                if nbr_c not in small_ids:
                    neighbor_comms[nbr_c] += 1
            if neighbor_comms:
                best = neighbor_comms.most_common(1)[0][0]
                n2c[node] = best
                changed = True

    return n2c

def _reindex(n2c: Dict[int, int], all_nodes: List[int]) -> Tuple[Dict[int, int], List[List[int]]]:
    old_ids = sorted(set(n2c.values()))
    remap = {old: new for new, old in enumerate(old_ids)}
    new_n2c = {n: remap[c] for n, c in n2c.items()}
    for n in all_nodes:
        if n not in new_n2c:
            new_n2c[n] = -1

    K = max(new_n2c.values()) + 1 if new_n2c else 0
    communities: List[List[int]] = [[] for _ in range(K)]
    for n, c in new_n2c.items():
        if c >= 0:
            communities[c].append(n)
    return new_n2c, communities

def _print_summary(communities: List[List[int]], method: str) -> None:
    sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"[partition/{method}] {len(communities)} communities | sizes (top-10): {sizes[:10]}")
