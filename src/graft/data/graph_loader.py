
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

def load_cora(data_dir: str) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    all_csv = data_dir / "all.csv"
    edges_csv = data_dir / "edges.csv"
    if not all_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError(
            f"Cora raw files not found under {data_dir}. Expected 'all.csv' and "
            "'edges.csv'. Download the Cora TAG dataset and place both files "
            "there, or use `graft.data.load_synthetic` for a self-contained "
            "smoke test / demo graph."
        )

    nodes: Dict[int, Dict[str, Any]] = {}
    with open(all_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nid = int(row["id"])
            nodes[nid] = {
                "text": row["T"].strip(),
                "abstract": row.get("A", "").strip(),
                "label": int(row["label"]) if row.get("label", "").strip() else None,
                "class": row.get("class", "").strip(),
            }

    raw_edges: List[Tuple[int, int]] = []
    with open(edges_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tcol = next(k for k in reader.fieldnames if k.startswith("target"))
        for row in reader:
            citer = int(row[tcol])
            cited = int(row["source"])
            raw_edges.append((citer, cited))

    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    G.add_edges_from(raw_edges)

    max_in = max(dict(G.in_degree()).values())
    assert max_in > 50, (
        f"Edge direction sanity FAILED: max in-degree = {max_in} (expected >50). "
        "Check the CSV direction convention."
    )

    edges_repr = [{"src": s, "rel": "cites", "tgt": t} for s, t in raw_edges]

    return {"nodes": nodes, "edges": edges_repr, "nx_graph": G}

def load_synthetic(json_path: str) -> Dict[str, Any]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for nid, attrs in data["nodes"].items():
        G.add_node(int(nid), **attrs)
    raw_edges = []
    for e in data["edges"]:
        G.add_edge(e["src"], e["tgt"])
        raw_edges.append({"src": e["src"], "rel": e.get("rel", "linked"), "tgt": e["tgt"]})

    return {
        "nodes": {int(k): v for k, v in data["nodes"].items()},
        "edges": raw_edges,
        "nx_graph": G,
        "community_map": {int(k): v for k, v in data.get("community_map", {}).items()},
    }
