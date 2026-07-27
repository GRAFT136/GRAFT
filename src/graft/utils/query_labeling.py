
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional

GLOBAL_KEYWORDS: List[str] = [
    "how many papers",
    "total number",
    "total nodes",
    "total edges",
    "average degree",
    "average out-degree",
    "average in-degree",
    "is the cora",
    "is there a cycle",
    "dag",
    "most incoming",
    "most outgoing",
    "highest in-degree",
    "highest out-degree",
    "connected component",
    "number of components",
]

class QueryLabeler:

    def __init__(self, node_to_community: Dict[int, int], global_keywords: Optional[List[str]] = None) -> None:
        self.n2c = node_to_community
        self.global_kws = [k.lower() for k in (global_keywords or GLOBAL_KEYWORDS)]

    def label_by_nodes(self, supporting_nodes: List[int]) -> str:
        if not supporting_nodes:
            return "global"
        comms = {self.n2c.get(n, -1) for n in supporting_nodes}
        comms.discard(-1)
        if len(comms) == 0:
            return "global"
        if len(comms) == 1:
            return "intra_community"
        return "cross_community"

    def label_by_text(self, query_text: str) -> Optional[str]:
        q_lower = query_text.lower()
        for kw in self.global_kws:
            if kw in q_lower:
                return "global"
        return None

    def label(self, query_text: str, supporting_nodes: Optional[List[int]] = None) -> str:
        text_label = self.label_by_text(query_text)
        if text_label == "global":
            return "global"
        if supporting_nodes is not None:
            return self.label_by_nodes(supporting_nodes)
        return "unknown"

def label_jsonl(
    jsonl_path: str,
    node_to_community: Dict[int, int],
    supporting_nodes_key: str = "supporting_nodes",
    query_key: str = "query",
    output_key: str = "community_label",
) -> List[Dict]:
    labeler = QueryLabeler(node_to_community)
    labeled: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            snodes = rec.get(supporting_nodes_key, None)
            rec[output_key] = labeler.label(rec.get(query_key, ""), snodes)
            labeled.append(rec)
    return labeled

def bucket_stats(labeled: List[Dict], label_key: str = "community_label") -> Dict[str, int]:
    counts = Counter(r.get(label_key, "unknown") for r in labeled)
    total = sum(counts.values())
    print(f"[query_labeling] Total={total}")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20s}: {cnt:5d}  ({100 * cnt / total:.1f}%)")
    return dict(counts)
