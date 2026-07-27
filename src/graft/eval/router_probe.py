
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import torch

@torch.no_grad()
def evaluate_router_hit_rate(
    model,
    router,
    community_embeddings: torch.Tensor,
    eval_items: List[Dict],
    tokenizer,
    community_key: str = "community",
    device: str = "cpu",
    batch_size: int = 16,
) -> Dict:
    model.eval()
    router.eval()

    top_kappa = router.top_kappa
    num_communities = community_embeddings.shape[0]
    random_baseline = top_kappa / num_communities

    per_community_hits: Dict[int, List[bool]] = defaultdict(list)
    all_indices: List[torch.Tensor] = []

    for batch_start in range(0, len(eval_items), batch_size):
        batch = eval_items[batch_start : batch_start + batch_size]
        queries = [item["query"] for item in batch]
        gt_communities = [item[community_key] for item in batch]

        enc = tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)

        embeds = model.get_input_embeddings()(enc["input_ids"])
        mask = enc["attention_mask"].unsqueeze(-1).float()
        query_repr = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        weights, indices, scores = router(query_repr.to(device), community_embeddings.to(device))

        for i, gt_comm in enumerate(gt_communities):
            selected = indices[i].tolist()
            per_community_hits[gt_comm].append(gt_comm in selected)

        all_indices.append(indices.cpu())

    all_hits = [h for hits in per_community_hits.values() for h in hits]
    overall_hit_rate = sum(all_hits) / len(all_hits) if all_hits else 0.0
    per_community_hit_rate = {c: sum(h) / len(h) for c, h in per_community_hits.items()}

    all_idx_tensor = torch.cat(all_indices, dim=0).reshape(-1) if all_indices else torch.empty(0)
    freq = torch.zeros(num_communities)
    for eid in all_idx_tensor:
        freq[eid] += 1
    if freq.sum() > 0:
        freq = freq / freq.sum()

    return {
        "overall_hit_rate": overall_hit_rate,
        "random_baseline": random_baseline,
        "hit_rate_above_random": overall_hit_rate > random_baseline * 1.5,
        "per_community_hit_rate": per_community_hit_rate,
        "load_distribution": freq.tolist(),
        "num_eval_items": len(eval_items),
        "top_kappa": top_kappa,
        "num_communities": num_communities,
    }

def write_report(result: Dict, output_dir: str, name: str = "router_probe_report.md") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    lines = [
        "# Router Probe Report",
        "",
        f"- Eval items: {result['num_eval_items']}",
        f"- Overall hit rate: {result['overall_hit_rate']:.4f}",
        f"- Random baseline (kappa/K): {result['random_baseline']:.4f}",
        f"- Above 1.5x random baseline: {result['hit_rate_above_random']}",
        "",
        "## Per-community hit rate",
        "",
    ]
    for c, hr in sorted(result["per_community_hit_rate"].items()):
        lines.append(f"- community {c}: {hr:.4f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
