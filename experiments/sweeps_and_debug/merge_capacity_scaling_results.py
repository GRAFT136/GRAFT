from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from capacity_scaling_sweep import CFG, _estimate_cliff, _write_results_artifacts

def _load_results(input_dirs: List[str]) -> List[Dict[str, object]]:
    merged: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for input_dir in input_dirs:
        results_path = Path(input_dir) / "results.json"
        with open(results_path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            key = (
                row["family"],
                row["arch"],
                int(row["rank"]),
                int(row["edge_budget"]),
                int(row["num_experts"]),
            )
            merged[key] = row
    family_order = {"sbm": 0, "er": 1, "ba": 2}
    arch_order = {"single_lora": 0, "graft_oracle": 1}
    return sorted(
        merged.values(),
        key=lambda row: (
            family_order.get(str(row["family"]), 99),
            arch_order.get(str(row["arch"]), 99),
            int(row["edge_budget"]),
            int(row["rank"]),
            int(row["num_experts"]),
        ),
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--family_plot_rank", type=int, default=64)
    args = parser.parse_args()

    results = _load_results(args.input_dirs)
    cliff_by_rank = {}
    for rank in sorted({int(row["rank"]) for row in results if row["family"] == "sbm" and row["arch"] == "single_lora"}):
        cliff_by_rank[rank] = _estimate_cliff(results, rank, CFG["cliff_fraction"])
    _write_results_artifacts(results, cliff_by_rank, args.output_dir, args.family_plot_rank)
    print(f"[merge] wrote {len(results)} rows to {args.output_dir}")

if __name__ == "__main__":
    main()