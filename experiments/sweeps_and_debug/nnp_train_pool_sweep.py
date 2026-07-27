
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

from nnp_benchmark import (
    ADAPTER_METHODS,
    MODEL_7B,
    assign_communities,
    load_nnp_records,
    load_related_sft_train_records,
    resolve_model_path,
    run_adapter_method,
    save_json,
    split_records,
    summarise_train_pool,
)

DATASET_ALIASES = {
    "children": "children",
    "stack": "stack_elec",
    "stack_elec": "stack_elec",
    "fb": "fb",
    "fb15k-237": "fb",
    "fb15k237": "fb",
}

DISPLAY_NAMES = {
    "children": "children",
    "stack_elec": "stack",
    "fb": "fb15k-237",
}

DEFAULT_AUX_SCHEDULE = ["0", "1000", "5000", "10000", "20000", "full"]

def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset alias: {name}")
    return DATASET_ALIASES[key]

def display_dataset_name(canonical_name: str) -> str:
    return DISPLAY_NAMES[canonical_name]

def parse_aux_schedule(values: Sequence[str]) -> List[str]:
    schedule: List[str] = []
    for value in values:
        token = value.strip().lower()
        if not token:
            continue
        if token == "full":
            schedule.append(token)
            continue
        size = int(token)
        if size < 0:
            raise ValueError(f"aux schedule values must be non-negative, got {value}")
        schedule.append(str(size))
    if not schedule:
        raise ValueError("aux schedule must not be empty")
    return schedule

def materialise_aux_sizes(schedule: Sequence[str], aux_pool_size: int) -> List[int]:
    sizes = {0}
    for token in schedule:
        if token == "full":
            sizes.add(aux_pool_size)
        else:
            sizes.add(min(int(token), aux_pool_size))
    return sorted(sizes)

def stable_shuffle(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    shuffled = [dict(record) for record in records]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled

def tag_train_sources(records: List[Dict[str, Any]], source_name: str) -> None:
    for record in records:
        record.setdefault("train_source", source_name)

def summary_row_to_csv(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "method": row["method"],
        "aux_train_records": row["aux_train_records"],
        "nnp_train_records": row["nnp_train_records"],
        "total_train_records": row["total_train_records"],
        "n_eval_records": row["n_eval_records"],
        "n_f1": row["n_f1"],
        "enm": row["enm"],
        "her": row["her"],
        "precision": row["precision"],
        "recall": row["recall"],
        "parsed_rate": row["parsed_rate"],
        "avg_gold_size": row["avg_gold_size"],
        "avg_pred_size": row["avg_pred_size"],
        "tp": row["tp"],
        "fp": row["fp"],
        "fn": row["fn"],
    }

def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "aux_train_records",
        "nnp_train_records",
        "total_train_records",
        "n_eval_records",
        "n_f1",
        "enm",
        "her",
        "precision",
        "recall",
        "parsed_rate",
        "avg_gold_size",
        "avg_pred_size",
        "tp",
        "fp",
        "fn",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(summary_row_to_csv(row))

def write_report(path: Path, rows: List[Dict[str, Any]], dataset_notes: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# NNP Train-Pool Scale Sweep\n")
    lines.append("\n")
    lines.append("Evaluation datasets stay fixed at children / stack / fb15k-237.\n")
    lines.append("Training pool is expanded as NNP train split + nested prefix of same-source SFT data.\n")
    lines.append("\n")

    for dataset in sorted(dataset_notes.keys()):
        note = dataset_notes[dataset]
        lines.append(f"## {dataset}\n")
        lines.append("\n")
        lines.append(
            f"- NNP train/eval: {note['nnp_train_records']} / {note['eval_records']}\n"
        )
        lines.append(
            f"- Same-source aux pool: {note['aux_pool_records']}\n"
        )
        lines.append(
            f"- Swept aux sizes: {', '.join(str(value) for value in note['aux_sizes'])}\n"
        )
        lines.append("\n")
        lines.append("| Method | Aux | Total Train | N-F1 | ENM | HER |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            if row["dataset"] != dataset:
                continue
            lines.append(
                f"| {row['method']} | {row['aux_train_records']} | {row['total_train_records']} | "
                f"{row['n_f1']:.4f} | {row['enm']:.4f} | {row['her']:.4f} |\n"
            )
        lines.append("\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnp_root", default="../next_neighbor_prediction(nnp)")
    parser.add_argument("--stack_root", default="../Stack_elec_dataset")
    parser.add_argument("--other_sft_root", default="../other_sft_data")
    parser.add_argument("--other_sft_v2_root", default="../other_sft_data_v2")
    parser.add_argument("--other_graph_root", default="../other_graph_dataset")
    parser.add_argument("--datasets", nargs="+", default=["children", "stack", "fb15k-237"])
    parser.add_argument("--methods", nargs="+", default=["single_lora"])
    parser.add_argument("--model_path", default=MODEL_7B)
    parser.add_argument("--output_root", default="outputs/nnp_train_pool_sweep")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_records", type=int, default=2000)
    parser.add_argument("--max_eval_records", type=int, default=300)
    parser.add_argument("--max_aux_pool_records", type=int, default=0)
    parser.add_argument("--aux_train_schedule", nargs="+", default=list(DEFAULT_AUX_SCHEDULE))
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--single_rank", type=int, default=8)
    parser.add_argument("--single_alpha", type=float, default=16.0)
    parser.add_argument("--moe_rank", type=int, default=4)
    parser.add_argument("--moe_alpha", type=float, default=8.0)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--route_sup_weight", type=float, default=0.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    args.aux_train_schedule = parse_aux_schedule(args.aux_train_schedule)
    args.max_aux_train_records = args.max_aux_pool_records

    invalid_methods = [method for method in args.methods if method not in ADAPTER_METHODS]
    if invalid_methods:
        raise ValueError(
            "nnp_train_pool_sweep only supports trainable adapter methods: "
            f"{sorted(ADAPTER_METHODS)}; got {invalid_methods}"
        )

    random.seed(args.seed)

    canonical_datasets = [canonical_dataset_name(name) for name in args.datasets]
    run_name = args.tag or "default"
    output_root = Path(args.output_root) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    dataset_notes: Dict[str, Any] = {}

    for canonical_name in canonical_datasets:
        dataset_display = display_dataset_name(canonical_name)
        dataset_dir = output_root / dataset_display
        dataset_dir.mkdir(parents=True, exist_ok=True)

        raw_records = load_nnp_records(args.nnp_root, canonical_name)
        if not raw_records:
            raise RuntimeError(f"No evaluable NNP records found for {dataset_display}")
        assign_communities(raw_records, args.num_experts)
        nnp_train_records, eval_records = split_records(
            raw_records,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            max_train_records=args.max_train_records,
            max_eval_records=args.max_eval_records,
        )
        if not nnp_train_records or not eval_records:
            raise RuntimeError(
                f"Insufficient split for {dataset_display}: "
                f"train={len(nnp_train_records)} eval={len(eval_records)}"
            )
        tag_train_sources(nnp_train_records, "nnp")

        aux_pool = load_related_sft_train_records(args, canonical_name)
        aux_pool = stable_shuffle(aux_pool, args.seed)
        if args.max_aux_pool_records > 0:
            aux_pool = aux_pool[: args.max_aux_pool_records]
        tag_train_sources(aux_pool, "aux_sft")

        aux_sizes = materialise_aux_sizes(args.aux_train_schedule, len(aux_pool))
        dataset_notes[dataset_display] = {
            "canonical_name": canonical_name,
            "nnp_train_records": len(nnp_train_records),
            "eval_records": len(eval_records),
            "aux_pool_records": len(aux_pool),
            "aux_sizes": aux_sizes,
        }

        save_json(
            dataset_dir / "pool_summary.json",
            {
                "dataset": dataset_display,
                "dataset_canonical": canonical_name,
                "model_path": args.model_path,
                "nnp_train_records": len(nnp_train_records),
                "eval_records": len(eval_records),
                "aux_pool_records": len(aux_pool),
                "aux_sizes": aux_sizes,
                "nnp_summary": summarise_train_pool(nnp_train_records),
                "aux_summary": summarise_train_pool(aux_pool),
            },
        )
        print(
            f"[pool] dataset={dataset_display} nnp={len(nnp_train_records)} "
            f"aux_pool={len(aux_pool)} eval={len(eval_records)} scales={aux_sizes}",
            flush=True,
        )

        for aux_size in aux_sizes:
            scale_dir = dataset_dir / f"aux_{aux_size:07d}"
            scale_dir.mkdir(parents=True, exist_ok=True)
            aux_prefix = [dict(record) for record in aux_pool[:aux_size]]
            train_pool = [dict(record) for record in nnp_train_records] + aux_prefix
            assign_communities(train_pool, args.num_experts)

            save_json(
                scale_dir / "train_pool_summary.json",
                {
                    "dataset": dataset_display,
                    "aux_train_records": aux_size,
                    "nnp_train_records": len(nnp_train_records),
                    "total_train_records": len(train_pool),
                    "train_pool_summary": summarise_train_pool(train_pool),
                },
            )
            print(
                f"[scale] dataset={dataset_display} aux={aux_size} total={len(train_pool)}",
                flush=True,
            )

            for method in args.methods:
                result = run_adapter_method(
                    args=args,
                    dataset=dataset_display,
                    train_records=train_pool,
                    eval_records=eval_records,
                    method=method,
                    dataset_dir=scale_dir,
                )
                summary_rows.append(
                    {
                        **result,
                        "dataset": dataset_display,
                        "dataset_canonical": canonical_name,
                        "method": method,
                        "aux_train_records": aux_size,
                        "nnp_train_records": len(nnp_train_records),
                        "total_train_records": len(train_pool),
                        "n_eval_records": len(eval_records),
                    }
                )

    save_json(
        output_root / "summary.json",
        {
            "model_path": args.model_path,
            "methods": list(args.methods),
            "datasets": [display_dataset_name(name) for name in canonical_datasets],
            "results": summary_rows,
            "dataset_notes": dataset_notes,
        },
    )
    write_summary_csv(output_root / "summary.csv", summary_rows)
    write_report(output_root / "report.md", summary_rows, dataset_notes)
    print(f"[nnp-sweep] summary -> {output_root / 'summary.json'}", flush=True)

if __name__ == "__main__":
    main()
