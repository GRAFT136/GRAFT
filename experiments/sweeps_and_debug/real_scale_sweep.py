
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, get_moe_layers
from src.model.losses import compute_total_loss
from src.data.other_graph_loader import load_other_graph, generate_qa
from phase1_train import set_router_decision, clear_router_decision
from scale_sweep import (
    MODEL_PATHS, SweepDataset, _sweep_collate, evaluate_em,
)

CFG = {
    "num_nodes_ladder": {
        "citeseer": 3186,
        "wn18rr": 40943,
        "amazon-photo": 48362,
        "amazon-computers": 87229,
    },
    "dataset_root": "../other_graph_dataset",
    "moe_rank": 8,
    "global_rank": 8,
    "top_k": 2,
    "lora_alpha_scale": 2.0,
    "batch_size": 4,
    "grad_clip": 1.0,
    "aux_loss_weight": 0.01,
    "route_sup_weight": 0.3,
    "route_sup_anneal": True,
    "class_weighted_route_sup": True,
    "warmup_steps_per_expert": 20,
    "expert_warmup": True,
    "max_length": 256,
    "eval_batch_size": 8,
    "qa_per_node": 5,
    "train_qa_hard_cap": 15000,
    "max_eval_samples": 400,
    "num_epochs": 3,
    "max_steps": 6000,
    "lr_moe": 5e-4,
    "lr_single": 1e-4,
    "seed": 42,
}

def _warmup_experts(model, moe_layers, train_qa, tokenizer, num_experts, cfg, device):
    by_comm = defaultdict(list)
    for r in train_qa:
        by_comm[r["community"]].append(r)

    steps = cfg["warmup_steps_per_expert"]
    print(f"  [warmup] {num_experts} experts × {steps} steps")
    model.train()

    expert_params = []
    for layer in moe_layers:
        expert_params += [layer.lora_A_local, layer.lora_B_local]
    opt = torch.optim.AdamW(expert_params, lr=cfg["lr_moe"], weight_decay=0.01)

    for eid in range(num_experts):
        recs = by_comm.get(eid, [])
        if not recs:
            continue
        ds = SweepDataset(recs, tokenizer, cfg["max_length"])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=_sweep_collate, num_workers=0)
        it = iter(dl)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = input_ids.shape[0]
            rw = torch.ones(B, 1, device=device)
            ri = torch.full((B, 1), eid, dtype=torch.long, device=device)
            set_router_decision(moe_layers, rw, ri)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            for layer in moe_layers:
                for p in (layer.lora_A_local, layer.lora_B_local):
                    if p.grad is not None:
                        m = torch.zeros_like(p.grad); m[eid] = 1.0
                        p.grad.mul_(m)
            torch.nn.utils.clip_grad_norm_(expert_params, cfg["grad_clip"])
            opt.step()
            clear_router_decision(moe_layers)
    print("  [warmup] done")

def run_one_real(
    dataset: str,
    arch: str,
    model_path: str,
    tokenizer,
    cfg: Dict, device: str,
    output_dir: str,
    rng: random.Random,
    max_steps: int,
) -> Dict:
    print(f"\n{'='*62}")
    print(f"  dataset={dataset}  arch={arch}")
    print(f"{'='*62}")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
    )

    graph = load_other_graph(dataset, cfg["dataset_root"])
    num_experts = graph["num_classes"]
    n_nodes = len(graph["nodes"])
    print(f"  graph: {n_nodes:,} nodes, {len(graph['edges']):,} edges, "
          f"{num_experts} classes/relations")

    train_cap = min(cfg["train_qa_hard_cap"], n_nodes * cfg["qa_per_node"])
    max_qa = train_cap + cfg["max_eval_samples"]
    all_qa = generate_qa(graph, rng, max_qa)
    n_eval = min(cfg["max_eval_samples"], max(50, len(all_qa) // 7))
    train_qa = all_qa[n_eval:]
    eval_qa = all_qa[:n_eval]
    print(f"  QA: {len(train_qa)} train (cap {train_cap}), {len(eval_qa)} eval")
    print(f"  train community dist: {dict(sorted(Counter(r['community'] for r in train_qa).items()))}")

    comm_freq = Counter(r["community"] for r in train_qa)
    raw_counts = torch.tensor([comm_freq.get(i, 1) for i in range(num_experts)],
                              dtype=torch.float32)
    inv_freq = 1.0 / raw_counts
    class_weights = (inv_freq / inv_freq.mean()).to(device)

    router = None
    moe_layers = []

    if arch == "moe":
        moe_rank = cfg["moe_rank"]
        model, router = inject_moe_lora(
            model,
            rank=moe_rank,
            lora_alpha=moe_rank * cfg["lora_alpha_scale"],
            num_local_experts=num_experts,
            use_global_expert=True,
            top_k=cfg["top_k"],
        )
        router = router.to(device)
        moe_layers = get_moe_layers(model)

        _orig_fwd = GlobalLocalLoraLinear.forward
        def _patched(self, x, rw=None, ri=None):
            _rw = rw if rw is not None else getattr(self, "_cached_rw", None)
            _ri = ri if ri is not None else getattr(self, "_cached_ri", None)
            return _orig_fwd(self, x, _rw, _ri)
        GlobalLocalLoraLinear.forward = _patched
        _forward_patched = True
    else:
        _forward_patched = False
        single_rank = num_experts * cfg["moe_rank"] + cfg["global_rank"]
        model, _ = inject_moe_lora(
            model,
            rank=single_rank,
            lora_alpha=single_rank * cfg["lora_alpha_scale"],
            num_local_experts=0,
            use_global_expert=True,
            top_k=1,
        )
        moe_layers = get_moe_layers(model)
        print(f"  single_lora rank={single_rank} (matched to MoE budget)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")

    if arch == "moe" and cfg["expert_warmup"]:
        _warmup_experts(model, moe_layers, train_qa, tokenizer, num_experts, cfg, device)

    train_ds = SweepDataset(train_qa, tokenizer, cfg["max_length"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=_sweep_collate, num_workers=0)

    _seen = set()
    trainable = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in _seen:
            _seen.add(id(p)); trainable.append(p)
    lr = cfg["lr_moe"] if arch == "moe" else cfg["lr_single"]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    total_steps = min(max_steps, len(train_dl) * cfg["num_epochs"])
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    model.train()
    if router is not None:
        router.train()

    step = 0
    it = iter(train_dl)
    running = 0.0
    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_dl); batch = next(it)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        comm_labels = batch["community"].to(device)

        if arch == "moe":
            with torch.no_grad():
                embed = model.model.embed_tokens(input_ids)
                mf = attn.unsqueeze(-1).float()
                qr = (embed * mf).sum(1) / mf.sum(1).clamp(min=1)
            rw, ri, logits = router(qr.to(torch.float32))
            set_router_decision(moe_layers, rw, ri)

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        lm_loss = out.loss

        if arch == "moe":
            frac = step / max(1, total_steps)
            lam = cfg["route_sup_weight"] * (1.0 - frac) if cfg["route_sup_anneal"] else cfg["route_sup_weight"]
            if cfg["class_weighted_route_sup"]:
                r_loss = F.cross_entropy(logits, comm_labels, weight=class_weights)
            else:
                r_loss = F.cross_entropy(logits, comm_labels)
            loss = compute_total_loss(lm_loss, logits, num_experts,
                                      cfg["top_k"], cfg["aux_loss_weight"]) + lam * r_loss
            clear_router_decision(moe_layers)
        else:
            loss = lm_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        opt.step()
        sched.step()
        running += loss.item()
        step += 1
        if step % 100 == 0:
            print(f"    step {step:4d}/{total_steps} | loss={loss.item():.4f} | lm={lm_loss.item():.4f}")

    avg_loss = running / step
    print(f"  train done — avg_loss={avg_loss:.4f}")

    em = evaluate_em(model, router, moe_layers, eval_qa, tokenizer, device, arch,
                     batch_size=cfg["eval_batch_size"])
    print(f"  EM={em['overall_em']:.3f} (n_valid={em['n_valid']})")

    ckpt = {"lora_state": {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}}
    if router is not None:
        ckpt["router"] = router.state_dict()
    torch.save(ckpt, os.path.join(output_dir, "ckpt.pt"))

    result = {
        "dataset": dataset,
        "num_nodes": n_nodes,
        "arch": arch,
        "num_experts": num_experts,
        "avg_train_loss": avg_loss,
        "total_steps": step,
        "trainable_params": n_trainable,
        **em,
    }

    if _forward_patched:
        GlobalLocalLoraLinear.forward = _orig_fwd
    del model
    if router is not None:
        del router
    torch.cuda.empty_cache()

    return result

def write_report(results: List[Dict], path: str):
    ladder = sorted(set((r["num_nodes"], r["dataset"]) for r in results))
    lines = ["# Real-Data Capacity Crossover Report\n\n"]
    lines.append("Scale ladder of real graph datasets; EM = Exact Match accuracy.\n\n")
    lines.append("| Dataset | Nodes | Experts | Single LoRA EM | MoE EM | Δ (MoE−Single) | Winner |\n")
    lines.append("|---------|------:|:-------:|:--------------:|:------:|:--------------:|--------|\n")

    crossover = None
    for n_nodes, ds in ladder:
        s = next((r for r in results if r["dataset"] == ds and r["arch"] == "single_lora"), None)
        m = next((r for r in results if r["dataset"] == ds and r["arch"] == "moe"), None)
        if not s or not m:
            continue
        d = m["overall_em"] - s["overall_em"]
        if d > 0.01:
            win = "⭐ MoE"
            if crossover is None:
                crossover = (ds, n_nodes)
        elif d < -0.01:
            win = "Single LoRA"
        else:
            win = "≈ Tie"
        ne = m["num_experts"]
        lines.append(f"| {ds} | {n_nodes:,} | {ne} | {s['overall_em']:.3f} | "
                     f"{m['overall_em']:.3f} | {d:+.3f} | {win} |\n")

    lines.append("\n## Interpretation\n\n")
    if crossover:
        lines.append(f"**CROSSOVER FOUND at `{crossover[0]}` ({crossover[1]:,} nodes).**\n\n")
        lines.append("MoE overtakes single LoRA once the graph is large enough that single "
                     "LoRA's fixed capacity saturates. This is direct evidence for the core "
                     "thesis: sparse MoE's larger total capacity pays off under capacity pressure.\n")
    else:
        lines.append("**No crossover in tested range — single LoRA still wins everywhere.**\n\n")
        lines.append("Check whether single-LoRA train loss is plateauing at the largest scale "
                     "(sign of approaching saturation). If not saturated, go bigger. If saturated "
                     "and MoE still loses, the sparse decomposition itself is the bottleneck.\n")

    lines.append("\n## Single-LoRA train loss (saturation diagnostic)\n\n")
    lines.append("| Dataset | Nodes | Single LoRA train loss | MoE train loss |\n")
    lines.append("|---------|------:|:----------------------:|:--------------:|\n")
    for n_nodes, ds in ladder:
        s = next((r for r in results if r["dataset"] == ds and r["arch"] == "single_lora"), None)
        m = next((r for r in results if r["dataset"] == ds and r["arch"] == "moe"), None)
        sl = f"{s['avg_train_loss']:.4f}" if s else "—"
        ml = f"{m['avg_train_loss']:.4f}" if m else "—"
        lines.append(f"| {ds} | {n_nodes:,} | {sl} | {ml} |\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[sweep] Report → {path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["citeseer", "wn18rr", "amazon-photo", "amazon-computers"])
    p.add_argument("--archs", nargs="+", choices=["moe", "single_lora"],
                   default=["single_lora", "moe"])
    p.add_argument("--model", choices=["0.5B", "3B", "7B"], default="3B")
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--output_dir", default="outputs/real_scale_sweep")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = dict(CFG)
    cfg["seed"] = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sweep] device={device}, model={args.model}, datasets={args.datasets}")

    model_path = MODEL_PATHS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(args.seed)
    results = []
    os.makedirs(args.output_dir, exist_ok=True)

    ordered = sorted(args.datasets, key=lambda d: cfg["num_nodes_ladder"].get(d, 0))

    for ds in ordered:
        for arch in args.archs:
            run_dir = os.path.join(args.output_dir, ds, arch)
            try:
                res = run_one_real(ds, arch, model_path, tokenizer, cfg, device, run_dir, rng, args.max_steps)
                results.append(res)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[sweep] FAILED {ds}/{arch}: {e}")
                import traceback; traceback.print_exc()

    write_report(results, os.path.join(args.output_dir, "sweep_report.md"))

    print(f"\n{'='*60}")
    print("  REAL-DATA CAPACITY CROSSOVER SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<18} {'Nodes':>8} {'Single':>8} {'MoE':>8} {'Δ':>7}")
    for ds in ordered:
        s = next((r for r in results if r["dataset"] == ds and r["arch"] == "single_lora"), None)
        m = next((r for r in results if r["dataset"] == ds and r["arch"] == "moe"), None)
        se = f"{s['overall_em']:.3f}" if s else "N/A"
        me = f"{m['overall_em']:.3f}" if m else "N/A"
        de = f"{m['overall_em']-s['overall_em']:+.3f}" if (s and m) else "N/A"
        nn = cfg["num_nodes_ladder"].get(ds, 0)
        print(f"  {ds:<18} {nn:>8,} {se:>8} {me:>8} {de:>7}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
