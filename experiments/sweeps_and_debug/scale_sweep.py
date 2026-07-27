
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.router import SharedGlobalRouter
from src.model.injection import inject_moe_lora, get_moe_layers
from src.model.losses import compute_total_loss
from src.data.synthetic_graph import generate_sbm
from src.eval.monitor import RouterMonitor
from phase1_train import set_router_decision, clear_router_decision, _init_tb

MODEL_PATHS = {
    "0.5B": (
        "/home/USER/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "3B": (
        "/home/USER/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
        "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    ),
    "7B": (
        "/home/USER/.cache/huggingface/hub/"
        "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ),
}

SWEEP_CFG = {
    "num_communities": 8,
    "p_in": 0.15,
    "p_out": 0.005,
    "bridge_fraction": 0.05,
    "qa_per_node": 2,
    "eval_fraction": 0.15,
    "max_train_samples": 4000,
    "max_eval_samples": 600,
    "num_local_experts": 8,
    "top_k": 2,
    "moe_rank": 8,
    "global_rank": 8,
    "single_lora_rank": 72,
    "lora_alpha_moe": 16.0,
    "lora_alpha_single": 144.0,
    "batch_size": 4,
    "grad_clip": 1.0,
    "aux_loss_weight": 0.01,
    "route_sup_weight": 0.3,
    "route_sup_anneal": True,
    "warmup_steps_per_expert": 30,
    "max_length": 256,
    "seed": 42,
    "eval_batch_size": 8,
    "steps_per_1k_nodes": 300,
}

def generate_sbm_qa(graph_data: Dict, rng: random.Random, max_qa: int) -> List[Dict]:
    nodes = graph_data["nodes"]
    edge_set = {(e["src"], e["tgt"]) for e in graph_data["edges"]}
    community_map = graph_data["community_map"]

    node_ids = list(nodes.keys())
    C = max(community_map.values()) + 1
    nodes_by_comm = defaultdict(list)
    for nid, c in community_map.items():
        nodes_by_comm[c].append(nid)

    qa_list = []

    target = max_qa // 2
    generated = 0
    attempts = 0
    while generated < target and attempts < target * 10:
        attempts += 1
        u = rng.choice(node_ids)
        v = rng.choice(node_ids)
        if u == v:
            continue
        c_u = community_map[u]
        c_v = community_map[v]
        has_edge = (u, v) in edge_set
        bucket = "intra" if c_u == c_v else "cross"
        u_text = nodes[u]["text"]
        v_text = nodes[v]["text"]
        qa_list.append({
            "query": f"Is there a direct connection from node {u} ({u_text[:40]}) to node {v} ({v_text[:40]})?",
            "answer": "Yes." if has_edge else "No.",
            "community": c_u,
            "bucket": bucket,
        })
        generated += 1

    target2 = max_qa // 4
    comm_sample = rng.sample(node_ids, min(target2, len(node_ids)))
    for nid in comm_sample:
        c = community_map[nid]
        n_text = nodes[nid]["text"]
        size = len(nodes_by_comm[c])
        qa_list.append({
            "query": f"Which community does node {nid} ({n_text[:40]}) belong to?",
            "answer": f"Community {c}.",
            "community": c,
            "bucket": "intra",
        })
        if rng.random() < 0.3:
            qa_list.append({
                "query": f"How many nodes are in community {c}?",
                "answer": f"{size}.",
                "community": c,
                "bucket": "global",
            })

    rng.shuffle(qa_list)
    return qa_list[:max_qa]

class SweepDataset(Dataset):
    def __init__(self, qa_list: List[Dict], tokenizer, max_length: int):
        self.records = qa_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._assistant_ids = tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = (
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['answer']}<|im_end|>"
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        n = len(self._assistant_ids)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == self._assistant_ids:
                labels[: i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": rec.get("community", 0),
        }

def _sweep_collate(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn_mask[i, :L] = 1
        comms[i] = b["community"]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attn_mask, "community": comms}

_POS_KW = ("yes", "true", "correct", "indeed")
_NEG_KW = ("no", "false", "incorrect", "not")

def _em_check(generated: str, ground_truth: str) -> Optional[bool]:
    gt = ground_truth.lower().strip().rstrip(".")
    gen = generated.lower().strip()

    import re
    gt_nums = re.findall(r'\d+', gt)
    if gt_nums:
        gen_nums = re.findall(r'\d+', gen)
        if gen_nums:
            return int(gt_nums[0]) == int(gen_nums[0])
        return None

    gt_comm = re.findall(r'community\s+(\d+)', gt)
    if gt_comm:
        gen_comm = re.findall(r'community\s+(\d+)', gen)
        if gen_comm:
            return gt_comm[0] == gen_comm[0]
        return None

    gt_label = next((k for k in _POS_KW if gt.startswith(k)), None) or \
               next((k for k in _NEG_KW if gt.startswith(k)), None)
    gen_label = next((k for k in _POS_KW if gen.startswith(k)), None) or \
                next((k for k in _NEG_KW if gen.startswith(k)), None)
    if gt_label and gen_label:
        gt_pos = gt_label in _POS_KW
        gen_pos = gen_label in _POS_KW
        return gt_pos == gen_pos

    def _norm(s: str) -> str:
        s = s.lower().strip().strip('".\'')
        return " ".join(s.split())
    gt_n = _norm(gt)
    gen_n = _norm(gen)
    if gt_n:
        return gt_n in gen_n
    return None

@torch.no_grad()
def evaluate_em(
    model, router, moe_layers, eval_qa: List[Dict],
    tokenizer, device: str,
    arch: str,
    batch_size: int = 4,
    max_new: int = 30,
) -> Dict:
    model.eval()
    if router is not None:
        router.eval()

    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id

    results = defaultdict(list)

    for i in range(0, len(eval_qa), batch_size):
        batch = eval_qa[i: i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=200,
        ).to(device)

        if arch == "moe" and router is not None:
            embed = model.model.embed_tokens(enc["input_ids"])
            mask_f = enc["attention_mask"].unsqueeze(-1).float()
            qr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            rw, ri, _ = router(qr.to(torch.float32))
            set_router_decision(moe_layers, rw, ri)

        out_ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if arch == "moe":
            clear_router_decision(moe_layers)

        prompt_len = enc["input_ids"].shape[1]
        for j, rec in enumerate(batch):
            new_ids = out_ids[j, prompt_len:]
            gen_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            em = _em_check(gen_text, rec["answer"])
            if em is not None:
                results[rec["community"]].append(int(em))

    overall = []
    per_comm = {}
    for c, vals in results.items():
        per_comm[c] = sum(vals) / len(vals) if vals else float("nan")
        overall.extend(vals)

    return {
        "overall_em": sum(overall) / len(overall) if overall else float("nan"),
        "n_valid": len(overall),
        "per_community_em": per_comm,
    }

def run_one(
    graph_size: int,
    arch: str,
    model, tokenizer,
    cfg: Dict,
    device: str,
    output_dir: str,
    rng: random.Random,
) -> Dict:

    print(f"\n{'='*60}")
    print(f"  graph_size={graph_size:,}  arch={arch}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    C = cfg["num_communities"]
    nodes_per_comm = graph_size // C
    actual_size = nodes_per_comm * C
    print(f"  Generating SBM: {C} communities × {nodes_per_comm} nodes = {actual_size:,} total")

    graph_data = generate_sbm(
        num_communities=C,
        nodes_per_community=nodes_per_comm,
        p_in=cfg["p_in"],
        p_out=cfg["p_out"],
        bridge_edges=max(10, int(actual_size * cfg["bridge_fraction"])),
        seed=cfg["seed"],
    )
    print(f"  Graph: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    max_qa = min(cfg["max_train_samples"] + cfg["max_eval_samples"],
                 actual_size * cfg["qa_per_node"])
    all_qa = generate_sbm_qa(graph_data, rng, max_qa)
    n_eval = min(cfg["max_eval_samples"], max(50, len(all_qa) // 7))
    n_train = min(cfg["max_train_samples"], len(all_qa) - n_eval)
    train_qa = all_qa[n_eval: n_eval + n_train]
    eval_qa = all_qa[:n_eval]
    print(f"  QA: {len(train_qa)} train, {len(eval_qa)} eval")

    from collections import Counter
    comm_freq = Counter(r["community"] for r in train_qa)
    raw_counts = torch.tensor([comm_freq.get(i, 1) for i in range(C)], dtype=torch.float32)
    inv_freq = 1.0 / raw_counts
    class_weights = (inv_freq / inv_freq.mean()).to(device)

    router = None
    moe_layers = []

    if arch == "moe":
        model, router = inject_moe_lora(
            model,
            rank=cfg["moe_rank"],
            lora_alpha=cfg["lora_alpha_moe"],
            num_local_experts=C,
            use_global_expert=True,
            top_k=cfg["top_k"],
        )
        router = router.to(device)
        moe_layers = get_moe_layers(model)

        from src.model.moe_lora import GlobalLocalLoraLinear as GLL
        from src.model.moe_lora import GlobalLocalLoraLinear
        _orig_fwd = GlobalLocalLoraLinear.forward

        def _patched_fwd(self, x, rw=None, ri=None):
            _rw = rw if rw is not None else getattr(self, "_cached_rw", None)
            _ri = ri if ri is not None else getattr(self, "_cached_ri", None)
            return _orig_fwd(self, x, _rw, _ri)

        GlobalLocalLoraLinear.forward = _patched_fwd

    else:
        model, _ = inject_moe_lora(
            model,
            rank=cfg["single_lora_rank"],
            lora_alpha=cfg["lora_alpha_single"],
            num_local_experts=0,
            use_global_expert=True,
            top_k=1,
        )
        moe_layers = get_moe_layers(model)

    total_steps = max(200, (graph_size // 1000) * cfg["steps_per_1k_nodes"])
    batch_size = cfg["batch_size"]

    train_ds = SweepDataset(train_qa, tokenizer, cfg["max_length"])
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_sweep_collate, num_workers=0,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    if router is not None:
        trainable += list(router.parameters())
    lr = 5e-4 if arch == "moe" else 1e-4
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    if router is not None:
        router.train()

    step = 0
    dl_iter = iter(train_dl)
    running_loss = 0.0

    while step < total_steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(train_dl)
            batch = next(dl_iter)

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        community_labels = batch["community"].to(device)

        if arch == "moe":
            with torch.no_grad():
                embed = model.model.embed_tokens(input_ids)
                mf = attn_mask.unsqueeze(-1).float()
                qr = (embed * mf).sum(1) / mf.sum(1).clamp(min=1)
            rw, ri, logits = router(qr.to(torch.float32))
            set_router_decision(moe_layers, rw, ri)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        lm_loss = outputs.loss

        if arch == "moe":
            frac = step / max(1, total_steps)
            lam_r = cfg["route_sup_weight"] * (1.0 - frac) if cfg["route_sup_anneal"] else cfg["route_sup_weight"]
            r_loss = F.cross_entropy(logits, community_labels, weight=class_weights)
            loss = compute_total_loss(lm_loss, logits, C, cfg["top_k"], cfg["aux_loss_weight"]) + lam_r * r_loss
            clear_router_decision(moe_layers)
        else:
            loss = lm_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            print(f"    step {step:4d}/{total_steps} | loss={loss.item():.4f} | lm={lm_loss.item():.4f}")

    avg_train_loss = running_loss / step
    print(f"  Training done — avg_loss={avg_train_loss:.4f}")

    em_result = evaluate_em(
        model, router, moe_layers, eval_qa,
        tokenizer, device, arch,
        batch_size=cfg["eval_batch_size"],
    )
    print(f"  EM={em_result['overall_em']:.3f} (n_valid={em_result['n_valid']})")

    ckpt = {"lora_state": {n: p.data for n, p in model.named_parameters() if p.requires_grad}}
    if router is not None:
        ckpt["router"] = router.state_dict()
    torch.save(ckpt, os.path.join(output_dir, "ckpt.pt"))

    return {
        "graph_size": graph_size,
        "arch": arch,
        "avg_train_loss": avg_train_loss,
        "total_steps": step,
        **em_result,
    }

def write_sweep_report(all_results: List[Dict], output_path: str):
    sizes = sorted(set(r["graph_size"] for r in all_results))
    lines = ["# Capacity Crossover Sweep Report\n\n"]

    lines.append("## EM Accuracy vs Graph Size\n\n")
    lines.append(f"| Graph Size | Single LoRA EM | MoE EM | Δ (MoE − Single) | Status |\n")
    lines.append(f"|------------|:--------------:|:------:|:----------------:|--------|\n")

    crossover_found = False
    for sz in sizes:
        single_em = next((r["overall_em"] for r in all_results if r["graph_size"] == sz and r["arch"] == "single_lora"), None)
        moe_em = next((r["overall_em"] for r in all_results if r["graph_size"] == sz and r["arch"] == "moe"), None)
        if single_em is None or moe_em is None:
            continue
        delta = moe_em - single_em
        if delta > 0.01:
            status = "⭐ MoE WINS"
            crossover_found = True
        elif delta < -0.01:
            status = "Single LoRA wins"
        else:
            status = "≈ Tied"
        lines.append(f"| {sz:>10,} | {single_em:.3f} | {moe_em:.3f} | {delta:+.3f} | {status} |\n")

    lines.append("\n## Conclusion\n\n")
    if crossover_found:
        cross_sz = next(
            (sz for sz in sizes
             for r in all_results
             if r["graph_size"] == sz and r["arch"] == "moe"
             and any(r2["graph_size"] == sz and r2["arch"] == "single_lora"
                     and r["overall_em"] - r2["overall_em"] > 0.01
                     for r2 in all_results)),
            None
        )
        lines.append(f"**Crossover found at graph_size ≈ {cross_sz:,}.**\n")
        lines.append("MoE outperforms single LoRA once single LoRA hits its capacity ceiling.\n")
        lines.append("This validates the core thesis: MoE provides capacity benefit when single LoRA is saturated.\n")
    else:
        single_ems = [(r["graph_size"], r["overall_em"]) for r in all_results if r["arch"] == "single_lora"]
        single_ems.sort()
        if len(single_ems) >= 2 and abs(single_ems[-1][1] - single_ems[-2][1]) < 0.02:
            lines.append("**Single LoRA shows signs of saturation at large sizes, but MoE has not overtaken it yet.**\n")
            lines.append("Consider: (a) larger graph sizes, (b) fixing routing failures, (c) revisiting thesis framing.\n")
        else:
            lines.append("**No crossover found in the tested range.**\n")
            lines.append("Single LoRA outperforms MoE at all tested graph sizes.\n")
            lines.append("Possible causes:\n")
            lines.append("  1. Graph sizes still too small to saturate single LoRA capacity.\n")
            lines.append("  2. Routing failure (dead experts) handicapping MoE.\n")
            lines.append("  3. Architectural tension: sparse top-k activation dilutes per-query capacity.\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"[sweep] Report saved → {output_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[3000, 10000, 30000, 100000],
                   help="Graph node counts to sweep over")
    p.add_argument("--model", choices=["0.5B", "3B", "7B"], default="3B",
                   help="Base model size (0.5B/3B are faster for sweeping)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override steps_per_1k_nodes based on epoch count (approx)")
    p.add_argument("--archs", nargs="+", choices=["moe", "single_lora"],
                   default=["moe", "single_lora"])
    p.add_argument("--output_dir", default="outputs/scale_sweep")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = dict(SWEEP_CFG)
    cfg["seed"] = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sweep] Device: {device}")
    print(f"[sweep] Graph sizes: {args.sizes}")
    print(f"[sweep] Architectures: {args.archs}")
    print(f"[sweep] Base model: {args.model}")

    model_path = MODEL_PATHS[args.model]
    print(f"[sweep] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    rng = random.Random(args.seed)
    all_results = []
    os.makedirs(args.output_dir, exist_ok=True)

    for sz in args.sizes:
        for arch in args.archs:
            run_dir = os.path.join(args.output_dir, f"size_{sz}", arch)
            result = run_one(
                graph_size=sz,
                arch=arch,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                output_dir=run_dir,
                rng=rng,
            )
            all_results.append(result)
            print(f"[sweep] Done: size={sz}, arch={arch}, EM={result['overall_em']:.3f}")

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[sweep] Raw results saved → {results_path}")

    write_sweep_report(
        all_results,
        output_path=os.path.join(args.output_dir, "sweep_report.md"),
    )

    print(f"\n{'='*55}")
    print("  CAPACITY CROSSOVER SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Size':>8} | {'Single LoRA EM':>14} | {'MoE EM':>8} | {'Delta':>7}")
    print(f"  {'-'*8} | {'-'*14} | {'-'*8} | {'-'*7}")
    for sz in sorted(args.sizes):
        s = next((r for r in all_results if r["graph_size"] == sz and r["arch"] == "single_lora"), None)
        m = next((r for r in all_results if r["graph_size"] == sz and r["arch"] == "moe"), None)
        se = f"{s['overall_em']:.3f}" if s else "  N/A"
        me = f"{m['overall_em']:.3f}" if m else "  N/A"
        de = f"{m['overall_em']-s['overall_em']:+.3f}" if (s and m) else "   N/A"
        print(f"  {sz:>8,} | {se:>14} | {me:>8} | {de:>7}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
