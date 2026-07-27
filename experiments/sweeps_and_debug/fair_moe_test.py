
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.router import SharedGlobalRouter
from src.model.injection import inject_moe_lora, get_moe_layers
from src.model.losses import load_balancing_loss

MODEL_PATH = (
    "/home/USER/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

CFG = {
    "base_rank": 8,
    "num_communities": 8,
    "top_k": 1,
    "p_in": 0.10,
    "p_out": 0.002,
    "qa_per_node": 6,
    "max_train_qa": 10000,
    "max_eval_qa": 800,
    "batch_size": 4,
    "lr_single": 1e-4,
    "lr_moe": 5e-4,
    "route_sup_weight": 0.5,
    "aux_loss_weight": 0.01,
    "grad_clip": 1.0,
    "warmup_steps_per_expert": 30,
    "max_length": 96,
    "eval_batch_size": 16,
    "seed": 42,
    "device": "cuda",
}

def generate_sbm(nodes_per_community: int, num_communities: int, p_in: float,
                 p_out: float, seed: int) -> Dict:
    rng = random.Random(seed)
    import numpy as np
    np_rng = np.random.default_rng(seed)

    C = num_communities
    N = nodes_per_community * C

    community_map: Dict[int, int] = {}
    nodes: Dict[int, Dict] = {}
    for nid in range(N):
        c = nid // nodes_per_community
        community_map[nid] = c
        nodes[nid] = {"id": nid, "community": c}

    nodes_by_comm = [list(range(c * nodes_per_community, (c + 1) * nodes_per_community))
                     for c in range(C)]

    edges = []
    edge_set = set()

    def add_edge(u, v):
        if u != v and (u, v) not in edge_set:
            edges.append((u, v))
            edge_set.add((u, v))

    for c in range(C):
        cn = nodes_by_comm[c]
        for i in range(len(cn)):
            for j in range(i + 1, len(cn)):
                if np_rng.random() < p_in:
                    add_edge(cn[i], cn[j])

    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            for u in nodes_by_comm[c1]:
                for v in nodes_by_comm[c2]:
                    if np_rng.random() < p_out:
                        add_edge(u, v)

    print(f"  SBM: {N} nodes, {len(edges)} edges, "
          f"avg_intra~{p_in*nodes_per_community:.0f} per node")
    return {
        "nodes": nodes,
        "edges": edge_set,
        "community_map": community_map,
        "nodes_by_comm": nodes_by_comm,
    }

def generate_intra_qa(graph: Dict, rng: random.Random, max_qa: int) -> List[Dict]:
    edge_set = graph["edges"]
    community_map = graph["community_map"]
    nodes_by_comm = graph["nodes_by_comm"]
    C = len(nodes_by_comm)
    target_per_comm = max_qa // C

    qa = []
    for comm_id in range(C):
        cn = nodes_by_comm[comm_id]
        if len(cn) < 2:
            continue
        pos_added = neg_added = 0
        half = target_per_comm // 2
        attempts = 0
        while (pos_added + neg_added) < target_per_comm and attempts < target_per_comm * 30:
            attempts += 1
            u = rng.choice(cn)
            v = rng.choice(cn)
            if u == v:
                continue
            has_edge = (u, v) in edge_set or (v, u) in edge_set
            if has_edge and pos_added >= half:
                continue
            if not has_edge and neg_added >= half:
                continue
            qa.append({
                "query": f"In graph G, is there a direct edge from node_{u} to node_{v}?",
                "answer": "Yes, there is a direct edge." if has_edge else "No, there is no direct edge.",
                "community": comm_id,
            })
            if has_edge:
                pos_added += 1
            else:
                neg_added += 1

    rng.shuffle(qa)
    print(f"  QA: {len(qa)} total (target {max_qa}), "
          f"pos~{sum(1 for q in qa if q['answer'].startswith('Yes'))}, "
          f"neg~{sum(1 for q in qa if q['answer'].startswith('No'))}")
    return qa[:max_qa]

class QADataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._asst = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

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
        n = len(self._asst)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == self._asst:
                labels[: i + n] = -100
                break
        return {"input_ids": input_ids, "labels": labels, "community": rec["community"]}

def _collate(batch):
    max_l = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    iids = torch.zeros(B, max_l, dtype=torch.long)
    lbls = torch.full((B, max_l), -100, dtype=torch.long)
    attn = torch.zeros(B, max_l, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        iids[i, :L] = b["input_ids"]
        lbls[i, :L] = b["labels"]
        attn[i, :L] = 1
        comms[i] = b["community"]
    return {"input_ids": iids, "labels": lbls, "attention_mask": attn, "community": comms}

def set_router(moe_layers, rw, ri):
    for l in moe_layers:
        l._cached_rw = rw
        l._cached_ri = ri

def clear_router(moe_layers):
    for l in moe_layers:
        l._cached_rw = None
        l._cached_ri = None

def warmup_experts(model, moe_layers, train_qa, tokenizer, num_experts, cfg, device):
    by_comm = defaultdict(list)
    for r in train_qa:
        by_comm[r["community"]].append(r)
    steps = cfg["warmup_steps_per_expert"]
    print(f"  [warmup] {num_experts} experts × {steps} steps")
    model.train()
    expert_params = [p for l in moe_layers for p in (l.lora_A_local, l.lora_B_local)]
    opt = torch.optim.AdamW(expert_params, lr=cfg["lr_moe"], weight_decay=0.01)
    for eid in range(num_experts):
        recs = by_comm.get(eid, [])
        if not recs:
            continue
        ds = QADataset(recs, tokenizer, cfg["max_length"])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=_collate, num_workers=0)
        it = iter(dl)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            iids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            B = iids.shape[0]
            rw = torch.ones(B, 1, device=device)
            ri = torch.full((B, 1), eid, dtype=torch.long, device=device)
            set_router(moe_layers, rw, ri)
            loss = model(input_ids=iids, attention_mask=attn, labels=lbls).loss
            opt.zero_grad(); loss.backward()
            for l in moe_layers:
                for p in (l.lora_A_local, l.lora_B_local):
                    if p.grad is not None:
                        m = torch.zeros_like(p.grad); m[eid] = 1.0; p.grad.mul_(m)
            torch.nn.utils.clip_grad_norm_(expert_params, cfg["grad_clip"])
            opt.step(); clear_router(moe_layers)
    print("  [warmup] done")

@torch.no_grad()
def evaluate(model, router, moe_layers, eval_qa, tokenizer, device, arch,
             batch_size=16, max_new=20):
    model.eval()
    if router: router.eval()
    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id
    correct = total = 0
    for i in range(0, len(eval_qa), batch_size):
        batch = eval_qa[i: i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=80).to(device)
        if arch == "moe" and router:
            embed = model.model.embed_tokens(enc["input_ids"])
            mf = enc["attention_mask"].unsqueeze(-1).float()
            qr = (embed * mf).sum(1) / mf.sum(1).clamp(min=1)
            rw, ri, _ = router(qr.to(torch.float32))
            set_router(moe_layers, rw, ri)
        gen = model.generate(
            enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if arch == "moe":
            clear_router(moe_layers)
        plen = enc["input_ids"].shape[1]
        for j, rec in enumerate(batch):
            gen_text = tokenizer.decode(gen[j, plen:], skip_special_tokens=True).lower().strip()
            gt = rec["answer"].lower()
            is_pos = gt.startswith("yes")
            pred_pos = gen_text.startswith("yes") or "direct edge" in gen_text and "no" not in gen_text[:10]
            pred_neg = gen_text.startswith("no") or ("no" in gen_text[:20] and "direct" in gen_text)
            if (is_pos and pred_pos) or (not is_pos and pred_neg):
                correct += 1
            total += 1
    em = correct / total if total > 0 else float("nan")
    return em, total

def run_one(nodes_per_comm: int, arch: str, tokenizer, cfg: Dict,
            device: str, output_dir: str, rng: random.Random,
            max_steps: int) -> Dict:
    C = cfg["num_communities"]
    N = nodes_per_comm * C
    print(f"\n{'='*60}")
    print(f"  nodes/comm={nodes_per_comm}  total={N}  arch={arch}")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )

    graph = generate_sbm(nodes_per_comm, C, cfg["p_in"], cfg["p_out"], cfg["seed"])
    all_qa = generate_intra_qa(graph, rng,
                               nodes_per_comm * C * cfg["qa_per_node"])
    n_eval = min(cfg["max_eval_qa"], max(100, len(all_qa) // 6))
    train_qa = all_qa[n_eval:][:cfg["max_train_qa"]]
    eval_qa = all_qa[:n_eval]
    print(f"  train={len(train_qa)}, eval={len(eval_qa)}")

    router = None
    moe_layers = []

    base_rank = cfg["base_rank"]
    if arch == "moe":
        model, router = inject_moe_lora(
            model, rank=base_rank, lora_alpha=base_rank * 2.0,
            num_local_experts=C, use_global_expert=True, top_k=cfg["top_k"],
        )
        router = router.to(device)
        moe_layers = get_moe_layers(model)

        _orig = GlobalLocalLoraLinear.forward
        def _patched(self, x, rw=None, ri=None):
            return _orig(self, x,
                         rw if rw is not None else getattr(self, "_cached_rw", None),
                         ri if ri is not None else getattr(self, "_cached_ri", None))
        GlobalLocalLoraLinear.forward = _patched

        warmup_experts(model, moe_layers, train_qa, tokenizer, C, cfg, device)

    else:
        model, _ = inject_moe_lora(
            model, rank=base_rank, lora_alpha=base_rank * 2.0,
            num_local_experts=0, use_global_expert=True, top_k=1,
        )
        moe_layers = get_moe_layers(model)

    seen = set()
    trainable = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); trainable.append(p)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable={n_trainable:,}")

    lr = cfg["lr_moe"] if arch == "moe" else cfg["lr_single"]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    ds = QADataset(train_qa, tokenizer, cfg["max_length"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    collate_fn=_collate, num_workers=0)
    total_steps = min(max_steps, len(dl) * 3)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    if router: router.train()
    it = iter(dl)
    running = step = 0

    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        iids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        comm_lbls = batch["community"].to(device)

        if arch == "moe":
            with torch.no_grad():
                embed = model.model.embed_tokens(iids)
                mf = attn.unsqueeze(-1).float()
                qr = (embed * mf).sum(1) / mf.sum(1).clamp(min=1)
            rw, ri, logits = router(qr.to(torch.float32))
            set_router(moe_layers, rw, ri)

        out = model(input_ids=iids, attention_mask=attn, labels=lbls)
        lm_loss = out.loss

        if arch == "moe":
            r_loss = F.cross_entropy(logits, comm_lbls)
            aux = load_balancing_loss(logits, C, cfg["top_k"])
            loss = lm_loss + cfg["route_sup_weight"] * r_loss + cfg["aux_loss_weight"] * aux
            clear_router(moe_layers)
        else:
            loss = lm_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        opt.step(); sched.step()
        running += lm_loss.item()
        step += 1

        if step % 200 == 0:
            print(f"    step {step:4d}/{total_steps} | lm={lm_loss.item():.4f}", flush=True)

    avg_lm_loss = running / step
    print(f"  train done — avg_lm_loss={avg_lm_loss:.4f}")

    em, n_valid = evaluate(model, router, moe_layers, eval_qa, tokenizer,
                           device, arch, batch_size=cfg["eval_batch_size"])
    print(f"  EM={em:.3f} (n={n_valid})")

    if arch == "moe":
        GlobalLocalLoraLinear.forward = _orig
    del model
    if router: del router
    torch.cuda.empty_cache()

    return {
        "arch": arch, "nodes_per_comm": nodes_per_comm, "total_nodes": N,
        "avg_lm_loss": avg_lm_loss, "em": em, "n_valid": n_valid,
        "trainable_params": n_trainable,
    }

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000],
                   help="nodes per community (total = size × 8)")
    p.add_argument("--archs", nargs="+", default=["single_lora", "moe"])
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/fair_moe_test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = dict(CFG)
    cfg["seed"] = args.seed
    cfg["device"] = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[fair] device={cfg['device']}, sizes={args.sizes}, archs={args.archs}")
    print(f"[fair] KEY: rank_single = rank_local = {cfg['base_rank']} (EQUAL)")
    print(f"[fair] MoE: {cfg['num_communities']} experts × rank={cfg['base_rank']}, top_k={cfg['top_k']}")
    print(f"[fair] Single LoRA: rank={cfg['base_rank']} (same per-query capacity)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.Random(args.seed)
    results = []

    for size in args.sizes:
        for arch in args.archs:
            run_dir = os.path.join(args.output_dir, f"n{size*8}_{arch}")
            try:
                r = run_one(size, arch, tokenizer, cfg, cfg["device"],
                            run_dir, rng, args.max_steps)
                results.append(r)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[fair] FAILED n={size*8} {arch}: {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'='*62}")
    print("  EQUAL-RANK CAPACITY CROSSOVER  (rank_single = rank_local = 8)")
    print(f"{'='*62}")
    print(f"  {'Total N':>8} | {'single EM':>10} | {'MoE EM':>8} | {'Δ':>7} | {'single loss':>12} | {'MoE loss':>10}")
    for size in args.sizes:
        N = size * CFG["num_communities"]
        s = next((r for r in results if r["total_nodes"] == N and r["arch"] == "single_lora"), None)
        m = next((r for r in results if r["total_nodes"] == N and r["arch"] == "moe"), None)
        se = f"{s['em']:.3f}" if s else "  N/A"
        me = f"{m['em']:.3f}" if m else "  N/A"
        de = f"{m['em']-s['em']:+.3f}" if (s and m) else "  N/A"
        sl = f"{s['avg_lm_loss']:.4f}" if s else "    N/A"
        ml = f"{m['avg_lm_loss']:.4f}" if m else "    N/A"
        moe_wins = "⭐ MoE WINS" if (s and m and m["em"] > s["em"] + 0.01) else ""
        print(f"  {N:>8,} | {se:>10} | {me:>8} | {de:>7} | {sl:>12} | {ml:>10}  {moe_wins}")
    print(f"{'='*62}")
    print("\nInterpretation of train loss:")
    print("  If single_loss >> MoE_loss at large N → single LoRA saturated, MoE didn't")
    print("  If Δ EM > 0 at large N → capacity crossover confirmed ✓")

if __name__ == "__main__":
    main()
