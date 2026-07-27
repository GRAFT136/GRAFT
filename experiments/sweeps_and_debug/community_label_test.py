
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, get_moe_layers
from src.model.losses import load_balancing_loss

MODEL_PATH = (
    "/home/USER/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

COMMUNITY_LABELS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA"]

CFG = {
    "base_rank":               4,
    "num_communities":         8,
    "top_k":                   1,
    "eval_frac":               0.15,
    "batch_size":              4,
    "lr_single":               2e-4,
    "lr_moe":                  5e-4,
    "route_sup_weight":        0.5,
    "aux_loss_weight":         0.01,
    "grad_clip":               1.0,
    "warmup_steps_per_expert": 30,
    "max_length":              80,
    "eval_batch_size":         16,
    "seed":                    42,
    "epochs":                  12,
}

def generate_community_map(nodes_per_community: int, num_communities: int,
                           seed: int) -> Dict:
    rng = random.Random(seed)
    C = num_communities
    N = nodes_per_community * C

    perm = list(range(N))
    rng.shuffle(perm)

    community_map: Dict[int, int] = {}
    nodes_by_comm: List[List[int]] = [[] for _ in range(C)]

    for rank_in_perm, node_id in enumerate(perm):
        comm = rank_in_perm // nodes_per_community
        community_map[node_id] = comm
        nodes_by_comm[comm].append(node_id)

    print(f"  Assignment: {N} nodes → {C} communities ({nodes_per_community}/community), shuffled")
    return {
        "N": N,
        "community_map": community_map,
        "nodes_by_comm": nodes_by_comm,
    }

def generate_qa(graph: Dict, eval_frac: float, rng: random.Random,
                labels: List[str]) -> Tuple[List[Dict], List[Dict]]:
    community_map = graph["community_map"]
    nodes_by_comm = graph["nodes_by_comm"]

    train_qa, eval_qa = [], []
    for comm_id, nodes in enumerate(nodes_by_comm):
        label = labels[comm_id]
        nodes_shuf = list(nodes)
        rng.shuffle(nodes_shuf)
        n_eval = max(1, int(len(nodes_shuf) * eval_frac))
        for node_id in nodes_shuf[:n_eval]:
            eval_qa.append({
                "node": node_id,
                "query": f"Which community does node_{node_id} belong to?",
                "answer": f"Node_{node_id} belongs to community {label}.",
                "label": label,
                "community": comm_id,
            })
        for node_id in nodes_shuf[n_eval:]:
            train_qa.append({
                "node": node_id,
                "query": f"Which community does node_{node_id} belong to?",
                "answer": f"Node_{node_id} belongs to community {label}.",
                "label": label,
                "community": comm_id,
            })

    rng.shuffle(train_qa)
    rng.shuffle(eval_qa)
    print(f"  QA: train={len(train_qa)}, eval={len(eval_qa)}")
    return train_qa, eval_qa

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
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        n = len(self._asst)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == self._asst:
                labels[: i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": rec["community"],
            "label_str": rec["label"],
        }

def _collate(batch):
    max_l = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    iids = torch.zeros(B, max_l, dtype=torch.long)
    lbls = torch.full((B, max_l), -100, dtype=torch.long)
    attn = torch.zeros(B, max_l, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    label_strs = [b["label_str"] for b in batch]
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        iids[i, :L] = b["input_ids"]
        lbls[i, :L] = b["labels"]
        attn[i, :L] = 1
        comms[i] = b["community"]
    return {"input_ids": iids, "labels": lbls, "attention_mask": attn,
            "community": comms, "label_strs": label_strs}

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
                        mask = torch.zeros_like(p.grad)
                        mask[eid] = 1.0
                        p.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_(expert_params, cfg["grad_clip"])
            opt.step()
            clear_router(moe_layers)
    print("  [warmup] done")

@torch.no_grad()
def evaluate(model, router, moe_layers, eval_qa, tokenizer, device, arch,
             batch_size=16, max_new=20, det_route=False) -> Tuple[float, int]:
    model.eval()
    if router: router.eval()
    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id

    correct = total = 0
    for i in range(0, len(eval_qa), batch_size):
        batch_recs = eval_qa[i: i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch_recs
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)
        if arch == "moe":
            if det_route:
                comm_ids = torch.tensor([r["community"] for r in batch_recs],
                                        dtype=torch.long, device=device)
                rw = torch.ones(len(batch_recs), 1, device=device)
                ri = comm_ids.unsqueeze(1)
                set_router(moe_layers, rw, ri)
            elif router:
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
        for j, rec in enumerate(batch_recs):
            gen_text = tokenizer.decode(gen[j, plen:], skip_special_tokens=True).upper()
            if rec["label"] in gen_text:
                correct += 1
            total += 1
    em = correct / total if total > 0 else float("nan")
    return em, total

def run_one(nodes_per_comm: int, arch: str, tokenizer, cfg: Dict,
            device: str, output_dir: str, rng: random.Random,
            max_steps: int, labels: List[str], det_route: bool = False) -> Dict:
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

    graph = generate_community_map(nodes_per_comm, C, cfg["seed"])
    train_qa, eval_qa = generate_qa(graph, cfg["eval_frac"], rng, labels)

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

        _orig_fwd = GlobalLocalLoraLinear.forward
        def _patched(self, x, rw=None, ri=None):
            return _orig_fwd(self, x,
                             rw if rw is not None else getattr(self, "_cached_rw", None),
                             ri if ri is not None else getattr(self, "_cached_ri", None))
        GlobalLocalLoraLinear.forward = _patched
        warmup_experts(model, moe_layers, train_qa, tokenizer, C, cfg, device)

        if det_route:
            for p in router.parameters():
                p.requires_grad_(False)
            router = None

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
    steps_per_epoch = max(1, len(ds) // cfg["batch_size"])
    total_steps = min(max_steps, steps_per_epoch * cfg["epochs"])
    total_steps = max(total_steps, min(max_steps, 200))

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
            if det_route:
                rw = torch.ones(comm_lbls.shape[0], 1, device=device)
                ri = comm_lbls.unsqueeze(1)
                set_router(moe_layers, rw, ri)
            else:
                with torch.no_grad():
                    embed = model.model.embed_tokens(iids)
                    mf = attn.unsqueeze(-1).float()
                    qr = (embed * mf).sum(1) / mf.sum(1).clamp(min=1)
                rw, ri, logits = router(qr.to(torch.float32))
                set_router(moe_layers, rw, ri)

        out = model(input_ids=iids, attention_mask=attn, labels=lbls)
        lm_loss = out.loss

        if arch == "moe":
            if det_route:
                loss = lm_loss
            else:
                r_loss = F.cross_entropy(logits, comm_lbls)
                aux = load_balancing_loss(logits, C, cfg["top_k"])
                loss = (lm_loss + cfg["route_sup_weight"] * r_loss
                        + cfg["aux_loss_weight"] * aux)
            clear_router(moe_layers)
        else:
            loss = lm_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        opt.step(); sched.step()
        running += lm_loss.item()
        step += 1

        if step % 100 == 0:
            avg = running / step
            print(f"    step {step:5d}/{total_steps} | lm={lm_loss.item():.4f} | avg={avg:.4f}",
                  flush=True)

    avg_lm_loss = running / step
    print(f"  train done — avg_lm_loss={avg_lm_loss:.4f}  (steps={step})")

    model.eval()
    sample_recs = eval_qa[:3]
    print("  [debug] sample generations:")
    for rec in sample_recs:
        prompt = (f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        gen = model.generate(enc["input_ids"], max_new_tokens=20, do_sample=False,
                             eos_token_id=eos_id,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        gen_text = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        print(f"    Q: {rec['query']}")
        print(f"    GT: {rec['label']}  |  Gen: {gen_text!r}")
    model.train()

    em, n_valid = evaluate(model, router, moe_layers, eval_qa, tokenizer,
                           device, arch, batch_size=cfg["eval_batch_size"],
                           det_route=det_route)
    route_note = " [oracle]" if (det_route and arch == "moe") else ""
    print(f"  EM={em:.3f} (n={n_valid}){route_note}")

    result = {
        "arch": arch, "nodes_per_comm": nodes_per_comm, "total_nodes": N,
        "avg_lm_loss": avg_lm_loss, "em": em, "n_valid": n_valid,
        "trainable_params": n_trainable, "steps": step,
        "det_route": det_route,
    }
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if arch == "moe":
        GlobalLocalLoraLinear.forward = _orig_fwd
    del model
    if router: del router
    torch.cuda.empty_cache()
    return result

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512],
                   help="nodes per community (total = size × 8)")
    p.add_argument("--archs", nargs="+", default=["single_lora", "moe"])
    p.add_argument("--det_route", action="store_true",
                   help="MoE uses oracle (deterministic) routing — bypasses learned router")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/comm_label_test")
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
    labels = COMMUNITY_LABELS[: cfg["num_communities"]]
    print(f"[comm_label] device={cfg['device']}, sizes={args.sizes}, archs={args.archs}")
    print(f"[comm_label] BASE_RANK={cfg['base_rank']} (EQUAL for both architectures)")
    print(f"[comm_label] Community labels: {labels}")
    print(f"[comm_label] Task: node_id → Greek letter label (trivial baseline = 1/8 = 12.5%)")
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
            run_dir = os.path.join(args.output_dir,
                                   f"n{size * cfg['num_communities']}_{arch}")
            try:
                r = run_one(size, arch, tokenizer, cfg, cfg["device"],
                            run_dir, rng, args.max_steps, labels,
                            det_route=(args.det_route and arch == "moe"))
                results.append(r)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[comm_label] FAILED n={size * cfg['num_communities']} {arch}: {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'='*72}")
    print("  COMMUNITY LABEL MEMORISATION  (rank_single = rank_local = 4)")
    print(f"  trivial baseline = 12.5 %  (8-class random guess)")
    print(f"{'='*72}")
    print(f"  {'Total N':>8} | {'single EM':>9} | {'MoE EM':>8} | {'Δ':>6} | verdict")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-{'-'*20}")
    by_n: Dict[int, Dict] = {}
    for r in results:
        key = r["total_nodes"]
        if key not in by_n:
            by_n[key] = {}
        by_n[key][r["arch"]] = r["em"]
    for n in sorted(by_n):
        s = by_n[n].get("single_lora", float("nan"))
        m = by_n[n].get("moe", float("nan"))
        d = m - s
        verdict = ("MoE WINS ✓" if d > 0.05 else
                   ("single wins" if d < -0.05 else "tie / noise"))
        print(f"  {n:>8} | {s:>9.3f} | {m:>8.3f} | {d:>+6.3f} | {verdict}")
    print()

if __name__ == "__main__":
    main()
