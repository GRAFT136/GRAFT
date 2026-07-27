
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, get_moe_layers

MODEL_PATH = (
    "/home/USER/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)

COMMUNITY_LABELS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "ETA", "THETA"]
STRUCT_DIM = 16
BASE_RANK = 4

CFG = {
    "num_communities":   8,
    "top_k":             1,
    "p_in":              0.10,
    "p_out":             0.002,
    "eval_frac":         0.15,
    "batch_size":        4,
    "lr_single":         2e-4,
    "lr_moe":            5e-4,
    "route_sup_weight":  1.0,
    "grad_clip":         1.0,
    "warmup_steps_per_expert": 30,
    "max_length":        80,
    "eval_batch_size":   16,
    "epochs":            12,
    "oracle_fraction":   0.5,
    "anneal_fraction":   0.3,
    "seed":              42,
}

def generate_sbm(nodes_per_community: int, C: int, p_in: float,
                 p_out: float, seed: int, display_shuffle_seed: int) -> Dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    N = nodes_per_community * C

    community_map: Dict[int, int] = {v: v // nodes_per_community for v in range(N)}
    nodes_by_comm: List[List[int]] = [
        list(range(c * nodes_per_community, (c + 1) * nodes_per_community))
        for c in range(C)
    ]

    edges: List[Tuple[int, int]] = []
    edge_set: set = set()

    def add_edge(u: int, v: int) -> None:
        if u != v and (min(u, v), max(u, v)) not in edge_set:
            edge_set.add((min(u, v), max(u, v)))
            edges.append((u, v))

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

    print(f"  SBM: {N} nodes, {len(edges)} edges "
          f"(p_in={p_in}, p_out={p_out})")

    disp_rng = random.Random(display_shuffle_seed)
    perm = list(range(N))
    disp_rng.shuffle(perm)
    inv_perm: Dict[int, int] = {perm[v]: v for v in range(N)}

    return {
        "N":            N,
        "edges":        edges,
        "community_map": community_map,
        "nodes_by_comm": nodes_by_comm,
        "perm":         perm,
        "inv_perm":     inv_perm,
    }

def compute_laplacian_pe(edges: List[Tuple[int, int]], N: int,
                          k: int = STRUCT_DIM) -> np.ndarray:
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:
        raise ImportError("scipy is required for Laplacian PE: pip install scipy")

    if not edges:
        return np.zeros((N, k), dtype=np.float32)

    rows_fwd, cols_fwd = zip(*edges)
    rows = list(rows_fwd) + list(cols_fwd)
    cols = list(cols_fwd) + list(rows_fwd)
    data = [1.0] * len(rows)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

    deg = np.array(A.sum(axis=1)).flatten()
    inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    D_inv_sqrt = sp.diags(inv_sqrt_deg)

    L = sp.eye(N, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    k_req = min(k + 1, N - 1)
    try:
        _, eigvecs = spla.eigsh(L, k=k_req, which="SM", tol=1e-4, maxiter=3000)
    except Exception:
        L_dense = L.toarray()
        _, eigvecs = np.linalg.eigh(L_dense)
        eigvecs = eigvecs[:, :k_req]

    pe = eigvecs[:, 1:k_req]
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])), mode="constant")

    return pe.astype(np.float32)

def generate_qa(graph: Dict, eval_frac: float,
                rng: random.Random, labels: List[str]) -> Tuple[List, List]:
    community_map = graph["community_map"]
    nodes_by_comm = graph["nodes_by_comm"]
    perm = graph["perm"]
    labels_used = labels[:len(nodes_by_comm)]

    train_qa, eval_qa = [], []
    for comm_id, nodes in enumerate(nodes_by_comm):
        label = labels_used[comm_id]
        shuf = list(nodes)
        rng.shuffle(shuf)
        n_eval = max(1, int(len(shuf) * eval_frac))
        for orig in shuf[:n_eval]:
            disp = perm[orig]
            eval_qa.append({
                "display_id":  disp,
                "original_id": orig,
                "query":  f"Which community does node_{disp} belong to?",
                "answer": f"Node_{disp} belongs to community {label}.",
                "label":  label,
                "community": comm_id,
            })
        for orig in shuf[n_eval:]:
            disp = perm[orig]
            train_qa.append({
                "display_id":  disp,
                "original_id": orig,
                "query":  f"Which community does node_{disp} belong to?",
                "answer": f"Node_{disp} belongs to community {label}.",
                "label":  label,
                "community": comm_id,
            })

    rng.shuffle(train_qa)
    rng.shuffle(eval_qa)
    print(f"  QA: train={len(train_qa)}, eval={len(eval_qa)}")
    return train_qa, eval_qa

class TopoAwareRouter(nn.Module):

    def __init__(self, hidden_size: int, struct_dim: int,
                 num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        inner = max(struct_dim * 2, 64)
        self.film_net = nn.Sequential(
            nn.Linear(struct_dim, inner),
            nn.GELU(),
        )
        self.film_gamma = nn.Linear(inner, hidden_size)
        self.film_beta  = nn.Linear(inner, hidden_size)

        self.text_head = nn.Linear(hidden_size, num_experts)

        self.struct_head = nn.Linear(struct_dim, num_experts)

        self.mix_logit = nn.Parameter(torch.zeros(1))

        self.fallback_head = nn.Sequential(
            nn.Linear(hidden_size, max(hidden_size // 2, 64)),
            nn.GELU(),
            nn.Linear(max(hidden_size // 2, 64), num_experts),
        )

    def forward(
        self,
        h_mean: torch.Tensor,
        struct_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if struct_emb is not None:
            z_inner = self.film_net(struct_emb)
            gamma   = self.film_gamma(z_inner)
            beta    = self.film_beta(z_inner)
            h_cond  = h_mean * (1.0 + gamma) + beta

            g_text   = self.text_head(h_cond)
            g_struct = self.struct_head(struct_emb)

            alpha  = torch.sigmoid(self.mix_logit)
            logits = alpha * g_text + (1.0 - alpha) * g_struct
        else:
            logits = self.fallback_head(h_mean)

        probs   = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices, logits

class TextOnlyRouter(nn.Module):

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        inner = max(hidden_size // 2, 256)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, num_experts),
        )

    def forward(self, h_mean: torch.Tensor,
                struct_emb=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(h_mean)
        probs  = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, indices, logits

class OracleAnnealScheduler:

    def __init__(self, total_steps: int, oracle_fraction: float,
                 anneal_fraction: float) -> None:
        self.oracle_steps = int(total_steps * oracle_fraction)
        anneal_steps      = int(total_steps * anneal_fraction)
        self.anneal_end   = self.oracle_steps + anneal_steps

    def get_epsilon(self, step: int) -> float:
        if step < self.oracle_steps:
            return 0.0
        if step >= self.anneal_end:
            return 1.0
        return (step - self.oracle_steps) / max(1, self.anneal_end - self.oracle_steps)

    def __repr__(self) -> str:
        return (f"OracleAnnealScheduler("
                f"oracle_steps={self.oracle_steps}, "
                f"anneal_end={self.anneal_end})")

_NODE_PAT = re.compile(r"\bnode_(\d+)\b")

class AnchorLinker:

    def __init__(self, inv_perm: Dict[int, int],
                 lap_pe: torch.Tensor) -> None:
        self.inv_perm = inv_perm
        self.lap_pe   = lap_pe

    def link_oracle(self, original_id: int) -> torch.Tensor:
        return self.lap_pe[original_id]

    def link_text(self, query_text: str) -> Optional[torch.Tensor]:
        m = _NODE_PAT.search(query_text)
        if m is None:
            return None
        display_id = int(m.group(1))
        orig = self.inv_perm.get(display_id)
        if orig is None:
            return None
        return self.lap_pe[orig]

    def batch_link_text(self, queries: List[str],
                        device: torch.device) -> Optional[torch.Tensor]:
        pes = []
        failed = 0
        for q in queries:
            pe = self.link_text(q)
            if pe is None:
                failed += 1
                pes.append(torch.zeros(self.lap_pe.shape[1]))
            else:
                pes.append(pe)
        if failed == len(queries):
            return None
        return torch.stack(pes).to(device)

    def batch_link_oracle(self, original_ids: List[int],
                          device: torch.device) -> torch.Tensor:
        return torch.stack([self.lap_pe[oid] for oid in original_ids]).to(device)

class QADataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self._asst = tokenizer.encode(
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
            text, truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels    = input_ids.clone()
        n  = len(self._asst)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == self._asst:
                labels[: i + n] = -100
                break
        return {
            "input_ids":   input_ids,
            "labels":      labels,
            "community":   rec["community"],
            "original_id": rec["original_id"],
            "query":       rec["query"],
            "label":       rec["label"],
        }

def _collate(batch):
    max_l = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    iids  = torch.zeros(B, max_l, dtype=torch.long)
    lbls  = torch.full((B, max_l), -100, dtype=torch.long)
    attn  = torch.zeros(B, max_l, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    origs = torch.zeros(B, dtype=torch.long)
    queries = []
    label_strs = []
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        iids[i, :L] = b["input_ids"]
        lbls[i,  :L] = b["labels"]
        attn[i,  :L] = 1
        comms[i]     = b["community"]
        origs[i]     = b["original_id"]
        queries.append(b["query"])
        label_strs.append(b["label"])
    return {
        "input_ids":    iids,
        "labels":       lbls,
        "attention_mask": attn,
        "community":    comms,
        "original_ids": origs,
        "queries":      queries,
        "label_strs":   label_strs,
    }

def set_router(moe_layers, rw: torch.Tensor, ri: torch.Tensor) -> None:
    for l in moe_layers:
        l._cached_rw = rw
        l._cached_ri = ri

def clear_router(moe_layers) -> None:
    for l in moe_layers:
        l._cached_rw = None
        l._cached_ri = None

def get_mean_repr(model, iids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        emb = model.model.embed_tokens(iids)
    mf = attn.unsqueeze(-1).float()
    return (emb * mf).sum(1) / mf.sum(1).clamp(min=1)

def warmup_experts(model, moe_layers, train_qa, tokenizer, C, cfg, device):
    by_comm = defaultdict(list)
    for r in train_qa:
        by_comm[r["community"]].append(r)
    steps = cfg["warmup_steps_per_expert"]
    print(f"  [warmup] {C} experts × {steps} steps")
    model.train()
    expert_params = [p for l in moe_layers for p in (l.lora_A_local, l.lora_B_local)]
    opt = torch.optim.AdamW(expert_params, lr=cfg["lr_moe"], weight_decay=0.01)
    for eid in range(C):
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
            lbl  = batch["labels"].to(device)
            B    = iids.shape[0]
            rw   = torch.ones(B, 1, device=device)
            ri   = torch.full((B, 1), eid, dtype=torch.long, device=device)
            set_router(moe_layers, rw, ri)
            loss = model(input_ids=iids, attention_mask=attn, labels=lbl).loss
            opt.zero_grad()
            loss.backward()
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
def measure_routing_accuracy(router, linker, eval_qa, model, moe_layers,
                              device, routing_mode) -> float:
    router.eval()
    correct = total = 0
    bs = 64
    for i in range(0, len(eval_qa), bs):
        batch = eval_qa[i: i + bs]
        queries    = [r["query"] for r in batch]
        true_comms = [r["community"] for r in batch]

        tokenizer_out = linker
        if routing_mode == "moe_topo":
            origs = [r["original_id"] for r in batch]
            struct_emb = linker.batch_link_oracle(origs, device)
            h_dummy = torch.zeros(len(batch), router.text_head.in_features, device=device)
            _, indices, _ = router(h_dummy, struct_emb)
        else:
            return float("nan")

        pred_comms = indices[:, 0].cpu().tolist()
        for pred, true in zip(pred_comms, true_comms):
            if pred == true:
                correct += 1
            total += 1
    return correct / total if total > 0 else float("nan")

@torch.no_grad()
def evaluate(model, router, linker, moe_layers, eval_qa,
             tokenizer, device, routing_mode,
             batch_size=16, max_new=20) -> Tuple[float, int]:
    model.eval()
    if router is not None:
        router.eval()
    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id  = eos_ids[0] if eos_ids else tokenizer.eos_token_id

    correct = total = 0
    route_correct = route_total = 0

    for i in range(0, len(eval_qa), batch_size):
        recs    = eval_qa[i: i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in recs
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=64).to(device)

        if routing_mode == "oracle":
            comm_ids = torch.tensor([r["community"] for r in recs],
                                    dtype=torch.long, device=device)
            rw = torch.ones(len(recs), 1, device=device)
            ri = comm_ids.unsqueeze(1)
            set_router(moe_layers, rw, ri)

        elif routing_mode in ("moe_topo", "moe_text"):
            h_mean = get_mean_repr(model, enc["input_ids"], enc["attention_mask"])

            if routing_mode == "moe_topo":
                struct_emb = linker.batch_link_text(
                    [r["query"] for r in recs], device
                )
                rw, ri, logits = router(h_mean.float(), struct_emb)
            else:
                rw, ri, logits = router(h_mean.float(), None)

            set_router(moe_layers, rw, ri)

            pred = ri[:, 0].cpu().tolist()
            true = [r["community"] for r in recs]
            for p, t in zip(pred, true):
                route_correct += int(p == t)
                route_total   += 1

        gen = model.generate(
            enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if routing_mode in ("oracle", "moe_topo", "moe_text"):
            clear_router(moe_layers)

        plen = enc["input_ids"].shape[1]
        for j, rec in enumerate(recs):
            gen_text = tokenizer.decode(
                gen[j, plen:], skip_special_tokens=True
            ).upper()
            if rec["label"] in gen_text:
                correct += 1
            total += 1

    em = correct / total if total > 0 else float("nan")
    route_acc = route_correct / route_total if route_total > 0 else float("nan")
    return em, total, route_acc

def run_one(nodes_per_comm: int, arch: str, routing_mode: str,
            graph: Dict, lap_pe_np: np.ndarray,
            tokenizer, cfg: Dict, device: str,
            output_dir: str, rng: random.Random,
            max_steps: int, labels: List[str]) -> Dict:

    C = cfg["num_communities"]
    N = nodes_per_comm * C
    print(f"\n{'='*62}")
    print(f"  nodes/comm={nodes_per_comm}  N={N}  arch={arch}  routing={routing_mode}")
    print(f"{'='*62}")
    os.makedirs(output_dir, exist_ok=True)

    train_qa, eval_qa = generate_qa(graph, cfg["eval_frac"], rng, labels)

    lap_pe_t = torch.tensor(lap_pe_np, dtype=torch.float32)
    linker   = AnchorLinker(graph["inv_perm"], lap_pe_t)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )

    router     = None
    moe_layers = []
    _orig_fwd  = None

    if arch == "single_lora":
        model, _ = inject_moe_lora(
            model, rank=BASE_RANK, lora_alpha=BASE_RANK * 2.0,
            num_local_experts=0, use_global_expert=True, top_k=1,
        )
        moe_layers = get_moe_layers(model)

    else:
        model, _shared_router = inject_moe_lora(
            model, rank=BASE_RANK, lora_alpha=BASE_RANK * 2.0,
            num_local_experts=C, use_global_expert=True, top_k=cfg["top_k"],
        )
        moe_layers = get_moe_layers(model)

        _orig_fwd = GlobalLocalLoraLinear.forward
        def _patched(self, x, rw=None, ri=None):
            return _orig_fwd(self, x,
                             rw if rw is not None else getattr(self, "_cached_rw", None),
                             ri if ri is not None else getattr(self, "_cached_ri", None))
        GlobalLocalLoraLinear.forward = _patched

        hidden_size = model.config.hidden_size
        if routing_mode == "moe_topo":
            router = TopoAwareRouter(hidden_size, STRUCT_DIM, C,
                                     top_k=cfg["top_k"]).to(device)
        elif routing_mode == "moe_text":
            router = TextOnlyRouter(hidden_size, C, top_k=cfg["top_k"]).to(device)

        for p in _shared_router.parameters():
            p.requires_grad_(False)

        warmup_experts(model, moe_layers, train_qa, tokenizer, C, cfg, device)

    seen      = set()
    trainable = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); trainable.append(p)
    if router is not None:
        for p in router.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); trainable.append(p)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  trainable={n_trainable:,}")

    scheduler_epochs = cfg["epochs"]
    ds = QADataset(train_qa, tokenizer, cfg["max_length"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    collate_fn=_collate, num_workers=0)
    steps_per_epoch  = max(1, len(ds) // cfg["batch_size"])
    total_steps      = min(max_steps, steps_per_epoch * scheduler_epochs)
    total_steps      = max(total_steps, 100)

    oracle_sched = None
    if arch.startswith("moe") and routing_mode not in ("oracle",):
        oracle_sched = OracleAnnealScheduler(
            total_steps,
            oracle_fraction=cfg["oracle_fraction"],
            anneal_fraction=cfg["anneal_fraction"],
        )
        print(f"  {oracle_sched}")

    lr   = cfg["lr_moe"] if arch.startswith("moe") else cfg["lr_single"]
    opt  = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    if router is not None:
        router.train()
    it      = iter(dl)
    running = step = 0
    route_sup_w = cfg["route_sup_weight"]

    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        iids      = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        lbl       = batch["labels"].to(device)
        comm_lbls = batch["community"].to(device)
        orig_ids  = batch["original_ids"].tolist()

        loss = None

        if arch == "single_lora":
            out  = model(input_ids=iids, attention_mask=attn, labels=lbl)
            loss = out.loss

        elif routing_mode == "oracle":
            rw = torch.ones(iids.shape[0], 1, device=device)
            ri = comm_lbls.unsqueeze(1)
            set_router(moe_layers, rw, ri)
            out  = model(input_ids=iids, attention_mask=attn, labels=lbl)
            loss = out.loss
            clear_router(moe_layers)

        else:
            epsilon = oracle_sched.get_epsilon(step)

            if epsilon < 1.0:
                rw = torch.ones(iids.shape[0], 1, device=device)
                ri = comm_lbls.unsqueeze(1)
                set_router(moe_layers, rw, ri)

            h_mean = get_mean_repr(model, iids, attn)
            if routing_mode == "moe_topo":
                struct_emb = linker.batch_link_oracle(orig_ids, device)
                rw_learned, ri_learned, logits = router(h_mean.float(), struct_emb)
            else:
                rw_learned, ri_learned, logits = router(h_mean.float(), None)

            if epsilon >= 1.0:
                set_router(moe_layers, rw_learned, ri_learned)

            out      = model(input_ids=iids, attention_mask=attn, labels=lbl)
            lm_loss  = out.loss

            r_loss   = F.cross_entropy(logits, comm_lbls)
            loss     = lm_loss + route_sup_w * r_loss
            clear_router(moe_layers)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        opt.step()
        sched.step()
        running += (loss.item() if arch == "single_lora"
                    else out.loss.item())
        step += 1

        if step % 200 == 0:
            eps_str = (f" ε={oracle_sched.get_epsilon(step):.2f}"
                       if oracle_sched else "")
            print(f"    step {step:5d}/{total_steps}"
                  f" | lm={out.loss.item():.4f}"
                  f"{eps_str}", flush=True)

    avg_lm_loss = running / step
    print(f"  train done — avg_lm={avg_lm_loss:.4f}  steps={step}")

    model.eval()
    print("  [debug] sample generations (topo-linked routing):")
    for rec in eval_qa[:3]:
        prompt = (f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        enc_s  = tokenizer(prompt, return_tensors="pt").to(device)
        eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

        if arch.startswith("moe") and routing_mode != "single_lora":
            if routing_mode == "oracle":
                ri_s = torch.tensor([[rec["community"]]], device=device)
                set_router(moe_layers, torch.ones(1, 1, device=device), ri_s)
            elif routing_mode == "moe_topo":
                pe_s = linker.link_oracle(rec["original_id"]).unsqueeze(0).to(device)
                h_s  = get_mean_repr(model, enc_s["input_ids"],
                                     enc_s["attention_mask"])
                rw_s, ri_s, _ = router(h_s.float(), pe_s)
                set_router(moe_layers, rw_s, ri_s)
            else:
                h_s  = get_mean_repr(model, enc_s["input_ids"],
                                     enc_s["attention_mask"])
                rw_s, ri_s, _ = router(h_s.float(), None)
                set_router(moe_layers, rw_s, ri_s)

        gen_s    = model.generate(enc_s["input_ids"], max_new_tokens=24,
                                  do_sample=False, eos_token_id=eos_id,
                                  pad_token_id=tokenizer.pad_token_id
                                  or tokenizer.eos_token_id)
        if arch.startswith("moe") and routing_mode != "single_lora":
            clear_router(moe_layers)
        plen_s   = enc_s["input_ids"].shape[1]
        gen_text = tokenizer.decode(gen_s[0, plen_s:], skip_special_tokens=True)
        correct  = "✓" if rec["label"] in gen_text.upper() else "✗"
        print(f"    {correct} GT={rec['label']:7s} → Gen: {gen_text!r}")

    actual_routing = routing_mode if arch.startswith("moe") else "none"
    em, n_valid, route_acc = evaluate(
        model, router, linker, moe_layers, eval_qa,
        tokenizer, device, actual_routing,
        batch_size=cfg["eval_batch_size"],
    )
    route_str = (f", route_acc={route_acc:.3f}" if not math.isnan(route_acc) else "")
    print(f"  EM={em:.3f} (n={n_valid}){route_str}")

    result = {
        "arch":          arch,
        "routing_mode":  routing_mode,
        "nodes_per_comm": nodes_per_comm,
        "total_nodes":   N,
        "avg_lm_loss":   avg_lm_loss,
        "em":            em,
        "route_acc":     route_acc if not math.isnan(route_acc) else None,
        "n_valid":       n_valid,
        "trainable":     n_trainable,
        "steps":         step,
    }
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    if arch.startswith("moe"):
        GlobalLocalLoraLinear.forward = _orig_fwd
    del model
    if router:
        del router
    torch.cuda.empty_cache()
    return result

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes",    nargs="+", type=int, default=[128, 256, 512])
    p.add_argument("--archs",    nargs="+",
                   default=["single_lora", "moe_oracle", "moe_text", "moe_topo"])
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--gpu",      type=int,  default=0)
    p.add_argument("--output_dir", default="outputs/topo_aware_test")
    p.add_argument("--seed",     type=int,  default=42)
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = dict(CFG)
    cfg["seed"]   = args.seed
    cfg["device"] = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = cfg["device"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    C = cfg["num_communities"]
    labels = COMMUNITY_LABELS[:C]

    print(f"[topo_aware] device={device}")
    print(f"[topo_aware] sizes={args.sizes}  archs={args.archs}")
    print(f"[topo_aware] BASE_RANK={BASE_RANK} | STRUCT_DIM={STRUCT_DIM}")
    print(f"[topo_aware] labels={labels}")
    print(f"[topo_aware] oracle_fraction={cfg['oracle_fraction']} | "
          f"anneal_fraction={cfg['anneal_fraction']}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng     = random.Random(args.seed)
    results = []

    for size in args.sizes:
        N = size * C
        print(f"\n{'─'*62}")
        print(f"  Generating SBM N={N} ({size}/community) + Laplacian PE …")
        graph = generate_sbm(
            size, C,
            cfg["p_in"], cfg["p_out"],
            seed=args.seed,
            display_shuffle_seed=args.seed + 1,
        )
        print(f"  Computing Laplacian PE (k={STRUCT_DIM}) …", end=" ", flush=True)
        lap_pe_np = compute_laplacian_pe(graph["edges"], N, k=STRUCT_DIM)
        print(f"done  shape={lap_pe_np.shape}")

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        X   = lap_pe_np
        y   = np.array([graph["community_map"][v] for v in range(N)])
        clf = LogisticRegression(max_iter=500, C=10.0).fit(X, y)
        lap_clf_acc = accuracy_score(y, clf.predict(X))
        print(f"  LapPE linear-clf accuracy (in-sample): {lap_clf_acc:.3f}  "
              f"({'✓ strong' if lap_clf_acc > 0.9 else '⚠ weak'} structural signal)")

        for arch in args.archs:
            if arch == "single_lora":
                arch_type, routing = "single_lora", "none"
            elif arch == "moe_oracle":
                arch_type, routing = "moe", "oracle"
            elif arch == "moe_text":
                arch_type, routing = "moe", "moe_text"
            elif arch == "moe_topo":
                arch_type, routing = "moe", "moe_topo"
            else:
                raise ValueError(f"Unknown arch: {arch}")

            run_dir = os.path.join(args.output_dir, f"n{N}_{arch}")
            try:
                r = run_one(
                    nodes_per_comm=size,
                    arch=arch_type,
                    routing_mode=routing,
                    graph=graph,
                    lap_pe_np=lap_pe_np,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    output_dir=run_dir,
                    rng=rng,
                    max_steps=args.max_steps,
                    labels=labels,
                )
                r["arch_label"] = arch
                results.append(r)
                with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"[topo_aware] FAILED {arch} N={N}: {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'='*80}")
    print("  TOPOLOGY-AWARE ROUTER CAPACITY TEST")
    print(f"  rank={BASE_RANK} per expert | {C} communities | "
          f"trivial baseline = {1/C:.1%}")
    print(f"{'='*80}")
    print(f"  {'N':>6} | {'arch':14} | {'EM':>6} | {'RouteAcc':>9} | verdict")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*6}-+-{'-'*9}-+-{'-'*20}")

    by_n: Dict[int, Dict] = {}
    for r in results:
        n   = r["total_nodes"]
        lbl = r["arch_label"]
        by_n.setdefault(n, {})[lbl] = r

    for n in sorted(by_n):
        for lbl in ["single_lora", "moe_oracle", "moe_text", "moe_topo"]:
            if lbl not in by_n[n]:
                continue
            r     = by_n[n][lbl]
            em    = r["em"]
            racc  = r.get("route_acc") or float("nan")
            racc_s = f"{racc:.3f}" if not math.isnan(racc) else "  —  "

            if lbl == "moe_oracle":
                verdict = "upper bound"
            elif lbl == "single_lora":
                verdict = "baseline"
            elif lbl == "moe_topo":
                ref = by_n[n].get("moe_text", {}).get("em", float("nan"))
                delta = em - ref if not math.isnan(ref) else float("nan")
                verdict = (f"Δ+{delta:.3f} vs text-route"
                           if not math.isnan(delta) else "topo proposed")
            else:
                verdict = "text-only (expected fail)"
            print(f"  {n:>6} | {lbl:14} | {em:.3f} | {racc_s:>9} | {verdict}")
    print()

if __name__ == "__main__":
    main()
