
from __future__ import annotations

import argparse
import csv
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
from src.model.moe_lora import GlobalLocalLoraLinear, VariableRankGlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, inject_variable_rank_moe_lora, get_moe_layers

from topo_aware_moe import (
    TopoAwareRouter,
    TextOnlyRouter,
    OracleAnnealScheduler,
    compute_laplacian_pe,
    set_router,
    clear_router,
    get_mean_repr,
    STRUCT_DIM,
)

MODEL_PATH = (
    "/home/USER/.cache/huggingface/hub/"
    "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

CFG: Dict = {
    "base_model":         MODEL_PATH,
    "cora_dir":           "../Cora/cora_dataset",
    "rewritten_dir":      "../Cora/sft_data/rewritten",
    "output_dir":         "outputs/cora_topo",
    "n_communities":      8,
    "struct_dim":         STRUCT_DIM,
    "rank_budget":        64,
    "global_rank":        16,
    "lora_alpha":         32.0,
    "global_alpha":       32.0,
    "top_k":              2,
    "num_epochs":         2,
    "batch_size":         2,
    "lr":                 5e-4,
    "max_length":         256,
    "max_train_per_bucket": 800,
    "max_eval_per_bucket":  150,
    "route_sup_weight":   1.0,
    "oracle_fraction":    0.5,
    "anneal_fraction":    0.3,
    "warmup_steps_per_expert": 50,
    "grad_clip":          1.0,
    "eval_batch_size":    8,
    "seed":               42,
}

BUCKETS = ["intra", "cross", "global"]
TITLE_PAT = re.compile(r"<([^>]+)>")

def load_cora_graph(cora_dir: str) -> Tuple[object, Dict[int, str], Dict[str, int]]:
    import networkx as nx

    all_csv = os.path.join(cora_dir, "all.csv")
    edges_csv = os.path.join(cora_dir, "edges.csv")

    node2title: Dict[int, str] = {}
    title2node: Dict[str, int] = {}

    with open(all_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row["id"])
            title = row["T"].strip()
            node2title[nid] = title
            title2node[title.lower()] = nid

    G = nx.Graph()
    for nid in node2title:
        G.add_node(nid)

    with open(edges_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src, tgt = int(row["source"]), int(row["target"])
            G.add_edge(src, tgt)

    print(f"[cora] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, node2title, title2node

def build_topo_partition(
    G, n_communities: int = 8, seed: int = 42
) -> Tuple[Dict[int, int], List[int]]:
    try:
        import community as community_louvain
        raw = community_louvain.best_partition(G, random_state=seed)
    except ImportError:
        import networkx.algorithms.community as nxc
        comms = nxc.louvain_communities(G, seed=seed)
        raw = {n: i for i, c in enumerate(comms) for n in c}

    comm_sizes = Counter(raw.values())
    top_comms = [c for c, _ in comm_sizes.most_common(n_communities)]
    remap = {c: i for i, c in enumerate(top_comms)}
    top_set = set(top_comms)

    node2part: Dict[int, int] = {}
    deferred: List[int] = []
    for node, comm in raw.items():
        if comm in remap:
            node2part[node] = remap[comm]
        else:
            deferred.append(node)

    for _ in range(3):
        still_deferred = []
        for node in deferred:
            nb_parts = [node2part[nb] for nb in G.neighbors(node) if nb in node2part]
            if nb_parts:
                node2part[node] = Counter(nb_parts).most_common(1)[0][0]
            else:
                still_deferred.append(node)
        deferred = still_deferred
        if not deferred:
            break
    for node in deferred:
        node2part[node] = 0

    part_sizes = [0] * n_communities
    for pid in node2part.values():
        part_sizes[pid] += 1

    print(f"[partition] {n_communities} communities: sizes = {part_sizes}")
    return node2part, part_sizes

def compute_rank_allocation(
    part_sizes: List[int],
    rank_budget: int,
    min_rank: int = 4,
    max_rank: int = 128,
) -> List[int]:
    total = sum(part_sizes)
    ranks = [max(min_rank, min(max_rank, int(rank_budget * s / total)))
             for s in part_sizes]
    print(f"[capacity] Expert rank allocation: {ranks}  (budget={rank_budget})")
    return ranks

class CoraAnchorLinker:

    def __init__(self, title2node: Dict[str, int], lap_pe: torch.Tensor) -> None:
        self.title2node = title2node
        self.lap_pe = lap_pe

    def link_oracle(self, node_id: int) -> torch.Tensor:
        return self.lap_pe[node_id]

    def batch_link_oracle(self, node_ids: List[int], device: torch.device) -> torch.Tensor:
        return torch.stack([self.lap_pe[nid] for nid in node_ids]).to(device)

    def link_text(self, query: str) -> Optional[torch.Tensor]:
        for title in TITLE_PAT.findall(query):
            nid = self.title2node.get(title.lower().strip())
            if nid is not None:
                return self.lap_pe[nid]
        return None

    def batch_link_text(
        self, queries: List[str], device: torch.device
    ) -> Optional[torch.Tensor]:
        pes, n_fail = [], 0
        zero = torch.zeros(self.lap_pe.shape[1])
        for q in queries:
            pe = self.link_text(q)
            if pe is None:
                n_fail += 1
                pes.append(zero)
            else:
                pes.append(pe)
        if n_fail == len(queries):
            return None
        return torch.stack(pes).to(device)

def load_cora_records(
    rewritten_dir: str,
    node2part: Dict[int, int],
    title2node: Dict[str, int],
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    file_bucket = [
        ("01_existence_qa.jsonl",  None),
        ("02_counting_qa.jsonl",   "intra"),
        ("03_traversal_qa.jsonl",  "global"),
        ("04_substructure_qa.jsonl", "global"),
        ("05_multihop_qa.jsonl",   "cross"),
    ]
    records: List[Dict] = []
    for fname, default_bucket in file_bucket:
        fpath = os.path.join(rewritten_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        rng.shuffle(lines)

        for item in lines:
            if "query" not in item or "answer" not in item:
                continue
            titles = [t.strip() for t in TITLE_PAT.findall(item["query"])]
            if not titles:
                continue

            anchor_title = titles[0]
            anchor_nid = title2node.get(anchor_title.lower())
            if anchor_nid is None:
                continue
            anchor_part = node2part.get(anchor_nid)
            if anchor_part is None:
                continue

            if default_bucket is None:
                if len(titles) >= 2:
                    t2 = titles[1]
                    nid2 = title2node.get(t2.lower())
                    part2 = node2part.get(nid2) if nid2 else None
                    bucket = "intra" if (part2 is None or part2 == anchor_part) else "cross"
                else:
                    bucket = "intra"
            else:
                bucket = default_bucket

            records.append({
                "query":      item["query"],
                "answer":     item["answer"],
                "community":  anchor_part,
                "node_id":    anchor_nid,
                "bucket":     bucket,
            })

    print(f"[data] Loaded {len(records)} records  "
          f"(buckets: {dict(Counter(r['bucket'] for r in records))})")
    return records

def split_records(
    records: List[Dict],
    max_train_per_bucket: int,
    max_eval_per_bucket: int,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    by_bucket: Dict[str, List] = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)
    train, val = [], []
    for bucket, items in by_bucket.items():
        rng.shuffle(items)
        n_val   = min(max_eval_per_bucket,  max(1, len(items) // 7))
        n_train = min(max_train_per_bucket, len(items) - n_val)
        val.extend(items[:n_val])
        train.extend(items[n_val: n_val + n_train])
    rng.shuffle(train); rng.shuffle(val)
    print(f"[data] Train={len(train)}, Eval={len(val)}")
    return train, val

class CoraDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int) -> None:
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self._asst = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        text = (
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['answer']}<|im_end|>"
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        ids = input_ids.tolist()
        n = len(self._asst)
        for i in range(len(ids) - n):
            if ids[i: i + n] == self._asst:
                labels[: i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels":    labels,
            "community": rec["community"],
            "node_id":   rec["node_id"],
            "bucket":    BUCKETS.index(rec["bucket"]) if rec["bucket"] in BUCKETS else 0,
            "query":     rec["query"],
        }

def _collate(batch: List[Dict]) -> Dict:
    max_l = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    iids  = torch.zeros(B, max_l, dtype=torch.long)
    lbls  = torch.full((B, max_l), -100, dtype=torch.long)
    attn  = torch.zeros(B, max_l, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    nids  = torch.zeros(B, dtype=torch.long)
    bkts  = torch.zeros(B, dtype=torch.long)
    queries = []
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        iids[i, :L] = b["input_ids"]
        lbls[i, :L] = b["labels"]
        attn[i, :L] = 1
        comms[i]    = b["community"]
        nids[i]     = b["node_id"]
        bkts[i]     = b["bucket"]
        queries.append(b["query"])
    return {"input_ids": iids, "labels": lbls, "attention_mask": attn,
            "community": comms, "node_id": nids, "bucket": bkts, "queries": queries}

def warmup_vr_experts(
    model: nn.Module,
    moe_layers: List,
    train_records: List[Dict],
    tokenizer,
    n_communities: int,
    cfg: Dict,
    device: str,
) -> None:
    by_comm: Dict[int, List] = defaultdict(list)
    for r in train_records:
        by_comm[r["community"]].append(r)

    steps = cfg["warmup_steps_per_expert"]
    print(f"\n[warmup] {n_communities} experts × {steps} steps each")
    model.train()

    for eid in range(n_communities):
        recs = by_comm.get(eid, [])
        if not recs:
            print(f"[warmup] Expert {eid}: no samples, skipping")
            continue
        print(f"[warmup] Expert {eid}: {len(recs)} samples", flush=True)

        expert_params = []
        for layer in moe_layers:
            if hasattr(layer, "lora_A_locals"):
                expert_params += layer.expert_parameters(eid)
            elif hasattr(layer, "lora_A_local"):
                expert_params.extend([layer.lora_A_local, layer.lora_B_local])

        opt = torch.optim.AdamW(expert_params, lr=cfg["lr"], weight_decay=0.01)
        ds  = CoraDataset(recs, tokenizer, cfg["max_length"])
        dl  = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                         collate_fn=_collate, num_workers=0)
        it  = iter(dl)
        last_loss = torch.tensor(0.0)

        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            iids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            B    = iids.shape[0]

            rw = torch.ones(B, 1, device=device)
            ri = torch.full((B, 1), eid, dtype=torch.long, device=device)
            set_router(moe_layers, rw, ri)
            last_loss = model(input_ids=iids, attention_mask=attn, labels=lbls).loss
            opt.zero_grad()
            last_loss.backward()

            for layer in moe_layers:
                if hasattr(layer, "lora_A_local"):
                    for p in (layer.lora_A_local, layer.lora_B_local):
                        if p.grad is not None:
                            mask = torch.zeros_like(p.grad)
                            mask[eid] = 1.0
                            p.grad.mul_(mask)

            torch.nn.utils.clip_grad_norm_(expert_params, cfg["grad_clip"])
            opt.step()
            clear_router(moe_layers)

        print(f"[warmup] Expert {eid} done (last loss={last_loss.item():.4f})", flush=True)

    print("[warmup] All experts warmed up.\n", flush=True)

@torch.no_grad()
def evaluate_em(
    model: nn.Module,
    router: Optional[nn.Module],
    linker: CoraAnchorLinker,
    moe_layers: List,
    eval_records: List[Dict],
    tokenizer,
    device: str,
    routing_mode: str,
    batch_size: int = 8,
    max_new_tokens: int = 40,
) -> Dict:
    model.eval()
    if router is not None:
        router.eval()

    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id  = eos_ids[0] if eos_ids else tokenizer.eos_token_id

    bucket_correct:  Dict[str, int] = defaultdict(int)
    bucket_total:    Dict[str, int] = defaultdict(int)
    route_correct, route_total = 0, 0

    def _parse_yn(text: str) -> Optional[str]:
        t = text.lower()
        if "yes" in t or "indeed" in t or "direct" in t or "exist" in t:
            return "yes"
        if "no" in t or "not" in t or "negative" in t or "none" in t:
            return "no"
        return None

    for i in range(0, len(eval_records), batch_size):
        recs = eval_records[i: i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in recs
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(device)

        if routing_mode == "oracle":
            comm_ids = torch.tensor([r["community"] for r in recs],
                                    dtype=torch.long, device=device)
            set_router(moe_layers, torch.ones(len(recs), 1, device=device),
                       comm_ids.unsqueeze(1))

        elif routing_mode in ("moe_topo", "moe_text"):
            h_mean = get_mean_repr(model, enc["input_ids"], enc["attention_mask"])
            if routing_mode == "moe_topo":
                struct_emb = linker.batch_link_text([r["query"] for r in recs], device)
                rw, ri, _ = router(h_mean.float(), struct_emb)
            else:
                rw, ri, _ = router(h_mean.float(), None)
            set_router(moe_layers, rw, ri)
            pred = ri[:, 0].cpu().tolist()
            for p, rec in zip(pred, recs):
                route_correct += int(p == rec["community"])
                route_total   += 1

        gen = model.generate(
            enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if routing_mode != "single":
            clear_router(moe_layers)

        plen = enc["input_ids"].shape[1]
        for j, rec in enumerate(recs):
            gen_text = tokenizer.decode(gen[j, plen:], skip_special_tokens=True)
            pred_yn  = _parse_yn(gen_text)
            ref_yn   = _parse_yn(rec["answer"])
            bname    = BUCKETS[rec.get("bucket_idx", 0)] if "bucket_idx" in rec else rec.get("bucket", "intra")
            bucket_total[bname]   += 1
            if pred_yn is not None and ref_yn is not None and pred_yn == ref_yn:
                bucket_correct[bname] += 1

    em = {b: (bucket_correct[b] / bucket_total[b] if bucket_total[b] else float("nan"))
          for b in BUCKETS}
    route_acc = route_correct / route_total if route_total > 0 else float("nan")
    return {"em": em, "bucket_total": dict(bucket_total), "route_acc": route_acc}

def train_one(
    model: nn.Module,
    router: Optional[nn.Module],
    linker: CoraAnchorLinker,
    moe_layers: List,
    train_records: List[Dict],
    eval_records: List[Dict],
    tokenizer,
    device: str,
    routing_mode: str,
    cfg: Dict,
    output_dir: str,
) -> Dict:
    ds = CoraDataset(train_records, tokenizer, cfg["max_length"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    collate_fn=_collate, num_workers=0)
    steps_per_epoch = max(1, len(ds) // cfg["batch_size"])
    total_steps     = steps_per_epoch * cfg["num_epochs"]

    oracle_sched: Optional[OracleAnnealScheduler] = None
    if routing_mode not in ("oracle", "single"):
        oracle_sched = OracleAnnealScheduler(
            total_steps,
            oracle_fraction=cfg["oracle_fraction"],
            anneal_fraction=cfg["anneal_fraction"],
        )
        print(f"[train] {oracle_sched}")

    trainable = []
    seen = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); trainable.append(p)
    if router is not None:
        for p in router.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p)); trainable.append(p)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] Trainable parameters: {n_trainable:,}")

    opt  = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total_steps // 10),
                                            num_training_steps=total_steps)

    _orig_fwd = GlobalLocalLoraLinear.forward
    def _patched_uniform(self, x, rw=None, ri=None):
        return _orig_fwd(self, x,
                         rw if rw is not None else getattr(self, "_cached_rw", None),
                         ri if ri is not None else getattr(self, "_cached_ri", None))
    GlobalLocalLoraLinear.forward = _patched_uniform

    _orig_vr_fwd = VariableRankGlobalLocalLoraLinear.forward
    def _patched_vr(self, x, rw=None, ri=None):
        return _orig_vr_fwd(self, x,
                            rw if rw is not None else getattr(self, "_cached_rw", None),
                            ri if ri is not None else getattr(self, "_cached_ri", None))
    VariableRankGlobalLocalLoraLinear.forward = _patched_vr

    global_step = 0
    rng = random.Random(cfg["seed"])
    history = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        if router is not None:
            router.train()
        epoch_lm_loss = 0.0
        epoch_route_loss = 0.0
        n_batches = 0

        for batch in dl:
            iids  = batch["input_ids"].to(device)
            attn  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)
            comms = batch["community"].to(device)
            nids  = batch["node_id"].tolist()

            h_mean = get_mean_repr(model, iids, attn)

            if routing_mode == "moe_topo":
                struct_emb = linker.batch_link_oracle(nids, device)
                rw, ri, logits = router(h_mean.float(), struct_emb)
            elif routing_mode == "moe_text":
                rw, ri, logits = router(h_mean.float(), None)
            else:
                logits = None

            epsilon = oracle_sched.get_epsilon(global_step) if oracle_sched else 1.0

            if rng.random() < epsilon and routing_mode not in ("oracle", "single"):
                routing_rw, routing_ri = rw, ri
            else:
                B = iids.shape[0]
                routing_rw = torch.ones(B, 1, device=device)
                routing_ri = comms.unsqueeze(1)

            if routing_mode != "single":
                set_router(moe_layers, routing_rw, routing_ri)

            outputs  = model(input_ids=iids, attention_mask=attn, labels=lbls)
            lm_loss  = outputs.loss
            clear_router(moe_layers)

            route_loss = torch.tensor(0.0, device=device)
            if logits is not None:
                lam = cfg["route_sup_weight"]
                if oracle_sched is not None:
                    lam = lam * (1.0 - 0.5 * epsilon)
                route_loss = F.cross_entropy(logits, comms) * lam

            total_loss = lm_loss + route_loss
            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            opt.step()
            sched.step()

            epoch_lm_loss    += lm_loss.item()
            epoch_route_loss += route_loss.item()
            n_batches        += 1
            global_step      += 1

        avg_lm    = epoch_lm_loss    / max(n_batches, 1)
        avg_route = epoch_route_loss / max(n_batches, 1)
        eps_end   = oracle_sched.get_epsilon(global_step) if oracle_sched else 1.0
        print(f"[train] Epoch {epoch+1}/{cfg['num_epochs']}  "
              f"lm={avg_lm:.4f}  route={avg_route:.4f}  ε={eps_end:.2f}")
        history.append({"epoch": epoch + 1, "lm": avg_lm, "route": avg_route, "eps": eps_end})

    GlobalLocalLoraLinear.forward = _orig_fwd
    VariableRankGlobalLocalLoraLinear.forward = _orig_vr_fwd

    return {"history": history}

def run_experiment(
    arch: str,
    graph_data: Dict,
    records: Tuple[List[Dict], List[Dict]],
    tokenizer,
    cfg: Dict,
    device: str,
    output_dir: str,
) -> Dict:
    train_records, eval_records = records
    G              = graph_data["G"]
    node2part      = graph_data["node2part"]
    part_sizes     = graph_data["part_sizes"]
    expert_ranks   = graph_data["expert_ranks"]
    lap_pe_t       = graph_data["lap_pe_t"]
    title2node     = graph_data["title2node"]
    n_communities  = cfg["n_communities"]

    routing_mode_map = {
        "single_lora":  "single",
        "moe_oracle":   "oracle",
        "moe_text":     "moe_text",
        "moe_topo":     "moe_topo",
    }
    routing_mode = routing_mode_map[arch]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  arch={arch}  routing={routing_mode}  N={G.number_of_nodes()}")
    print(f"  expert_ranks={expert_ranks}  top_k={cfg['top_k']}")
    print(f"{'='*68}")

    linker = CoraAnchorLinker(title2node, lap_pe_t)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    router: Optional[nn.Module] = None
    moe_layers: List = []

    if arch == "single_lora":
        model, _ = inject_moe_lora(
            model, rank=cfg["rank_budget"],
            lora_alpha=cfg["lora_alpha"] * (cfg["rank_budget"] / cfg["global_rank"]),
            num_local_experts=0, use_global_expert=True, top_k=1,
        )
        moe_layers = get_moe_layers(model)

    else:
        model, _shared_router = inject_variable_rank_moe_lora(
            model,
            expert_ranks=expert_ranks,
            global_rank=cfg["global_rank"],
            lora_alpha=cfg["lora_alpha"],
            global_alpha=cfg["global_alpha"],
            use_global_expert=True,
            top_k=cfg["top_k"],
        )
        moe_layers = get_moe_layers(model)

        for p in _shared_router.parameters():
            p.requires_grad_(False)

        hidden_size = model.config.hidden_size
        if routing_mode == "moe_topo":
            router = TopoAwareRouter(hidden_size, cfg["struct_dim"],
                                     n_communities, top_k=cfg["top_k"]).to(device)
        elif routing_mode == "moe_text":
            router = TextOnlyRouter(hidden_size, n_communities, top_k=cfg["top_k"]).to(device)

        warmup_vr_experts(model, moe_layers, train_records, tokenizer,
                          n_communities, cfg, device)

    train_result = train_one(
        model, router, linker, moe_layers,
        train_records, eval_records,
        tokenizer, device, routing_mode, cfg, output_dir,
    )

    for rec in eval_records:
        rec["bucket_idx"] = BUCKETS.index(rec["bucket"]) if rec["bucket"] in BUCKETS else 0

    eval_result = evaluate_em(
        model, router, linker, moe_layers, eval_records,
        tokenizer, device, routing_mode,
        batch_size=cfg["eval_batch_size"],
    )

    result = {
        "arch":         arch,
        "routing_mode": routing_mode,
        "em":           eval_result["em"],
        "route_acc":    eval_result["route_acc"],
        "bucket_total": eval_result["bucket_total"],
        "train_history": train_result["history"],
    }
    out_path = os.path.join(output_dir, f"{arch}_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[result] {arch}: EM per bucket = {eval_result['em']}, "
          f"route_acc = {eval_result['route_acc']:.3f}")
    print(f"[result] Saved → {out_path}")

    del model
    if router is not None:
        del router
    torch.cuda.empty_cache()

    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Topology-Aware LoRA-MoE on Cora")
    parser.add_argument("--archs", nargs="+",
                        default=["single_lora", "moe_oracle", "moe_text", "moe_topo"],
                        choices=["single_lora", "moe_oracle", "moe_text", "moe_topo"])
    parser.add_argument("--n_communities",  type=int,   default=CFG["n_communities"])
    parser.add_argument("--rank_budget",    type=int,   default=CFG["rank_budget"])
    parser.add_argument("--global_rank",    type=int,   default=CFG["global_rank"])
    parser.add_argument("--epochs",         type=int,   default=CFG["num_epochs"])
    parser.add_argument("--max_train",      type=int,   default=CFG["max_train_per_bucket"])
    parser.add_argument("--max_eval",       type=int,   default=CFG["max_eval_per_bucket"])
    parser.add_argument("--output_dir",     type=str,   default=CFG["output_dir"])
    parser.add_argument("--cora_dir",       type=str,   default=CFG["cora_dir"])
    parser.add_argument("--rewritten_dir",  type=str,   default=CFG["rewritten_dir"])
    parser.add_argument("--seed",           type=int,   default=CFG["seed"])
    args = parser.parse_args()

    CFG.update({
        "n_communities":        args.n_communities,
        "rank_budget":          args.rank_budget,
        "global_rank":          args.global_rank,
        "num_epochs":           args.epochs,
        "max_train_per_bucket": args.max_train,
        "max_eval_per_bucket":  args.max_eval,
        "output_dir":           args.output_dir,
        "cora_dir":             args.cora_dir,
        "rewritten_dir":        args.rewritten_dir,
        "seed":                 args.seed,
    })

    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}, archs={args.archs}")

    print("\n[main] Loading Cora graph ...")
    G, node2title, title2node = load_cora_graph(CFG["cora_dir"])

    print("[main] Building topology partition (Louvain) ...")
    node2part, part_sizes = build_topo_partition(G, CFG["n_communities"], CFG["seed"])

    print("[main] Computing Laplacian PE ...")
    edges_list = list(G.edges())
    lap_pe_np  = compute_laplacian_pe(edges_list, G.number_of_nodes(), k=CFG["struct_dim"])
    lap_pe_t   = torch.tensor(lap_pe_np, dtype=torch.float32)

    expert_ranks = compute_rank_allocation(part_sizes, CFG["rank_budget"])

    print("[main] Loading SFT data ...")
    all_records = load_cora_records(CFG["rewritten_dir"], node2part, title2node, CFG["seed"])
    train_records, eval_records = split_records(
        all_records, CFG["max_train_per_bucket"], CFG["max_eval_per_bucket"], CFG["seed"]
    )

    print(f"[main] Loading tokenizer from {CFG['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["base_model"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    graph_data = {
        "G":            G,
        "node2part":    node2part,
        "part_sizes":   part_sizes,
        "expert_ranks": expert_ranks,
        "lap_pe_t":     lap_pe_t,
        "title2node":   title2node,
    }

    all_results = {}
    for arch in args.archs:
        arch_out = os.path.join(CFG["output_dir"], arch)
        result = run_experiment(
            arch, graph_data, (train_records, eval_records),
            tokenizer, CFG, device, arch_out,
        )
        all_results[arch] = result

    print(f"\n{'='*68}")
    print(f"  SUMMARY  (n_communities={CFG['n_communities']}, rank_budget={CFG['rank_budget']})")
    print(f"{'='*68}")
    header = f"{'arch':<14} {'intra_EM':>9} {'cross_EM':>9} {'global_EM':>10} {'route_acc':>10}"
    print(header)
    print("-" * len(header))
    for arch, res in all_results.items():
        em = res["em"]
        print(f"{arch:<14} {em.get('intra', float('nan')):9.3f} "
              f"{em.get('cross', float('nan')):9.3f} "
              f"{em.get('global', float('nan')):10.3f} "
              f"{res['route_acc']:10.3f}")

    summary_path = os.path.join(CFG["output_dir"], "summary.json")
    os.makedirs(CFG["output_dir"], exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[main] Summary saved → {summary_path}")

if __name__ == "__main__":
    main()
