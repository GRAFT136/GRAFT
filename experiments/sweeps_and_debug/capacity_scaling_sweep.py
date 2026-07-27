
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from phase1_single_lora import _binary_label
from phase1_train import clear_router_decision, set_router_decision
from src.model.injection import get_moe_layers, inject_moe_lora
from src.model.moe_lora import GlobalLocalLoraLinear

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

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

DEFAULT_EDGE_GRID = [10_000, 100_000, 1_000_000, 10_000_000, 20_000_000]

FAMILY_SEED_OFFSETS = {
    "sbm": 101,
    "er": 202,
    "ba": 303,
}

CFG = {
    "avg_degree": 20,
    "sbm_num_communities": 8,
    "sbm_cross_fraction": 0.15,
    "single_ranks": [16, 64, 256],
    "graft_local_rank": 16,
    "graft_min_experts": 8,
    "graft_max_experts": 64,
    "graft_top_k": 1,
    "queries_per_source": 4,
    "train_source_fraction": 0.35,
    "eval_source_fraction": 0.08,
    "min_train_sources": 96,
    "min_eval_sources": 48,
    "max_train_sources": 6000,
    "max_eval_sources": 800,
    "batch_size": 12,
    "eval_batch_size": 24,
    "lr_single": 2e-4,
    "lr_graft": 4e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "epochs": 2,
    "max_steps": 600,
    "max_length": 80,
    "max_new_tokens": 4,
    "seed": 42,
    "cliff_fraction": 0.85,
}

def _device_index_from_name(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0

def _resolve_device(gpu_arg: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if len(visible_ids) == 1:
            return "cuda:0"
        if 0 <= gpu_arg < len(visible_ids):
            return f"cuda:{gpu_arg}"

    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return "cpu"
    if gpu_arg >= device_count:
        return "cuda:0"
    return f"cuda:{gpu_arg}"

def _ensure_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def _load_model(model_path: str, device: str):
    device_index = _device_index_from_name(device)
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        trust_remote_code=True,
    )

def _patch_cached_router_forward() -> callable:
    original = GlobalLocalLoraLinear.forward

    def patched(self, x, rw=None, ri=None):
        use_rw = rw if rw is not None else getattr(self, "_cached_rw", None)
        use_ri = ri if ri is not None else getattr(self, "_cached_ri", None)
        return original(self, x, use_rw, use_ri)

    GlobalLocalLoraLinear.forward = patched
    return original

def _restore_cached_router_forward(original) -> None:
    GlobalLocalLoraLinear.forward = original

def _node_count_from_edges(edge_budget: int, avg_degree: int) -> int:
    return max(avg_degree + 2, int(math.ceil(edge_budget / max(1, avg_degree))))

def _make_log_grid(min_edges: int, max_edges: int, num_points: int) -> List[int]:
    if num_points <= 1:
        return [int(min_edges)]
    values = np.geomspace(min_edges, max_edges, num=num_points)
    out = sorted({int(round(v)) for v in values})
    out[0] = int(min_edges)
    out[-1] = int(max_edges)
    return out

def _community_of(node_id: int, num_nodes: int, num_communities: int) -> int:
    if num_nodes <= 0:
        return 0
    return min(num_communities - 1, ((node_id + 1) * num_communities - 1) // num_nodes)

def _community_bounds(community: int, num_nodes: int, num_communities: int) -> Tuple[int, int]:
    start = (community * num_nodes) // num_communities
    end = ((community + 1) * num_nodes) // num_communities
    return start, max(start + 1, end)

def _sbm_partition(source_node: int, num_nodes: int, num_communities: int, num_experts: int) -> int:
    community = _community_of(source_node, num_nodes, num_communities)
    blocks_per_comm = max(1, int(math.ceil(num_experts / max(1, num_communities))))
    comm_start = (community * num_nodes) // num_communities
    comm_end = ((community + 1) * num_nodes) // num_communities
    local_size = max(1, comm_end - comm_start)
    local_index = max(0, source_node - comm_start)
    subblock = min(blocks_per_comm - 1, int(local_index * blocks_per_comm / local_size))
    expert_id = community * blocks_per_comm + subblock
    return min(num_experts - 1, expert_id)

def _block_partition(source_node: int, num_nodes: int, num_experts: int) -> int:
    return min(num_experts - 1, int(source_node * num_experts / max(1, num_nodes)))

def _sample_train_eval_sources(
    num_nodes: int,
    min_node_id: int,
    cfg: Dict[str, float],
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    candidates = list(range(min_node_id, num_nodes))
    rng.shuffle(candidates)
    n_candidates = len(candidates)
    n_train = min(cfg["max_train_sources"], max(cfg["min_train_sources"], int(n_candidates * cfg["train_source_fraction"])))
    n_eval = min(cfg["max_eval_sources"], max(cfg["min_eval_sources"], int(n_candidates * cfg["eval_source_fraction"])))
    if n_train + n_eval > n_candidates:
        n_eval = max(1, min(n_eval, n_candidates // 5))
        n_train = max(1, min(n_train, n_candidates - n_eval))
    eval_nodes = sorted(candidates[:n_eval])
    train_nodes = sorted(candidates[n_eval:n_eval + n_train])
    return train_nodes, eval_nodes

def _sample_unique_uniform(
    rng: random.Random,
    upper: int,
    count: int,
    banned: Optional[set] = None,
) -> List[int]:
    banned = set() if banned is None else set(banned)
    out: List[int] = []
    while len(out) < count:
        cand = rng.randrange(upper)
        if cand in banned:
            continue
        banned.add(cand)
        out.append(cand)
    return out

def _build_er_selected_adjacency(
    num_nodes: int,
    avg_degree: int,
    selected_nodes: Sequence[int],
    seed: int,
) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {}
    for node_id in selected_nodes:
        rng = random.Random(seed * 1_000_003 + node_id * 97)
        adjacency[node_id] = _sample_unique_uniform(rng, num_nodes, avg_degree, banned={node_id})
    return adjacency

def _build_sbm_selected_adjacency(
    num_nodes: int,
    avg_degree: int,
    num_communities: int,
    cross_fraction: float,
    selected_nodes: Sequence[int],
    seed: int,
) -> Dict[int, List[int]]:
    adjacency: Dict[int, List[int]] = {}
    n_cross = int(round(avg_degree * cross_fraction))
    n_intra = max(0, avg_degree - n_cross)
    for node_id in selected_nodes:
        rng = random.Random(seed * 1_000_003 + node_id * 193)
        community = _community_of(node_id, num_nodes, num_communities)
        comm_start = (community * num_nodes) // num_communities
        comm_end = ((community + 1) * num_nodes) // num_communities
        used = {node_id}
        targets: List[int] = []

        while len(targets) < n_intra:
            cand = rng.randrange(comm_start, comm_end)
            if cand in used:
                continue
            used.add(cand)
            targets.append(cand)

        while len(targets) < avg_degree:
            cand = rng.randrange(num_nodes)
            if cand in used:
                continue
            if _community_of(cand, num_nodes, num_communities) == community:
                continue
            used.add(cand)
            targets.append(cand)

        adjacency[node_id] = targets
    return adjacency

if NUMBA_AVAILABLE:

    @njit
    def _numba_ba_selected_targets(num_nodes: int, m: int, selected_nodes: np.ndarray, seed: int) -> np.ndarray:
        np.random.seed(seed)
        selected_index = np.full(num_nodes, -1, dtype=np.int32)
        for idx in range(selected_nodes.shape[0]):
            selected_index[selected_nodes[idx]] = idx

        out = np.full((selected_nodes.shape[0], m), -1, dtype=np.int32)
        initial = m + 1
        rep_cap = 2 * m * num_nodes + initial * initial
        repeated = np.empty(rep_cap, dtype=np.int32)
        rep_len = 0

        for u in range(initial):
            for v in range(u):
                repeated[rep_len] = u
                rep_len += 1
                repeated[rep_len] = v
                rep_len += 1

        for u in range(initial, num_nodes):
            chosen = np.empty(m, dtype=np.int32)
            count = 0
            while count < m:
                cand = repeated[np.random.randint(0, rep_len)]
                duplicate = False
                for j in range(count):
                    if chosen[j] == cand:
                        duplicate = True
                        break
                if duplicate:
                    continue
                chosen[count] = cand
                count += 1

            sel_idx = selected_index[u]
            if sel_idx >= 0:
                for j in range(m):
                    out[sel_idx, j] = chosen[j]

            for j in range(m):
                repeated[rep_len] = u
                rep_len += 1
                repeated[rep_len] = chosen[j]
                rep_len += 1

        return out

def _build_ba_selected_adjacency(
    num_nodes: int,
    avg_degree: int,
    selected_nodes: Sequence[int],
    seed: int,
) -> Dict[int, List[int]]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the large-scale BA sweep in this environment")
    selected = np.asarray(sorted(selected_nodes), dtype=np.int32)
    matrix = _numba_ba_selected_targets(num_nodes, avg_degree, selected, seed)
    adjacency: Dict[int, List[int]] = {}
    for row, node_id in enumerate(selected.tolist()):
        targets = [int(v) for v in matrix[row].tolist() if int(v) >= 0]
        adjacency[node_id] = targets
    return adjacency

def _generate_selected_adjacency(
    family: str,
    num_nodes: int,
    avg_degree: int,
    selected_nodes: Sequence[int],
    cfg: Dict[str, float],
    seed: int,
) -> Dict[int, List[int]]:
    if family == "er":
        return _build_er_selected_adjacency(num_nodes, avg_degree, selected_nodes, seed)
    if family == "sbm":
        return _build_sbm_selected_adjacency(
            num_nodes=num_nodes,
            avg_degree=avg_degree,
            num_communities=cfg["sbm_num_communities"],
            cross_fraction=cfg["sbm_cross_fraction"],
            selected_nodes=selected_nodes,
            seed=seed,
        )
    if family == "ba":
        return _build_ba_selected_adjacency(num_nodes, avg_degree, selected_nodes, seed)
    raise ValueError(f"Unknown family: {family}")

def _build_records(
    family: str,
    num_nodes: int,
    adjacency: Dict[int, List[int]],
    avg_degree: int,
    cfg: Dict[str, float],
    rng: random.Random,
    num_experts: Optional[int] = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    per_source = cfg["queries_per_source"]
    n_positive = max(1, per_source // 2)
    n_negative = max(1, per_source - n_positive)
    num_communities = cfg["sbm_num_communities"]

    for source, neighbors in adjacency.items():
        neigh_set = set(neighbors)
        if family == "sbm":
            base_partition = _community_of(source, num_nodes, num_communities)
            comm_start, comm_end = _community_bounds(base_partition, num_nodes, num_communities)
        else:
            base_partition = 0

        if num_experts is None:
            partition = base_partition
        elif family == "sbm":
            partition = _sbm_partition(source, num_nodes, num_communities, num_experts)
        else:
            partition = _block_partition(source, num_nodes, num_experts)

        if len(neighbors) > n_positive:
            chosen_pos = rng.sample(neighbors, n_positive)
        else:
            chosen_pos = list(neighbors)
        for target in chosen_pos:
            records.append({
                "query": f"In graph G, does node_{source} have a direct edge to node_{target}? Answer only Yes or No.",
                "answer": "Yes.",
                "label": 1,
                "partition": partition,
            })

        negatives_added = 0
        local_attempts = 0
        while negatives_added < n_negative:
            if family == "sbm":
                cand = rng.randrange(comm_start, comm_end)
                if cand == source or cand in neigh_set:
                    local_attempts += 1
                    if local_attempts < max(16, 2 * avg_degree):
                        continue
                    community_span = comm_end - comm_start
                    start_offset = rng.randrange(max(1, community_span))
                    cand = -1
                    for offset in range(community_span):
                        probe = comm_start + ((start_offset + offset) % community_span)
                        if probe != source and probe not in neigh_set:
                            cand = probe
                            break
                    if cand < 0:
                        cand = rng.randrange(num_nodes)
            else:
                cand = rng.randrange(num_nodes)
            if cand == source or cand in neigh_set:
                continue
            records.append({
                "query": f"In graph G, does node_{source} have a direct edge to node_{cand}? Answer only Yes or No.",
                "answer": "No.",
                "label": 0,
                "partition": partition,
            })
            negatives_added += 1

    rng.shuffle(records)
    return records

class NNPDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]], tokenizer, max_length: int) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._assistant = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        text = (
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['answer']}<|im_end|>"
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        token_ids = input_ids.tolist()
        n = len(self._assistant)
        for i in range(len(token_ids) - n):
            if token_ids[i:i + n] == self._assistant:
                labels[: i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "partition": int(rec["partition"]),
            "label": int(rec["label"]),
        }

def _collate(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    partitions = torch.zeros(B, dtype=torch.long)
    gt = torch.zeros(B, dtype=torch.long)
    for idx, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[idx, :length] = item["input_ids"]
        labels[idx, :length] = item["labels"]
        attn[idx, :length] = 1
        partitions[idx] = int(item["partition"])
        gt[idx] = int(item["label"])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
        "partition": partitions,
        "label": gt,
    }

def _dedup_trainable(params: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    seen = set()
    out = []
    for param in params:
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        out.append(param)
    return out

def _parse_yes(text: str) -> bool:
    return _binary_label(text) == "pos"

@torch.no_grad()
def evaluate_neighbor_f1(
    model,
    tokenizer,
    eval_records: Sequence[Dict[str, object]],
    device: str,
    max_new_tokens: int,
    batch_size: int,
    moe_layers: Optional[List[GlobalLocalLoraLinear]] = None,
    oracle_route: bool = False,
) -> Dict[str, float]:
    model.eval()
    eos_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_ids[0] if eos_ids else tokenizer.eos_token_id

    tp = fp = fn = correct = 0
    total = 0

    for start in range(0, len(eval_records), batch_size):
        batch = eval_records[start:start + batch_size]
        prompts = [f"<|im_start|>user\n{rec['query']}<|im_end|>\n<|im_start|>assistant\n" for rec in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

        if oracle_route and moe_layers:
            B = len(batch)
            rw = torch.ones(B, 1, dtype=torch.float32, device=device)
            ri = torch.tensor([[int(rec["partition"])] for rec in batch], dtype=torch.long, device=device)
            set_router_decision(moe_layers, rw, ri)

        out_ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if oracle_route and moe_layers:
            clear_router_decision(moe_layers)

        prompt_len = enc["input_ids"].shape[1]
        for idx, rec in enumerate(batch):
            generated = tokenizer.decode(out_ids[idx, prompt_len:], skip_special_tokens=True).strip()
            pred_pos = _parse_yes(generated)
            gt_pos = bool(rec["label"])
            if pred_pos and gt_pos:
                tp += 1
            elif pred_pos and not gt_pos:
                fp += 1
            elif (not pred_pos) and gt_pos:
                fn += 1
            correct += int(pred_pos == gt_pos)
            total += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = correct / max(1, total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_eval": total,
    }

def _training_steps(num_records: int, batch_size: int, epochs: int, max_steps: int) -> int:
    steps = math.ceil(num_records / max(1, batch_size)) * max(1, epochs)
    return max(80, min(max_steps, steps))

def _run_single_lora(
    family: str,
    edge_budget: int,
    rank: int,
    model_path: str,
    tokenizer,
    train_records: Sequence[Dict[str, object]],
    eval_records: Sequence[Dict[str, object]],
    cfg: Dict[str, float],
    device: str,
) -> Dict[str, object]:
    model = _load_model(model_path, device)
    model, _ = inject_moe_lora(
        model,
        rank=rank,
        lora_alpha=rank * 2.0,
        num_local_experts=0,
        use_global_expert=True,
        top_k=1,
    )
    moe_layers = get_moe_layers(model)
    trainable = _dedup_trainable(model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr_single"], weight_decay=cfg["weight_decay"])

    train_ds = NNPDataset(train_records, tokenizer, cfg["max_length"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=_collate, num_workers=0)
    total_steps = _training_steps(len(train_ds), cfg["batch_size"], cfg["epochs"], cfg["max_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    iterator = iter(train_dl)
    running = 0.0
    model.train()
    for _ in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        running += float(loss.detach().cpu())

    metrics = evaluate_neighbor_f1(
        model=model,
        tokenizer=tokenizer,
        eval_records=eval_records,
        device=device,
        max_new_tokens=cfg["max_new_tokens"],
        batch_size=cfg["eval_batch_size"],
        moe_layers=moe_layers,
        oracle_route=False,
    )

    avg_train_loss = running / max(1, total_steps)
    trainable_params = sum(param.numel() for param in trainable)
    del model
    torch.cuda.empty_cache()
    return {
        "family": family,
        "edge_budget": edge_budget,
        "arch": "single_lora",
        "rank": rank,
        "num_experts": 1,
        "avg_train_loss": avg_train_loss,
        "trainable_params": trainable_params,
        **metrics,
    }

def _run_graft_oracle(
    edge_budget: int,
    num_experts: int,
    model_path: str,
    tokenizer,
    train_records: Sequence[Dict[str, object]],
    eval_records: Sequence[Dict[str, object]],
    cfg: Dict[str, float],
    device: str,
) -> Dict[str, object]:
    model = _load_model(model_path, device)
    model, _ = inject_moe_lora(
        model,
        rank=cfg["graft_local_rank"],
        lora_alpha=cfg["graft_local_rank"] * 2.0,
        num_local_experts=num_experts,
        use_global_expert=False,
        top_k=cfg["graft_top_k"],
    )
    original_forward = _patch_cached_router_forward()
    moe_layers = get_moe_layers(model)
    trainable = _dedup_trainable(model.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr_graft"], weight_decay=cfg["weight_decay"])

    train_ds = NNPDataset(train_records, tokenizer, cfg["max_length"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=_collate, num_workers=0)
    total_steps = _training_steps(len(train_ds), cfg["batch_size"], cfg["epochs"], cfg["max_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    iterator = iter(train_dl)
    running = 0.0
    model.train()
    for _ in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            batch = next(iterator)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        partitions = batch["partition"].to(device)
        rw = torch.ones(partitions.shape[0], 1, dtype=torch.float32, device=device)
        ri = partitions.view(-1, 1)
        set_router_decision(moe_layers, rw, ri)
        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        clear_router_decision(moe_layers)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        running += float(loss.detach().cpu())

    metrics = evaluate_neighbor_f1(
        model=model,
        tokenizer=tokenizer,
        eval_records=eval_records,
        device=device,
        max_new_tokens=cfg["max_new_tokens"],
        batch_size=cfg["eval_batch_size"],
        moe_layers=moe_layers,
        oracle_route=True,
    )

    avg_train_loss = running / max(1, total_steps)
    trainable_params = sum(param.numel() for param in trainable)
    _restore_cached_router_forward(original_forward)
    del model
    torch.cuda.empty_cache()
    return {
        "family": "sbm",
        "edge_budget": edge_budget,
        "arch": "graft_oracle",
        "rank": cfg["graft_local_rank"],
        "num_experts": num_experts,
        "avg_train_loss": avg_train_loss,
        "trainable_params": trainable_params,
        **metrics,
    }

def _estimate_cliff(results: Sequence[Dict[str, object]], rank: int, cliff_fraction: float) -> Optional[int]:
    sbm_rows = [row for row in results if row["family"] == "sbm" and row["arch"] == "single_lora" and row["rank"] == rank]
    if not sbm_rows:
        return None
    sbm_rows = sorted(sbm_rows, key=lambda row: row["edge_budget"])
    reference = max(row["f1"] for row in sbm_rows[: min(2, len(sbm_rows))])
    threshold = reference * cliff_fraction
    for row in sbm_rows:
        if row["f1"] < threshold:
            return int(row["edge_budget"])
    return int(sbm_rows[-1]["edge_budget"])

def _plot_figures(results: Sequence[Dict[str, object]], output_dir: str, rank_for_family_plot: int) -> None:
    output_path = Path(output_dir)

    plt.figure(figsize=(8, 5))
    for rank in sorted({int(row["rank"]) for row in results if row["family"] == "sbm" and row["arch"] == "single_lora"}):
        rows = sorted(
            [row for row in results if row["family"] == "sbm" and row["arch"] == "single_lora" and int(row["rank"]) == rank],
            key=lambda row: row["edge_budget"],
        )
        plt.plot([row["edge_budget"] for row in rows], [row["f1"] for row in rows], marker="o", label=f"Single LoRA r={rank}")
    graft_rows = sorted(
        [row for row in results if row["family"] == "sbm" and row["arch"] == "graft_oracle"],
        key=lambda row: row["edge_budget"],
    )
    if graft_rows:
        plt.plot(
            [row["edge_budget"] for row in graft_rows],
            [row["f1"] for row in graft_rows],
            marker="s",
            linewidth=2.5,
            linestyle="--",
            label=f"GRAFT oracle (local r={graft_rows[0]['rank']})",
        )
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Edge Budget M")
    plt.ylabel("N-F1")
    plt.title("Figure 2a Reproduction: SBM Capacity Cliff")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "fig2a_sbm_capacity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for family, color in [("sbm", "#1f77b4"), ("er", "#ff7f0e"), ("ba", "#2ca02c")]:
        rows = sorted(
            [row for row in results if row["family"] == family and row["arch"] == "single_lora" and int(row["rank"]) == rank_for_family_plot],
            key=lambda row: row["edge_budget"],
        )
        if not rows:
            continue
        plt.plot(
            [row["edge_budget"] for row in rows],
            [row["f1"] for row in rows],
            marker="o",
            color=color,
            label=f"{family.upper()} single r={rank_for_family_plot}",
        )
    if graft_rows:
        plt.plot(
            [row["edge_budget"] for row in graft_rows],
            [row["f1"] for row in graft_rows],
            marker="s",
            linestyle="--",
            color="#111111",
            label="SBM GRAFT oracle",
        )
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Edge Budget M")
    plt.ylabel("N-F1")
    plt.title("Figure 2b Reproduction: Degree-Skew Sensitivity")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "fig2b_family_sensitivity.png", dpi=200)
    plt.close()

def _write_results_artifacts(
    results: Sequence[Dict[str, object]],
    cliff_by_rank: Dict[int, Optional[int]],
    output_dir: str,
    rank_for_family_plot: int,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "results.json", "w", encoding="utf-8") as handle:
        json.dump(list(results), handle, ensure_ascii=False, indent=2)

    with open(output_path / "results_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "arch", "rank", "edge_budget", "num_experts", "f1", "precision", "recall", "accuracy", "avg_train_loss", "trainable_params"])
        for row in results:
            writer.writerow([
                row["family"],
                row["arch"],
                row["rank"],
                row["edge_budget"],
                row["num_experts"],
                f"{float(row['f1']):.6f}",
                f"{float(row['precision']):.6f}",
                f"{float(row['recall']):.6f}",
                f"{float(row['accuracy']):.6f}",
                f"{float(row['avg_train_loss']):.6f}",
                row["trainable_params"],
            ])

    lines: List[str] = []
    lines.append("# Capacity Scaling Sweep Report")
    lines.append("")
    lines.append("This sweep uses directed neighbor prediction (NNP) on synthetic graphs with fixed average out-degree 20.")
    lines.append("")
    lines.append("## SBM Capacity Cliff")
    lines.append("")
    lines.append("| Edge Budget M | r=16 | r=64 | r=256 | GRAFT oracle | GRAFT K |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    sbm_edges = sorted({int(row["edge_budget"]) for row in results if row["family"] == "sbm"})
    for edge_budget in sbm_edges:
        single_rows = {
            int(row["rank"]): row
            for row in results
            if row["family"] == "sbm" and row["arch"] == "single_lora" and int(row["edge_budget"]) == edge_budget
        }
        graft_row = next((row for row in results if row["family"] == "sbm" and row["arch"] == "graft_oracle" and int(row["edge_budget"]) == edge_budget), None)
        graft_f1 = float(graft_row["f1"]) if graft_row else float("nan")
        graft_k = graft_row["num_experts"] if graft_row else "N/A"
        lines.append(
            f"| {edge_budget:,} | {single_rows.get(16, {}).get('f1', float('nan')):.3f} | {single_rows.get(64, {}).get('f1', float('nan')):.3f} | {single_rows.get(256, {}).get('f1', float('nan')):.3f} | {graft_f1:.3f} | {graft_k} |"
        )
    lines.append("")
    lines.append("## Empirical Cliff Points")
    lines.append("")
    for rank in sorted(cliff_by_rank):
        edge_budget = cliff_by_rank[rank]
        if edge_budget is None:
            lines.append(f"- rank {rank}: cliff not observed in the tested range")
        else:
            lines.append(f"- rank {rank}: estimated M*(r) ≈ {edge_budget:,}")
    lines.append("")
    lines.append(f"## Family Sensitivity at rank {rank_for_family_plot}")
    lines.append("")
    lines.append("| Family | Best F1 | Worst F1 | Mean F1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for family in ["sbm", "er", "ba"]:
        family_rows = [row for row in results if row["family"] == family and row["arch"] == "single_lora" and int(row["rank"]) == rank_for_family_plot]
        if not family_rows:
            continue
        f1s = [float(row["f1"]) for row in family_rows]
        lines.append(f"| {family.upper()} | {max(f1s):.3f} | {min(f1s):.3f} | {mean(f1s):.3f} |")
    lines.append("")
    lines.append("## Plot Files")
    lines.append("")
    lines.append("- fig2a_sbm_capacity.png")
    lines.append("- fig2b_family_sensitivity.png")
    lines.append("")
    with open(output_path / "report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    _plot_figures(results, output_dir, rank_for_family_plot)

def _prepare_family_data(
    family: str,
    edge_budget: int,
    cfg: Dict[str, float],
    seed: int,
    num_experts: Optional[int] = None,
) -> Tuple[int, List[Dict[str, object]], List[Dict[str, object]]]:
    num_nodes = _node_count_from_edges(edge_budget, cfg["avg_degree"])
    min_node_id = cfg["avg_degree"] + 1 if family == "ba" else 0
    rng = random.Random(seed)
    train_nodes, eval_nodes = _sample_train_eval_sources(num_nodes, min_node_id, cfg, rng)
    selected_nodes = train_nodes + eval_nodes
    adjacency = _generate_selected_adjacency(
        family=family,
        num_nodes=num_nodes,
        avg_degree=cfg["avg_degree"],
        selected_nodes=selected_nodes,
        cfg=cfg,
        seed=seed,
    )

    train_adjacency = {node_id: adjacency[node_id] for node_id in train_nodes}
    eval_adjacency = {node_id: adjacency[node_id] for node_id in eval_nodes}
    train_records = _build_records(family, num_nodes, train_adjacency, cfg["avg_degree"], cfg, rng, num_experts=num_experts)
    eval_records = _build_records(family, num_nodes, eval_adjacency, cfg["avg_degree"], cfg, rng, num_experts=num_experts)
    return num_nodes, train_records, eval_records

def run_sweep(args) -> List[Dict[str, object]]:
    cfg = dict(CFG)
    cfg["seed"] = args.seed
    cfg["epochs"] = args.epochs
    cfg["max_steps"] = args.max_steps
    cfg["batch_size"] = args.batch_size
    cfg["eval_batch_size"] = args.eval_batch_size
    cfg["graft_local_rank"] = args.graft_local_rank
    cfg["graft_min_experts"] = args.graft_min_experts
    cfg["graft_max_experts"] = args.graft_max_experts
    cfg["queries_per_source"] = args.queries_per_source
    cfg["avg_degree"] = args.avg_degree
    cfg["sbm_num_communities"] = args.sbm_num_communities
    cfg["sbm_cross_fraction"] = args.sbm_cross_fraction

    edge_grid = args.edge_counts or _make_log_grid(args.min_edges, args.max_edges, args.num_points)
    model_path = MODEL_PATHS[args.model]
    tokenizer = _ensure_tokenizer(model_path)
    device = _resolve_device(args.gpu)

    print(f"[capacity] device={device} model={args.model} edge_grid={edge_grid}")
    print(f"[capacity] single ranks={args.single_ranks} families={args.families}")

    results: List[Dict[str, object]] = []

    for family in args.families:
        for edge_budget in edge_grid:
            num_nodes, train_records, eval_records = _prepare_family_data(
                family=family,
                edge_budget=edge_budget,
                cfg=cfg,
                seed=args.seed + FAMILY_SEED_OFFSETS[family] * 1_000_000 + edge_budget,
                num_experts=None,
            )
            print(f"[capacity] family={family} M={edge_budget:,} N={num_nodes:,} train={len(train_records)} eval={len(eval_records)}")
            for rank in args.single_ranks:
                print(f"  [single] family={family} M={edge_budget:,} rank={rank}")
                result = _run_single_lora(
                    family=family,
                    edge_budget=edge_budget,
                    rank=rank,
                    model_path=model_path,
                    tokenizer=tokenizer,
                    train_records=train_records,
                    eval_records=eval_records,
                    cfg=cfg,
                    device=device,
                )
                result["num_nodes"] = num_nodes
                results.append(result)
                print(f"    -> F1={result['f1']:.3f} acc={result['accuracy']:.3f}")

    cliff_edge = _estimate_cliff(results, args.graft_local_rank, cfg["cliff_fraction"])
    if cliff_edge is None:
        cliff_edge = edge_grid[-1]
    c_star = max(edge_grid[0], cliff_edge)
    print(f"[capacity] empirical c* from SBM rank {args.graft_local_rank}: {c_star:,}")

    if args.run_graft:
        for edge_budget in edge_grid:
            num_experts = max(cfg["graft_min_experts"], int(math.ceil(edge_budget / max(1, c_star))))
            num_experts = min(cfg["graft_max_experts"], num_experts)
            num_nodes, train_records, eval_records = _prepare_family_data(
                family="sbm",
                edge_budget=edge_budget,
                cfg=cfg,
                seed=args.seed + 77 + edge_budget % 10_000,
                num_experts=num_experts,
            )
            num_experts = min(num_experts, max(1, len({int(rec['partition']) for rec in train_records})))
            print(f"  [graft] SBM M={edge_budget:,} K={num_experts}")
            if num_experts != max(int(rec["partition"]) for rec in train_records) + 1:
                for rec in train_records:
                    rec["partition"] = min(num_experts - 1, int(rec["partition"]))
                for rec in eval_records:
                    rec["partition"] = min(num_experts - 1, int(rec["partition"]))
            result = _run_graft_oracle(
                edge_budget=edge_budget,
                num_experts=num_experts,
                model_path=model_path,
                tokenizer=tokenizer,
                train_records=train_records,
                eval_records=eval_records,
                cfg=cfg,
                device=device,
            )
            result["num_nodes"] = num_nodes
            result["c_star"] = c_star
            results.append(result)
            print(f"    -> GRAFT F1={result['f1']:.3f} acc={result['accuracy']:.3f}")

    cliff_by_rank = {rank: _estimate_cliff(results, rank, cfg["cliff_fraction"]) for rank in args.single_ranks if rank in args.single_ranks and any(row['family'] == 'sbm' and row['arch'] == 'single_lora' and int(row['rank']) == rank for row in results)}
    _write_results_artifacts(results, cliff_by_rank, args.output_dir, args.family_plot_rank)
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["0.5B", "3B", "7B"], default="0.5B")
    parser.add_argument("--families", nargs="+", default=["sbm", "er", "ba"])
    parser.add_argument("--single_ranks", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--edge_counts", nargs="+", type=int, default=None)
    parser.add_argument("--min_edges", type=int, default=10_000)
    parser.add_argument("--max_edges", type=int, default=20_000_000)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--avg_degree", type=int, default=20)
    parser.add_argument("--sbm_num_communities", type=int, default=8)
    parser.add_argument("--sbm_cross_fraction", type=float, default=0.15)
    parser.add_argument("--queries_per_source", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/capacity_scaling")
    parser.add_argument("--run_graft", action="store_true")
    parser.add_argument("--graft_local_rank", type=int, default=16)
    parser.add_argument("--graft_min_experts", type=int, default=8)
    parser.add_argument("--graft_max_experts", type=int, default=64)
    parser.add_argument("--family_plot_rank", type=int, default=64)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_sweep(args)

if __name__ == "__main__":
    main()