
import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import (
    inject_moe_lora, get_moe_layers, configure_incremental_update,
    expand_local_experts_to_rank,
)
from src.model.losses import compute_total_loss
from phase1_train import (
    CLASS_NAMES, BUCKETS,
    load_phase1_records, split_records,
    build_title2class, _extract_titles,
    Phase1Dataset, collate_fn,
    set_router_decision, clear_router_decision,
)
from phase1_single_lora import SingleLoraLinear, inject_single_lora, check_em, _classify
from src.data.other_sft_loader import (
    load_other_sft_records, num_communities as other_num_communities,
)

def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "all":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]

def _collect_trainable_state(model, router=None):
    state = {}
    for name, p in model.named_parameters():
        if ("lora_" in name) or name.startswith("router."):
            state[name] = p.detach().cpu()
    if router is not None:
        for name, p in router.named_parameters():
            key = f"router.{name}"
            if key not in state:
                state[key] = p.detach().cpu()
    return state

def _load_trainable_state(model, state_path: str, strict_shapes: bool = False):
    ckpt = torch.load(state_path, map_location="cpu")
    state = ckpt.get("trainable_state", ckpt)
    named = dict(model.named_parameters())

    loaded = 0
    skipped = 0
    for name, tensor in state.items():
        if name not in named:
            skipped += 1
            continue
        param = named[name]
        if tuple(param.shape) != tuple(tensor.shape):
            if strict_shapes:
                raise ValueError(f"Shape mismatch for {name}: model={tuple(param.shape)} ckpt={tuple(tensor.shape)}")
            skipped += 1
            continue
        with torch.no_grad():
            param.copy_(tensor.to(param.device, dtype=param.dtype))
        loaded += 1

    print(f"[state] Loaded trainable params from {state_path}: loaded={loaded}, skipped={skipped}")

def _first_int(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None

def _simple_binary_label(text: str) -> str:
    t = text.lower().strip()
    if t.startswith(("yes", "indeed", "correct")):
        return "pos"
    if t.startswith(("no", "negative")) or "does not" in t or "not " in t:
        return "neg"
    return "unk"

def _perturb_answer(query: str, answer: str) -> Optional[str]:
    q_type = _classify(query)
    if q_type == "counting":
        n = _first_int(answer)
        if n is None:
            return None
        return re.sub(r"\b\d+\b", str(n + 1), answer, count=1)

    label = _simple_binary_label(answer)
    if label == "pos":
        return "No."
    if label == "neg":
        return "Yes."
    return None

def _apply_label_flip_perturbation(records, communities: Optional[List[int]], split_name: str):
    if communities is None:
        return records
    selected = {int(c) for c in communities}
    out = []
    changed = 0
    selected_total = 0
    for rec in records:
        new_rec = dict(rec)
        new_rec["_perturbed"] = False
        if rec.get("community") in selected:
            selected_total += 1
            new_answer = _perturb_answer(rec["query"], rec["answer"])
            if new_answer is not None and new_answer != rec["answer"]:
                new_rec["answer"] = new_answer
                new_rec["_perturbed"] = True
                changed += 1
        out.append(new_rec)
    print(f"[perturb] {split_name}: communities={sorted(selected)} changed={changed}/{selected_total}")
    return out

def _force_perturbed_route(router_weights, router_indices, records, expert_id: Optional[int]):
    if expert_id is None:
        return router_weights, router_indices
    rows = [i for i, rec in enumerate(records) if rec.get("_perturbed", False)]
    if not rows:
        return router_weights, router_indices

    mask = torch.zeros(router_weights.shape[0], device=router_weights.device, dtype=torch.bool)
    mask[torch.tensor(rows, device=router_weights.device, dtype=torch.long)] = True
    return _force_route_by_mask(router_weights, router_indices, mask, expert_id)

def _force_route_by_mask(router_weights, router_indices, mask, expert_id: Optional[int]):
    if expert_id is None or mask is None or not bool(mask.any().item()):
        return router_weights, router_indices

    rw = router_weights.clone()
    ri = router_indices.clone()
    row_idx = mask.to(rw.device).nonzero(as_tuple=False).squeeze(-1)
    rw[row_idx] = 0.0
    rw[row_idx, 0] = 1.0
    ri[row_idx] = int(expert_id)
    return rw, ri

OTHER_DATASETS = ("citeseer", "amazon-photo", "amazon-computers", "wn18rr")

def load_records_multi(rewritten_dir: str, cora_dir: str, seed: int = 42):
    title2class = build_title2class(cora_dir)
    rng = random.Random(seed)
    file_info = [
        ("01_existence_qa.jsonl", None),
        ("02_counting_qa.jsonl", "intra"),
        ("03_traversal_qa.jsonl", "global"),
        ("04_substructure_qa.jsonl", "global"),
        ("05_multihop_qa.jsonl", "cross"),
    ]
    records = []
    for fname, default_bucket in file_info:
        fpath = os.path.join(rewritten_dir, fname)
        with open(fpath, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        rng.shuffle(lines)
        for item in lines:
            if "query" not in item or "answer" not in item:
                continue
            titles = _extract_titles(item["query"])
            if not titles:
                continue
            c0 = title2class.get(titles[0])
            if c0 is None:
                continue
            communities = [c0]
            for t in titles[1:]:
                c = title2class.get(t)
                if c is not None and c not in communities:
                    communities.append(c)
            communities = communities[:2]

            if default_bucket is None:
                c1 = communities[1] if len(communities) >= 2 else None
                bucket = "intra" if (c1 is None or c1 == c0) else "cross"
            else:
                bucket = default_bucket

            records.append({
                "query": item["query"],
                "answer": item["answer"],
                "community": c0,
                "communities": communities,
                "bucket": bucket,
            })
    return records

BASE_CFG = {
    "base_model": (
        "/home/USER/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    ),
    "cora_dir": "../Cora/cora_dataset",
    "rewritten_dir": "../Cora/sft_data/rewritten",
    "other_sft_root": "../other_sft_data",
    "other_graph_root": "../other_graph_dataset",
    "num_local_experts": 7,
    "rank": 16,
    "lora_alpha": 32.0,
    "use_global_expert": True,
    "top_k": 2,
    "num_epochs": 3,
    "batch_size": 8,
    "max_length": 256,
    "max_train_per_bucket": 250,
    "max_eval_per_bucket": 90,
    "aux_loss_weight": 0.01,
    "route_sup_weight": 0.5,
    "route_sup_anneal": True,
    "class_weighted_route_sup": True,
    "aux_underload_penalty": 3.0,
    "entropy_aux_weight": 0.0,
    "expert_warmup": True,
    "warmup_steps_per_expert": 30,
    "grad_clip": 1.0,
    "eval_batch_size": 16,
    "seed": 42,
}

def resolve_model_path(pattern: str) -> str:
    p = Path(pattern)
    if p.is_dir() and any(p.iterdir()):
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if subdirs:
            return str(subdirs[0])
        return str(p)
    return pattern

VARIANT_OVERRIDES = {
    "single_lora": {
        "rank": 128, "lora_alpha": 256.0, "lr": 1e-4,
        "expert_warmup": False, "route_sup_weight": 0.0,
    },
    "moe_sparse": {
        "lr": 5e-4,
    },
    "moe_dense": {
        "lr": 2e-4, "entropy_aux_weight": 0.0,
    },
    "moe_dense_entropy": {
        "lr": 2e-4, "entropy_aux_weight": 0.05,
    },
    "moe_dense_multitarget": {
        "lr": 2e-4, "entropy_aux_weight": 0.0,
    },
    "moe_dense_nowarmup": {
        "lr": 2e-4, "entropy_aux_weight": 0.0, "expert_warmup": False,
    },
    "moe_dense_lightwarmup": {
        "lr": 2e-4, "entropy_aux_weight": 0.0,
        "warmup_steps_per_expert": 10, "num_epochs": 4,
    },
    "moe_dense_nowarmup_multitarget": {
        "lr": 2e-4, "entropy_aux_weight": 0.0, "expert_warmup": False,
    },
    "moe_dense_nowarmup_multitarget_5ep": {
        "lr": 2e-4, "entropy_aux_weight": 0.0, "expert_warmup": False, "num_epochs": 5,
    },
    "moe_dense_lightwarmup2_multitarget": {
        "lr": 2e-4, "entropy_aux_weight": 0.0,
        "warmup_steps_per_expert": 15, "num_epochs": 5,
    },
    "moe_sparse_amatch": {
        "lr": 3e-4, "expert_warmup": False, "num_epochs": 5,
    },
}

def _soft_target_vec(communities, num_experts):
    v = [0.0] * num_experts
    for c in communities:
        v[c] += 1.0 / len(communities)
    return v

class MultiTargetDataset(Phase1Dataset):
    def __init__(self, records, tokenizer, max_length, num_experts):
        super().__init__(records, tokenizer, max_length)
        self.num_experts = num_experts

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        rec = self.records[idx]
        item["soft_target"] = torch.tensor(
            _soft_target_vec(rec.get("communities", [rec["community"]]), self.num_experts),
            dtype=torch.float32,
        )
        item["perturbed"] = bool(rec.get("_perturbed", False))
        return item

def collate_fn_multitarget(batch):
    out = collate_fn(batch)
    out["soft_target"] = torch.stack([b["soft_target"] for b in batch], dim=0)
    out["perturbed"] = torch.tensor([bool(b.get("perturbed", False)) for b in batch], dtype=torch.bool)
    return out

class DenseSoftRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        inner = max(256, num_experts * 4)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=True),
            nn.GELU(),
            nn.Linear(inner, num_experts, bias=True),
        )

    def forward(self, query_repr):
        logits = self.gate(query_repr)
        weights = F.softmax(logits, dim=-1)
        _, topk_idx = weights.topk(self.top_k, dim=-1)
        return weights, topk_idx, logits

def _dense_forward(self, x, router_weights=None, router_indices=None):
    rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)

    orig_shape = x.shape
    orig_dtype = x.dtype
    x_flat = x.reshape(-1, self.in_features)
    T = x_flat.shape[0]

    h = F.linear(x_flat, self.base_weight, self.base_bias)
    dev = x_flat.device
    x_lora = x_flat.to(torch.float32)

    if self.use_global_expert:
        A_g = self.lora_A_global.to(dev)
        B_g = self.lora_B_global.to(dev)
        h = h + ((x_lora @ A_g.T) @ B_g.T).to(orig_dtype) * self.scaling

    if rw is not None:
        B_batch, E = rw.shape
        tokens_per_seq = T // B_batch if T != B_batch else 1
        rw_exp = rw.to(dev).unsqueeze(1).expand(B_batch, tokens_per_seq, E).reshape(T, E)

        expert_delta = torch.zeros_like(h)
        for eid in range(E):
            w = rw_exp[:, eid]
            A = self.lora_A_local[eid].to(dev)
            B = self.lora_B_local[eid].to(dev)
            out = ((x_lora @ A.T) @ B.T).to(orig_dtype) * self.scaling
            expert_delta += w.to(orig_dtype).unsqueeze(-1) * out
        h = h + expert_delta

    return h.reshape(orig_shape[:-1] + (self.out_features,))

def entropy_aux_loss(gate_weights):
    E = gate_weights.shape[-1]
    entropy = -(gate_weights * gate_weights.clamp(min=1e-9).log()).sum(-1)
    return math.log(E) - entropy.mean()

def _run_expert_warmup_dense(model, moe_layers, all_records, tokenizer, cfg, device):
    num_experts = cfg["num_local_experts"]
    warmup_steps = cfg.get("warmup_steps_per_expert", 30)
    from phase1_train import _RecordDataset
    model.train()
    print(f"\n[warmup-dense] {num_experts} experts x {warmup_steps} steps each")
    for eid in range(num_experts):
        comm_records = [r for r in all_records if r.get("community", -1) == eid]
        if not comm_records:
            print(f"[warmup-dense] Expert {eid}: no samples, skipping")
            continue
        ds = _RecordDataset(comm_records, tokenizer, cfg["max_length"])
        dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
        dl_iter = iter(dl)
        expert_params = []
        for layer in moe_layers:
            expert_params.append(layer.lora_A_local)
            expert_params.append(layer.lora_B_local)
        opt = torch.optim.AdamW(expert_params, lr=cfg["lr"], weight_decay=0.01)

        loss = torch.tensor(0.0)
        for step in range(warmup_steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            B = input_ids.shape[0]

            rw = torch.zeros(B, num_experts, device=device)
            rw[:, eid] = 1.0
            set_router_decision(moe_layers, rw, None)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert_params, cfg["grad_clip"])
            opt.step()
            clear_router_decision(moe_layers)
        print(f"[warmup-dense] Expert {eid} ({CLASS_NAMES[eid]}) done (last loss={loss.item():.4f})", flush=True)
    print("[warmup-dense] All experts warmed up.\n", flush=True)

@torch.no_grad()
def em_eval(model, eval_records, tokenizer, device, batch_size, max_new=60,
            router=None, moe_layers=None, is_dense=False,
            oracle_route_perturbed_to: Optional[int] = None,
            target_communities: Optional[List[int]] = None):
    eos_list = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_list[0] if eos_list else tokenizer.eos_token_id
    bucket_results = defaultdict(list)
    group_results = defaultdict(list)

    for i in range(0, len(eval_records), batch_size):
        batch = eval_records[i:i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=280).to(device)
        prompt_len = enc["input_ids"].shape[1]

        if router is not None:
            embed = model.model.embed_tokens(enc["input_ids"])
            mask_f = enc["attention_mask"].unsqueeze(-1).float()
            qr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            rw, ri, _ = router(qr.to(torch.float32))
            rw, ri = _force_perturbed_route(rw, ri, batch, oracle_route_perturbed_to)
            set_router_decision(moe_layers, rw, ri)

        gen_ids = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=eos_id,
        )
        if router is not None:
            clear_router_decision(moe_layers)

        for j, rec in enumerate(batch):
            new_toks = gen_ids[j][prompt_len:]
            generated = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            q_type = _classify(rec["query"])
            correct = check_em(generated, rec["answer"], q_type)
            bucket_results[rec["bucket"]].append(correct)
            group = "changed" if rec.get("_perturbed", False) else "unchanged"
            group_results[group].append(correct)
            if target_communities is not None and not rec.get("_perturbed", False):
                target_set = {int(c) for c in target_communities}
                detail_group = "target_unchanged" if rec.get("community") in target_set else "other_unchanged"
                group_results[detail_group].append(correct)

        if (i // batch_size) % 5 == 0:
            print(f"  [em] {min(i+batch_size,len(eval_records))}/{len(eval_records)}", flush=True)

    stats = {}
    for b, results in bucket_results.items():
        valid = [r for r in results if r is not None]
        stats[b] = {
            "acc": sum(valid) / len(valid) if valid else float("nan"),
            "n_valid": len(valid),
            "n_total": len(results),
            "pct_valid": len(valid) / len(results) if results else 0,
        }
    if group_results:
        stats["_groups"] = {}
        for group, results in group_results.items():
            valid = [r for r in results if r is not None]
            stats["_groups"][group] = {
                "acc": sum(valid) / len(valid) if valid else float("nan"),
                "n_valid": len(valid),
                "n_total": len(results),
                "pct_valid": len(valid) / len(results) if results else 0,
            }
    return stats

@torch.no_grad()
def train_loss_diagnostic(model, records, tokenizer, device, batch_size, max_length,
                          router=None, moe_layers=None,
                          oracle_route_perturbed_to: Optional[int] = None):
    model.eval()
    if router is not None:
        router.eval()

    def _loss_for(sub_records):
        if not sub_records:
            return {"loss": float("nan"), "n": 0}
        ds = Phase1Dataset(sub_records, tokenizer, max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        total_loss = 0.0
        total_n = 0
        offset = 0
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            raw_batch = sub_records[offset: offset + input_ids.shape[0]]
            offset += input_ids.shape[0]

            if router is not None:
                embed = model.model.embed_tokens(input_ids)
                mask_f = attn_mask.unsqueeze(-1).float()
                qr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                rw, ri, _ = router(qr.to(torch.float32))
                rw, ri = _force_perturbed_route(rw, ri, raw_batch, oracle_route_perturbed_to)
                set_router_decision(moe_layers, rw, ri)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            if router is not None:
                clear_router_decision(moe_layers)

            n = input_ids.shape[0]
            total_loss += float(outputs.loss.detach().cpu()) * n
            total_n += n
        return {"loss": total_loss / total_n if total_n else float("nan"), "n": total_n}

    changed = [r for r in records if r.get("_perturbed", False)]
    unchanged = [r for r in records if not r.get("_perturbed", False)]
    return {
        "changed": _loss_for(changed),
        "unchanged": _loss_for(unchanged),
        "all": _loss_for(records),
    }

def router_diagnostic(model, router, eval_records, tokenizer, device, batch_size, num_experts, top_k):
    bucket_top1 = defaultdict(list)
    bucket_topk = defaultdict(list)
    for i in range(0, len(eval_records), batch_size):
        batch = eval_records[i:i + batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=280).to(device)
        with torch.no_grad():
            embed = model.model.embed_tokens(enc["input_ids"])
            mask_f = enc["attention_mask"].unsqueeze(-1).float()
            query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            _, topk_idx, logits = router(query_repr.to(torch.float32))
        top1_pred = logits.argmax(-1).tolist()
        topk_list = topk_idx.tolist()
        for j, rec in enumerate(batch):
            true_comms = set(rec.get("communities", [rec["community"]]))
            bucket_top1[rec["bucket"]].append(top1_pred[j] in true_comms)
            bucket_topk[rec["bucket"]].append(bool(true_comms & set(topk_list[j])))

    print(f"\n=== Router routing-accuracy diagnostic (num_experts={num_experts}, top_k={top_k}) ===")
    random_top1 = 1.0 / num_experts
    random_topk = 1.0 - (1.0 - top_k / num_experts)
    diag = {}
    for b in BUCKETS:
        t1 = bucket_top1.get(b, [])
        tk = bucket_topk.get(b, [])
        acc1 = sum(t1) / len(t1) if t1 else float("nan")
        acck = sum(tk) / len(tk) if tk else float("nan")
        print(f"  {b:8s} top1_acc={acc1:.3f}  topk_hit_rate={acck:.3f}  (n={len(t1)})")
        diag[b] = {"top1_acc": acc1, "topk_hit_rate": acck, "n": len(t1)}
    print(f"  [reference] random top1 baseline ~= {random_top1:.3f}, random topk baseline ~= {random_topk:.3f}")
    print("=" * 55)
    return diag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["single_lora", "moe_sparse", "moe_dense", "moe_dense_entropy",
                             "moe_dense_multitarget", "moe_dense_nowarmup",
                             "moe_dense_lightwarmup", "moe_dense_nowarmup_multitarget",
                             "moe_dense_nowarmup_multitarget_5ep", "moe_dense_lightwarmup2_multitarget",
                             "moe_sparse_amatch"])
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_train_per_bucket", type=int, default=None)
    ap.add_argument("--max_eval_per_bucket", type=int, default=None)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None, help="override rank (single_lora: total rank; MoE: per-expert rank)")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--route_sup_weight", type=float, default=None)
    ap.add_argument("--aux_loss_weight", type=float, default=None)
    ap.add_argument("--aux_underload_penalty", type=float, default=None)
    ap.add_argument("--dataset", default="cora",
                    choices=["cora", "citeseer", "amazon-photo", "amazon-computers", "wn18rr",
                             "fb15k-237", "pubmed"])
    ap.add_argument("--community_scheme", default="class", choices=["class", "louvain"],
                    help="node-clf datasets only: 'class' (semantic label, default) or "
                         "'louvain' (structural community detection)")
    ap.add_argument("--louvain_min_size", type=int, default=200,
                    help="merge threshold controlling louvain partition granularity "
                         "(smaller -> more, smaller communities/experts)")
    ap.add_argument("--fb15k_min_rel_freq", type=int, default=0,
                    help="fb15k-237 only: merge relations occurring fewer than this many "
                         "times into a single 'other' bucket (0 = no merging, ~192 rels)")
    ap.add_argument("--use_v2_data", action="store_true",
                    help="mix in other_sft_data_v2 supplementary data (amazon-computers/"
                         "amazon-photo/pubmed/fb15k-237 only) alongside the original data")
    ap.add_argument("--no_global_expert", action="store_true",
                    help="disable the always-on global expert to test whether it is "
                         "interfering with local specialization")
    ap.add_argument("--train_communities", default=None,
                    help="comma-separated community ids to keep for training only; when set, "
                         "router and global expert are frozen and only the listed local "
                         "experts receive gradient updates")
    ap.add_argument("--filter_train_communities", default=None,
                    help="comma-separated community ids used only to filter training data; "
                         "does not freeze router/global or mask experts")
    ap.add_argument("--perturb_communities", default=None,
                    help="comma-separated community ids whose labels are flipped in both "
                         "train and eval to simulate a changed community")
    ap.add_argument("--load_trainable_state", default=None,
                    help="path to a saved LoRA/router state to load before training")
    ap.add_argument("--save_trainable_state", default=None,
                    help="path to save LoRA/router state after training")
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training and only run evaluation after optional state loading")
    ap.add_argument("--oracle_route_perturbed_to", type=int, default=None,
                    help="during EM/loss diagnostics, force perturbed eval/update examples "
                         "to route all local weight to this expert id")
    ap.add_argument("--force_train_perturbed_to", type=int, default=None,
                    help="during training, force perturbed examples to route all local "
                         "weight to this expert id")
    ap.add_argument("--expand_train_experts_to_rank", type=int, default=None,
                    help="rank-surgery target rank for train_communities experts; newly "
                         "added B weights are zero-initialized")
    ap.add_argument("--extra_rank_scaling", type=float, default=1.0,
                    help="scaling multiplier applied only to rank-surgery extra dimensions")
    ap.add_argument("--diagnose_train_loss", action="store_true",
                    help="report LM loss on train/update records split by changed/unchanged")
    ap.add_argument("--skip_em_eval", action="store_true",
                    help="skip generation EM eval; useful when only train-loss diagnostics are needed")
    ap.add_argument("--tag", default=None, help="suffix for output_dir, e.g. seed123 or 3B")
    args = ap.parse_args()

    train_communities = _parse_int_list(args.train_communities)
    filter_train_communities = _parse_int_list(args.filter_train_communities)
    perturb_communities = _parse_int_list(args.perturb_communities)

    cfg = dict(BASE_CFG)
    cfg.update(VARIANT_OVERRIDES[args.variant])
    if args.model_path:
        cfg["base_model"] = args.model_path
    else:
        cfg["base_model"] = resolve_model_path(cfg["base_model"])
    for key in ("seed", "max_train_per_bucket", "max_eval_per_bucket", "num_epochs", "batch_size",
                "rank", "lora_alpha", "top_k", "lr", "route_sup_weight", "aux_loss_weight",
                "aux_underload_penalty"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.no_global_expert:
        cfg["use_global_expert"] = False
    if args.dataset != "cora":
        cfg["num_local_experts"] = other_num_communities(
            args.dataset, cfg["other_graph_root"], sft_root=cfg["other_sft_root"],
            community_scheme=args.community_scheme, louvain_min_size=args.louvain_min_size,
            fb15k_min_rel_freq=args.fb15k_min_rel_freq, use_v2=args.use_v2_data,
        )
    out_name = args.variant + (f"_{args.tag}" if args.tag else "")
    cfg["output_dir"] = f"outputs/matched_param/{args.dataset}/{out_name}"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.variant}] Device: {device}  Model: {cfg['base_model']}")
    if args.variant == "single_lora":
        print(f"[{args.variant}] rank-equivalent ACTIVE per forward = TOTAL = {cfg['rank']}")
    else:
        active_eq = cfg["rank"] * (1 + cfg["top_k"]) if cfg["use_global_expert"] else cfg["rank"] * cfg["top_k"]
        total_eq = cfg["rank"] * (cfg["num_local_experts"] + (1 if cfg["use_global_expert"] else 0))
        print(f"[{args.variant}] rank-equivalent ACTIVE per forward = {active_eq}  "
              f"(global={cfg['rank'] if cfg['use_global_expert'] else 0} + top_k={cfg['top_k']}*local={cfg['rank']})  "
              f"| TOTAL stored = {total_eq}")

    is_multitarget = args.variant in ("moe_dense_multitarget", "moe_dense_nowarmup_multitarget",
                                      "moe_dense_nowarmup_multitarget_5ep",
                                      "moe_dense_lightwarmup2_multitarget", "moe_sparse_amatch")
    if args.dataset != "cora":
        all_records = load_other_sft_records(
            args.dataset, cfg["other_sft_root"], cfg["other_graph_root"], seed=cfg["seed"],
            community_scheme=args.community_scheme, louvain_min_size=args.louvain_min_size,
            fb15k_min_rel_freq=args.fb15k_min_rel_freq, use_v2=args.use_v2_data,
        )
    elif is_multitarget:
        all_records = load_records_multi(cfg["rewritten_dir"], cfg["cora_dir"], seed=cfg["seed"])
    else:
        all_records = load_phase1_records(cfg["rewritten_dir"], cfg["cora_dir"], seed=cfg["seed"])
    train_records, eval_records = split_records(
        all_records, cfg["max_train_per_bucket"], cfg["max_eval_per_bucket"], seed=cfg["seed"]
    )

    if perturb_communities is not None:
        train_records = _apply_label_flip_perturbation(train_records, perturb_communities, "train")
        eval_records = _apply_label_flip_perturbation(eval_records, perturb_communities, "eval")

    active_train_filter = train_communities if train_communities is not None else filter_train_communities
    if active_train_filter is not None:
        train_records = [r for r in train_records if r.get("community") in active_train_filter]
        if not train_records:
            raise ValueError(f"No training records left after filtering to communities={active_train_filter}")

    if train_communities is not None:
        cfg["expert_warmup"] = False
        cfg["route_sup_weight"] = 0.0
        cfg["aux_loss_weight"] = 0.0
    print(f"[{args.variant}] Train: {len(train_records)}  Eval: {len(eval_records)}")
    print(f"[{args.variant}] Train bucket dist:", dict(Counter(r["bucket"] for r in train_records)))
    if active_train_filter is not None:
        print(f"[{args.variant}] Filtered training communities: {active_train_filter}")
    if perturb_communities is not None:
        print(f"[{args.variant}] Perturbed communities: {perturb_communities}")

    comm_freq = Counter(r["community"] for r in train_records)
    num_experts = cfg["num_local_experts"]
    raw_counts = torch.tensor([comm_freq.get(i, 1) for i in range(num_experts)], dtype=torch.float32)
    inv_freq = 1.0 / raw_counts
    class_weights_cpu = inv_freq / inv_freq.mean()

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    router = None
    moe_layers = None
    is_dense = args.variant in ("moe_dense", "moe_dense_entropy", "moe_dense_multitarget",
                                "moe_dense_nowarmup", "moe_dense_lightwarmup",
                                "moe_dense_nowarmup_multitarget", "moe_dense_nowarmup_multitarget_5ep",
                                "moe_dense_lightwarmup2_multitarget")
    is_moe = args.variant in ("moe_sparse", "moe_dense", "moe_dense_entropy", "moe_dense_multitarget",
                              "moe_dense_nowarmup", "moe_dense_lightwarmup",
                              "moe_dense_nowarmup_multitarget", "moe_dense_nowarmup_multitarget_5ep",
                              "moe_dense_lightwarmup2_multitarget", "moe_sparse_amatch")

    if args.variant == "single_lora":
        model = inject_single_lora(model, rank=cfg["rank"], lora_alpha=cfg["lora_alpha"])
    else:
        model, router = inject_moe_lora(
            model, rank=cfg["rank"], lora_alpha=cfg["lora_alpha"],
            num_local_experts=num_experts, use_global_expert=cfg["use_global_expert"],
            top_k=cfg["top_k"],
        )
        moe_layers = get_moe_layers(model)
        if is_dense:
            hidden_size = router.gate[0].in_features
            router = DenseSoftRouter(hidden_size, num_experts, top_k=cfg["top_k"])
            GlobalLocalLoraLinear.forward = _dense_forward
        else:
            _orig_sparse_forward = GlobalLocalLoraLinear.forward

            def _sparse_forward_with_cache(self, x, router_weights=None, router_indices=None):
                rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)
                ri = router_indices if router_indices is not None else getattr(self, "_cached_ri", None)
                return _orig_sparse_forward(self, x, rw, ri)

            GlobalLocalLoraLinear.forward = _sparse_forward_with_cache
        router = router.to(device)

    if train_communities is not None and is_moe:
        configure_incremental_update(
            model,
            router,
            train_local_experts=train_communities,
            train_global_expert=False,
            train_router=False,
        )
        print(f"[{args.variant}] Incremental-update mode: router/global frozen; local experts {train_communities} trainable")

    if args.load_trainable_state:
        _load_trainable_state(model, args.load_trainable_state)

    expanded_rank_added = 0
    if args.expand_train_experts_to_rank is not None:
        if train_communities is None:
            raise ValueError("--expand_train_experts_to_rank requires --train_communities")
        expanded_rank_added = expand_local_experts_to_rank(
            model, train_communities, args.expand_train_experts_to_rank,
            extra_scaling=args.extra_rank_scaling,
        )
        print(f"[{args.variant}] Expanded local experts {train_communities} to rank "
              f"{args.expand_train_experts_to_rank} (added rank dims across layers={expanded_rank_added})")

    if cfg["expert_warmup"] and not args.eval_only:
        if is_dense:
            _run_expert_warmup_dense(model, moe_layers, train_records, tokenizer, cfg, device)
        else:
            from phase1_train import _run_expert_warmup
            _run_expert_warmup(model, moe_layers, train_records, tokenizer, cfg, device)

    if is_multitarget:
        train_ds = MultiTargetDataset(train_records, tokenizer, cfg["max_length"], num_experts)
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn_multitarget, num_workers=0)
    else:
        train_ds = Phase1Dataset(train_records, tokenizer, cfg["max_length"])
        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collate_fn, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if router is not None:
        trainable_params += list(router.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[{args.variant}] Trainable params: {n_trainable:,}")

    total_steps = len(train_dl) * cfg["num_epochs"]
    if args.eval_only or cfg["num_epochs"] <= 0:
        print(f"[{args.variant}] Eval-only mode: skipping training")
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
        )

    global_step = 0
    for epoch in range(0 if args.eval_only else cfg["num_epochs"]):
        model.train()
        if router is not None:
            router.train()
        epoch_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            community_labels = batch["community"].to(device)
            soft_target = batch["soft_target"].to(device) if is_multitarget else None

            if is_moe:
                with torch.no_grad():
                    embed = model.model.embed_tokens(input_ids)
                    mask_f = attn_mask.unsqueeze(-1).float()
                    query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                weights, topk_idx, logits = router(query_repr.to(torch.float32))
                if args.force_train_perturbed_to is not None:
                    perturbed_mask = batch.get("perturbed")
                    if perturbed_mask is not None:
                        weights, topk_idx = _force_route_by_mask(
                            weights, topk_idx, perturbed_mask.to(weights.device), args.force_train_perturbed_to
                        )
                set_router_decision(moe_layers, weights, topk_idx)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            lm_loss = outputs.loss

            if is_moe:
                frac = global_step / max(1, total_steps)
                lam_route = cfg["route_sup_weight"] * (1.0 - frac) if cfg["route_sup_anneal"] else cfg["route_sup_weight"]
                if is_multitarget:
                    cw = class_weights_cpu.to(logits.device)
                    log_p = F.log_softmax(logits, dim=-1)
                    route_sup_loss = -(soft_target * cw.unsqueeze(0) * log_p).sum(-1).mean()
                elif cfg.get("class_weighted_route_sup", False):
                    cw = class_weights_cpu.to(logits.device)
                    route_sup_loss = F.cross_entropy(logits, community_labels, weight=cw)
                else:
                    route_sup_loss = F.cross_entropy(logits, community_labels)

                if is_dense:
                    ent_loss = entropy_aux_loss(weights)
                    loss = lm_loss + lam_route * route_sup_loss + cfg["entropy_aux_weight"] * ent_loss
                elif cfg.get("aux_underload_penalty", 1.0) > 1.0:
                    probs = F.softmax(logits, dim=-1)
                    P = probs.mean(0)
                    one_hot = F.one_hot(topk_idx, num_classes=num_experts).float()
                    f = one_hot.sum(1).mean(0)
                    target_load = cfg["top_k"] / num_experts
                    pen = torch.where(f < target_load, torch.full_like(f, cfg["aux_underload_penalty"]), torch.ones_like(f))
                    aux_loss = num_experts * (pen * f * P).sum()
                    loss = lm_loss + cfg["aux_loss_weight"] * aux_loss + lam_route * route_sup_loss
                else:
                    loss = compute_total_loss(lm_loss, logits, num_experts, cfg["top_k"], cfg["aux_loss_weight"]) + lam_route * route_sup_loss
            else:
                loss = lm_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            if is_moe:
                clear_router_decision(moe_layers)

            epoch_loss += loss.item()
            global_step += 1
            if global_step % 20 == 0:
                print(f"  step {global_step:4d} | loss={loss.item():.4f} | lm={lm_loss.item():.4f}", flush=True)

        print(f"[{args.variant}] Epoch {epoch+1}/{cfg['num_epochs']} avg_loss={epoch_loss/len(train_dl):.4f}")

    if args.save_trainable_state:
        save_path = Path(args.save_trainable_state)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trainable_state": _collect_trainable_state(model, router)}, save_path)
        print(f"[state] Saved trainable params -> {save_path}")

    loss_diag = None
    if args.diagnose_train_loss:
        loss_diag = train_loss_diagnostic(
            model, train_records, tokenizer, device, cfg["eval_batch_size"], cfg["max_length"],
            router=router, moe_layers=moe_layers,
            oracle_route_perturbed_to=args.oracle_route_perturbed_to,
        )
        print("\n[diagnostic] Train/update LM loss")
        for group, item in loss_diag.items():
            print(f"  {group:9s} loss={item['loss']:.4f}  (n={item['n']})")

    model.eval()
    if router is not None:
        router.eval()
    if args.skip_em_eval:
        print(f"\n[{args.variant}] Skipping EM eval")
        em_stats = {}
        router_diag = None
    else:
        print(f"\n[{args.variant}] Running EM eval ...")
        em_stats = em_eval(
            model, eval_records, tokenizer, device, cfg["eval_batch_size"],
            router=router, moe_layers=moe_layers, is_dense=is_dense,
            oracle_route_perturbed_to=args.oracle_route_perturbed_to,
            target_communities=perturb_communities,
        )

        router_diag = None
        if is_moe:
            router_diag = router_diagnostic(
                model, router, eval_records, tokenizer, device, cfg["eval_batch_size"],
                num_experts, cfg["top_k"],
            )

    print(f"\n{'='*55}\n  {args.variant} — EM Accuracy\n{'='*55}")
    result = {"variant": args.variant, "trainable_params": n_trainable, "buckets": {}}
    if args.oracle_route_perturbed_to is not None:
        result["oracle_route_perturbed_to"] = args.oracle_route_perturbed_to
    if args.force_train_perturbed_to is not None:
        result["force_train_perturbed_to"] = args.force_train_perturbed_to
    if args.expand_train_experts_to_rank is not None:
        result["rank_expansion"] = {
            "train_communities": train_communities,
            "target_rank": args.expand_train_experts_to_rank,
            "extra_rank_scaling": args.extra_rank_scaling,
            "added_rank_dims_across_layers": expanded_rank_added,
        }
    if loss_diag is not None:
        result["train_loss_diagnostic"] = loss_diag
    if train_communities is not None:
        result["incremental_update"] = {
            "train_communities": train_communities,
            "freeze_router": True,
            "freeze_global_expert": True,
        }
    for b in BUCKETS:
        s = em_stats.get(b, {})
        acc = s.get("acc", float("nan"))
        n = s.get("n_valid", 0)
        pct = s.get("pct_valid", 0)
        print(f"  {b:8s} EM={acc:.3f}  (n_valid={n}, {pct:.0%})")
        result["buckets"][b] = s
    group_stats = em_stats.get("_groups")
    if group_stats is not None:
        result["groups"] = group_stats
        print("  -- groups --")
        for group, s in group_stats.items():
            print(f"  {group:8s} EM={s.get('acc', float('nan')):.3f}  "
                  f"(n_valid={s.get('n_valid', 0)}, {s.get('pct_valid', 0):.0%})")
    print(f"{'='*55}")
    if router_diag is not None:
        result["router_diagnostic"] = router_diag

    with open(os.path.join(cfg["output_dir"], "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.variant}] Results saved -> {cfg['output_dir']}/results.json")

if __name__ == "__main__":
    main()
