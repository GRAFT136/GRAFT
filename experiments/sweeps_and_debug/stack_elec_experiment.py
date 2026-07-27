
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from phase1_single_lora import inject_single_lora
from phase1_train import clear_router_decision, set_router_decision
from src.model.injection import get_moe_layers, inject_moe_lora
from src.model.losses import compute_total_loss
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.token_injection import inject_token_moe_lora
from src.model.token_moe_lora import (
    clear_token_router_caches,
    token_load_balancing_loss,
    token_route_supervision_loss,
)

MODEL_7B = (
    "/home/USER/.cache/huggingface/hub/"
    "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

SFT_FILES = [
    ("01_existence_qa.jsonl", "existence"),
    ("02_counting_qa.jsonl", "counting"),
    ("03_traversal_qa.jsonl", "traversal"),
    ("04_substructure_qa.jsonl", "substructure"),
    ("05_multihop_qa.jsonl", "multihop"),
    ("06_node_info_qa.jsonl", "node_info"),
    ("07_edge_info_qa.jsonl", "edge_info"),
    ("08_node_basic_qa.jsonl", "node_basic"),
]

def _read_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def _reservoir_sample_jsonl(path: Path, limit: int, rng: random.Random) -> List[Dict]:
    if limit <= 0:
        return []
    sample: List[Dict] = []
    for seen, item in enumerate(_read_jsonl(path), start=1):
        if len(sample) < limit:
            sample.append(item)
            continue
        j = rng.randrange(seen)
        if j < limit:
            sample[j] = item
    rng.shuffle(sample)
    return sample

def _stable_hash_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)

def load_stack_pretrain(stack_root: Path, limit: int, num_experts: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    path = stack_root / "pretrain_data" / "pretrain_data.jsonl"
    records = []
    for item in _reservoir_sample_jsonl(path, limit, rng):
        source = int(item.get("source", item.get("edge_id", 0)))
        text = (
            "Stack Exchange Electrical Engineering graph edge summary:\n"
            f"{item.get('sentence1', '')}\n\n"
            "Related paraphrase:\n"
            f"{item.get('sentence2', '')}"
        )
        records.append({
            "text": text,
            "community": source % num_experts,
            "bucket": "pretrain",
        })
    return records

def load_stack_sft(
    stack_root: Path,
    max_train_per_file: int,
    max_eval_per_file: int,
    seed: int,
    include_augmented: bool = False,
    sft_expert_shards: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    train_records: List[Dict] = []
    eval_records: List[Dict] = []
    roots = [(stack_root / "sft_data", "")]
    if include_augmented:
        roots.append((stack_root / "sft_data" / "augmented", "_aug"))

    for root, suffix in roots:
        for community, (fname, bucket) in enumerate(SFT_FILES):
            path = root / (fname.replace(".jsonl", f"{suffix}.jsonl") if suffix else fname)
            if not path.exists():
                continue
            rows = [item for item in _read_jsonl(path) if "query" in item and "answer" in item]
            rng.shuffle(rows)
            if max_eval_per_file < 0:
                n_eval = len(rows)
            elif max_eval_per_file == 0:
                n_eval = max(1, len(rows) // 10)
            else:
                n_eval = min(max_eval_per_file, max(1, len(rows) // 10))
            eval_part = rows[:n_eval]
            train_part = rows[n_eval:n_eval + max_train_per_file]
            for split_rows, target in ((train_part, train_records), (eval_part, eval_records)):
                for item in split_rows:
                    query = str(item["query"])
                    if sft_expert_shards > 1:
                        shard = _stable_hash_int(query) % sft_expert_shards
                        expert_id = community * sft_expert_shards + shard
                    else:
                        expert_id = community
                    rec = {
                        "query": query,
                        "answer": str(item["answer"]),
                        "community": expert_id,
                        "bucket": bucket,
                        "source_file": path.name,
                    }
                    target.append(rec)

    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    return train_records, eval_records

class PretrainTextDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = rec["text"] + self.tokenizer.eos_token
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "community": int(rec.get("community", 0)),
        }

class StackSFTDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._assistant_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = (
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['answer']}<|im_end|>"
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        ids = input_ids.tolist()
        n = len(self._assistant_ids)
        for pos in range(len(ids) - n):
            if ids[pos:pos + n] == self._assistant_ids:
                labels[:pos + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": int(rec.get("community", 0)),
        }

def collate_batch(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    communities = torch.zeros(batch_size, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1
        communities[row] = item["community"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "community": communities,
    }

def _patch_old_moe_forward() -> None:
    if getattr(GlobalLocalLoraLinear, "_stack_elec_cache_patch", False):
        return
    original_forward = GlobalLocalLoraLinear.forward

    def _forward_with_cache(self, x, router_weights=None, router_indices=None):
        rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)
        ri = router_indices if router_indices is not None else getattr(self, "_cached_ri", None)
        return original_forward(self, x, rw, ri)

    GlobalLocalLoraLinear.forward = _forward_with_cache
    GlobalLocalLoraLinear._stack_elec_cache_patch = True

def _unique_trainable_params(model, router=None) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen = set()
    for param in model.parameters():
        if param.requires_grad and id(param) not in seen:
            params.append(param)
            seen.add(id(param))
    if router is not None:
        for param in router.parameters():
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params

def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    router = None
    old_moe_layers = []
    token_layers = []

    if args.arch == "single_lora":
        model = inject_single_lora(model, rank=args.single_rank, lora_alpha=args.single_alpha)
    elif args.arch == "old_moe":
        model, router = inject_moe_lora(
            model,
            rank=args.moe_rank,
            lora_alpha=args.moe_alpha,
            num_local_experts=args.num_experts,
            use_global_expert=True,
            top_k=args.top_k,
        )
        router = router.to("cuda" if torch.cuda.is_available() else "cpu")
        old_moe_layers = get_moe_layers(model)
        _patch_old_moe_forward()
    elif args.arch == "token_moe":
        model, token_layers = inject_token_moe_lora(
            model,
            rank=args.moe_rank,
            lora_alpha=args.moe_alpha,
            num_experts=args.num_experts,
            top_k=args.top_k,
            use_global_expert=True,
            router_temperature=args.router_temperature,
        )
    else:
        raise ValueError(f"Unknown arch: {args.arch}")

    return tokenizer, model, router, old_moe_layers, token_layers

def load_trainable_state(path: Path, model, router=None) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("trainable_state", ckpt)
    named = dict(model.named_parameters())
    router_named = dict(router.named_parameters()) if router is not None else {}
    loaded = 0
    skipped = 0

    for name, tensor in state.items():
        target_name = name
        target = named.get(target_name)
        if target is None and name.startswith("router."):
            target_name = name[len("router."):]
            target = router_named.get(target_name)
        if target is None or tuple(target.shape) != tuple(tensor.shape):
            skipped += 1
            continue
        with torch.no_grad():
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded += 1

    print(f"[state] loaded={loaded} skipped={skipped} from {path}")
    return ckpt

def train_stage(
    stage_name: str,
    args,
    model,
    router,
    old_moe_layers,
    token_layers,
    dataloader: DataLoader,
    max_steps: int,
    num_epochs: int,
    device: str,
) -> Dict:
    if len(dataloader) == 0 or max_steps == 0 or num_epochs == 0:
        print(f"[{args.arch}:{stage_name}] skipped")
        return {"steps": 0, "avg_loss": float("nan")}

    trainable = _unique_trainable_params(model, router)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    planned_steps = min(max_steps, len(dataloader) * num_epochs) if max_steps > 0 else len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, planned_steps // 10),
        num_training_steps=max(1, planned_steps),
    )

    print(
        f"[{args.arch}:{stage_name}] trainable={sum(p.numel() for p in trainable):,} "
        f"batches={len(dataloader)} planned_steps={planned_steps}"
    )
    model.train()
    if router is not None:
        router.train()

    total_loss = 0.0
    total_lm = 0.0
    step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            community = batch["community"].to(device)

            logits = None
            if args.arch == "old_moe":
                with torch.no_grad():
                    embed = model.model.embed_tokens(input_ids)
                    mask_f = attention_mask.unsqueeze(-1).float()
                    query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                weights, indices, logits = router(query_repr.to(torch.float32))
                set_router_decision(old_moe_layers, weights, indices)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs.loss
            loss = lm_loss

            if args.arch == "old_moe" and logits is not None:
                if args.route_sup_weight:
                    loss = loss + args.route_sup_weight * F.cross_entropy(logits, community)
                if args.aux_loss_weight:
                    loss = compute_total_loss(loss, logits, args.num_experts, args.top_k, args.aux_loss_weight)
            elif args.arch == "token_moe":
                if args.route_sup_weight:
                    loss = loss + args.route_sup_weight * token_route_supervision_loss(token_layers, community, attention_mask)
                if args.aux_loss_weight:
                    loss = loss + args.aux_loss_weight * token_load_balancing_loss(token_layers, attention_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()

            if args.arch == "old_moe":
                clear_router_decision(old_moe_layers)
            elif args.arch == "token_moe":
                clear_token_router_caches(token_layers)

            step += 1
            total_loss += float(loss.detach().cpu())
            total_lm += float(lm_loss.detach().cpu())
            if step % args.log_every == 0 or step == 1:
                print(
                    f"[{args.arch}:{stage_name}] step={step:4d}/{planned_steps} "
                    f"loss={loss.item():.4f} lm={lm_loss.item():.4f}",
                    flush=True,
                )
            if step >= planned_steps:
                avg = total_loss / max(1, step)
                avg_lm = total_lm / max(1, step)
                print(f"[{args.arch}:{stage_name}] done avg_loss={avg:.4f} avg_lm={avg_lm:.4f}")
                return {"steps": step, "avg_loss": avg, "avg_lm_loss": avg_lm}

    avg = total_loss / max(1, step)
    avg_lm = total_lm / max(1, step)
    print(f"[{args.arch}:{stage_name}] done avg_loss={avg:.4f} avg_lm={avg_lm:.4f}")
    return {"steps": step, "avg_loss": avg, "avg_lm_loss": avg_lm}

@torch.no_grad()
def eval_sft_loss(args, model, router, old_moe_layers, token_layers, dataloader, records, device: str) -> Dict:
    model.eval()
    if router is not None:
        router.eval()
    losses = defaultdict(list)
    offset = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        batch_records = records[offset:offset + input_ids.shape[0]]
        offset += input_ids.shape[0]

        if args.arch == "old_moe":
            embed = model.model.embed_tokens(input_ids)
            mask_f = attention_mask.unsqueeze(-1).float()
            query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            weights, indices, _ = router(query_repr.to(torch.float32))
            set_router_decision(old_moe_layers, weights, indices)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if args.arch == "old_moe":
            clear_router_decision(old_moe_layers)
        elif args.arch == "token_moe":
            clear_token_router_caches(token_layers)

        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = (shift_labels != -100).float()
        per_sample = (loss_flat.view(input_ids.shape[0], -1) * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        for idx, rec in enumerate(batch_records):
            losses[rec["bucket"]].append(float(per_sample[idx].cpu()))

    by_bucket = {bucket: {"loss": sum(vals) / len(vals), "ppl": math.exp(min(sum(vals) / len(vals), 20)), "n": len(vals)}
                 for bucket, vals in losses.items() if vals}
    all_vals = [value for vals in losses.values() for value in vals]
    return {
        "overall_loss": sum(all_vals) / len(all_vals) if all_vals else float("nan"),
        "overall_ppl": math.exp(min(sum(all_vals) / len(all_vals), 20)) if all_vals else float("nan"),
        "by_bucket": by_bucket,
    }

def _binary_label(text: str) -> str:
    value = text.lower().strip()
    if value.startswith(("yes", "indeed", "correct", "true")):
        return "pos"
    if value.startswith(("no", "negative", "false")) or " no evidence" in value or "not " in value:
        return "neg"
    return "unk"

def _first_int(text: str) -> Optional[int]:
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None

def _normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

def answer_match(query: str, generated: str, answer: str) -> Optional[bool]:
    q = query.lower()
    if q.startswith("did ") or q.startswith("has ") or q.startswith("was ") or q.startswith("is "):
        gt = _binary_label(answer)
        pred = _binary_label(generated)
        if gt != "unk" and pred != "unk":
            return gt == pred
    if "how many" in q or "count" in q or "degree" in q:
        gt_n = _first_int(answer)
        pred_n = _first_int(generated)
        if gt_n is not None and pred_n is not None:
            return gt_n == pred_n
    gt_norm = _normalized(answer)
    gen_norm = _normalized(generated)
    if not gt_norm or not gen_norm:
        return None
    return gt_norm[:80] in gen_norm or gen_norm[:80] in gt_norm

def strict_answer_match(query: str, generated: str, answer: str) -> Optional[bool]:
    q = query.lower().strip()
    if q.startswith(("did ", "has ", "was ", "is ")):
        gt = _binary_label(answer)
        pred = _binary_label(generated)
        if gt != "unk" and pred != "unk":
            return gt == pred
        return False
    if "how many" in q or "count" in q or "degree" in q:
        gt_n = _first_int(answer)
        pred_n = _first_int(generated)
        if gt_n is not None and pred_n is not None:
            return gt_n == pred_n
        return False
    return _normalized(generated) == _normalized(answer)

@torch.no_grad()
def generation_eval(args, model, router, old_moe_layers, token_layers, records, tokenizer, device: str) -> Dict:
    if args.eval_generation_samples <= 0:
        if args.eval_generation_samples == 0:
            return {}
        sample = list(records)
    else:
        rng = random.Random(args.seed + 17)
        sample = list(records)
        rng.shuffle(sample)
        sample = sample[:args.eval_generation_samples]

    eos_list = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_list[0] if eos_list else tokenizer.eos_token_id
    results = defaultdict(list)
    detail_handle = None
    if args.save_generations:
        detail_path = Path(args.save_generations)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_handle = open(detail_path, "w", encoding="utf-8")
    model.eval()
    if router is not None:
        router.eval()

    try:
        for start in range(0, len(sample), args.eval_batch_size):
            batch = sample[start:start + args.eval_batch_size]
            if start == 0 or (start // args.eval_batch_size) % max(1, args.eval_log_every) == 0:
                print(f"[eval:{args.arch}] generated {start}/{len(sample)}", flush=True)
            prompts = [f"<|im_start|>user\n{rec['query']}<|im_end|>\n<|im_start|>assistant\n" for rec in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(device)
            prompt_len = enc["input_ids"].shape[1]

            if args.arch == "old_moe":
                embed = model.model.embed_tokens(enc["input_ids"])
                mask_f = enc["attention_mask"].unsqueeze(-1).float()
                query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                weights, indices, _ = router(query_repr.to(torch.float32))
                set_router_decision(old_moe_layers, weights, indices)

            gen_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_id,
            )
            if args.arch == "old_moe":
                clear_router_decision(old_moe_layers)
            elif args.arch == "token_moe":
                clear_token_router_caches(token_layers)

            for idx, rec in enumerate(batch):
                generated = tokenizer.decode(gen_ids[idx][prompt_len:], skip_special_tokens=True).strip()
                if args.strict_em:
                    match = strict_answer_match(rec["query"], generated, rec["answer"])
                else:
                    match = answer_match(rec["query"], generated, rec["answer"])
                results[rec["bucket"]].append(match)
                if detail_handle is not None:
                    detail = {
                        "idx": start + idx,
                        "arch": args.arch,
                        "bucket": rec.get("bucket"),
                        "source_file": rec.get("source_file"),
                        "query": rec["query"],
                        "answer": rec["answer"],
                        "generated": generated,
                        "local_match": match,
                    }
                    detail_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    finally:
        if detail_handle is not None:
            detail_handle.close()

    by_bucket = {}
    for bucket, vals in results.items():
        valid = [value for value in vals if value is not None]
        by_bucket[bucket] = {
            "match": sum(valid) / len(valid) if valid else float("nan"),
            "n_valid": len(valid),
            "n_total": len(vals),
            "pct_valid": len(valid) / len(vals) if vals else 0.0,
        }
    all_valid = [value for vals in results.values() for value in vals if value is not None]
    return {
        "overall_match": sum(all_valid) / len(all_valid) if all_valid else float("nan"),
        "n_valid": len(all_valid),
        "n_total": sum(len(vals) for vals in results.values()),
        "strict_em": args.strict_em,
        "full_eval": args.eval_generation_samples < 0,
        "by_bucket": by_bucket,
    }

def save_trainable_state(path: Path, model, router, args, stage_metrics, eval_metrics) -> None:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad or "lora_" in name or "router" in name or "token_router" in name:
            state[name] = param.detach().cpu()
    if router is not None:
        for name, param in router.named_parameters():
            state[f"router.{name}"] = param.detach().cpu()
    torch.save({
        "arch": args.arch,
        "args": vars(args),
        "stage_metrics": stage_metrics,
        "eval_metrics": eval_metrics,
        "trainable_state": state,
    }, path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=["single_lora", "old_moe", "token_moe"])
    parser.add_argument("--model_path", default=MODEL_7B)
    parser.add_argument("--stack_root", default="../Stack_elec_dataset")
    parser.add_argument("--output_root", default="outputs/stack_elec_7b")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--sft_expert_shards", type=int, default=1)
    parser.add_argument("--single_rank", type=int, default=6)
    parser.add_argument("--single_alpha", type=float, default=12.0)
    parser.add_argument("--moe_rank", type=int, default=2)
    parser.add_argument("--moe_alpha", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_pretrain_samples", type=int, default=2000)
    parser.add_argument("--max_sft_train_per_file", type=int, default=200)
    parser.add_argument("--max_sft_eval_per_file", type=int, default=60)
    parser.add_argument("--max_pretrain_steps", type=int, default=400)
    parser.add_argument("--max_sft_steps", type=int, default=800)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--route_sup_weight", type=float, default=0.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--include_augmented", action="store_true")
    parser.add_argument("--eval_generation_samples", type=int, default=120)
    parser.add_argument("--strict_em", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load_adapter_state", default=None)
    parser.add_argument("--eval_log_every", type=int, default=50)
    parser.add_argument("--save_generations", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_root = Path(args.stack_root)
    if not stack_root.exists():
        raise FileNotFoundError(stack_root)
    required_experts = len(SFT_FILES) * max(1, args.sft_expert_shards)
    if args.num_experts < required_experts:
        raise ValueError(
            f"num_experts={args.num_experts} is too small for "
            f"sft_expert_shards={args.sft_expert_shards}; need >= {required_experts}"
        )

    run_name = args.arch + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stack] arch={args.arch} device={device} model={args.model_path}")
    print(f"[stack] output={out_dir}")
    active_rank = args.single_rank if args.arch == "single_lora" else args.moe_rank * (1 + args.top_k)
    print(f"[stack] active_rank_equiv={active_rank}")

    pretrain_records: List[Dict] = []
    if args.arch != "single_lora":
        pretrain_records = load_stack_pretrain(stack_root, args.max_pretrain_samples, args.num_experts, args.seed)
    sft_train, sft_eval = load_stack_sft(
        stack_root,
        args.max_sft_train_per_file,
        args.max_sft_eval_per_file,
        args.seed,
        include_augmented=args.include_augmented,
        sft_expert_shards=args.sft_expert_shards,
    )
    print(f"[stack] pretrain={len(pretrain_records)} sft_train={len(sft_train)} sft_eval={len(sft_eval)}")
    print("[stack] sft_train buckets:", dict(Counter(rec["bucket"] for rec in sft_train)))

    tokenizer, model, router, old_moe_layers, token_layers = build_model_and_tokenizer(args)
    loaded_ckpt = None
    if args.load_adapter_state:
        loaded_ckpt = load_trainable_state(Path(args.load_adapter_state), model, router)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    stage_metrics = loaded_ckpt.get("stage_metrics", {}) if loaded_ckpt is not None else {}
    if args.arch != "single_lora" and not args.eval_only:
        pretrain_ds = PretrainTextDataset(pretrain_records, tokenizer, args.max_length)
        pretrain_dl = DataLoader(
            pretrain_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, pad_token_id),
            num_workers=0,
        )
        stage_metrics["pretrain"] = train_stage(
            "pretrain", args, model, router, old_moe_layers, token_layers,
            pretrain_dl, args.max_pretrain_steps, args.pretrain_epochs, device,
        )

    if not args.eval_only:
        sft_ds = StackSFTDataset(sft_train, tokenizer, args.max_length)
        sft_dl = DataLoader(
            sft_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_batch(batch, pad_token_id),
            num_workers=0,
        )
        stage_metrics["sft"] = train_stage(
            "sft", args, model, router, old_moe_layers, token_layers,
            sft_dl, args.max_sft_steps, args.sft_epochs, device,
        )
    else:
        print(f"[stack] eval_only=True loaded_adapter={args.load_adapter_state}")

    eval_ds = StackSFTDataset(sft_eval, tokenizer, args.max_length)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
        num_workers=0,
    )
    eval_metrics = {
        "loss": eval_sft_loss(args, model, router, old_moe_layers, token_layers, eval_dl, sft_eval, device),
        "generation": generation_eval(args, model, router, old_moe_layers, token_layers, sft_eval, tokenizer, device),
    }

    result = {
        "arch": args.arch,
        "model_path": args.model_path,
        "active_rank_equiv": active_rank,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "data": {
            "pretrain_records": len(pretrain_records),
            "sft_train_records": len(sft_train),
            "sft_eval_records": len(sft_eval),
            "include_augmented": args.include_augmented,
            "sft_expert_shards": args.sft_expert_shards,
            "strict_em": args.strict_em,
            "eval_generation_samples": args.eval_generation_samples,
        },
        "stage_metrics": stage_metrics,
        "eval_metrics": eval_metrics,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    ckpt_path = out_dir / "adapter_state.pt"
    if not args.eval_only:
        save_trainable_state(ckpt_path, model, router, args, stage_metrics, eval_metrics)
    print(f"[stack] results -> {results_path}")
    if not args.eval_only:
        print(f"[stack] adapter -> {ckpt_path}")
    print(f"[stack] eval loss={eval_metrics['loss']['overall_loss']:.4f} ppl={eval_metrics['loss']['overall_ppl']:.2f}")
    gen = eval_metrics.get("generation", {})
    if gen:
        print(f"[stack] generation_match={gen.get('overall_match', float('nan')):.3f} n_valid={gen.get('n_valid', 0)}")

if __name__ == "__main__":
    main()