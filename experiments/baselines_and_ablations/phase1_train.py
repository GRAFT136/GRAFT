
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.router import SharedGlobalRouter
from src.model.injection import inject_moe_lora, get_moe_layers
from src.model.losses import compute_total_loss
from src.eval.monitor import RouterMonitor

CFG = {
    "base_model": (
        "/home/USER/.cache/huggingface/hub/"
        "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "cora_dir": "../Cora/cora_dataset",
    "rewritten_dir": "../Cora/sft_data/rewritten",
    "output_dir": "outputs/phase1",
    "rank": 16,
    "lora_alpha": 32.0,
    "num_local_experts": 7,
    "use_global_expert": True,
    "top_k": 2,
    "num_epochs": 2,
    "batch_size": 2,
    "lr": 5e-4,
    "max_length": 256,
    "max_train_per_bucket": 800,
    "max_eval_per_bucket": 150,
    "aux_loss_weight": 0.01,
    "route_sup_weight": 0.5,
    "route_sup_anneal": True,
    "class_weighted_route_sup": True,
    "aux_underload_penalty": 3.0,
    "expert_warmup": True,
    "warmup_steps_per_expert": 50,
    "grad_clip": 1.0,
    "eval_batch_size": 8,
    "seed": 42,
}

CLASS_NAMES = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory",
]

BUCKETS = ["intra", "cross", "global"]

def build_title2class(cora_dir: str) -> dict:
    mapping = {}
    with open(os.path.join(cora_dir, "all.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = row["T"].strip()
            class_name = row["class"].strip()
            if class_name in CLASS_NAMES:
                mapping[title] = CLASS_NAMES.index(class_name)
    return mapping

_TITLE_RE = re.compile(r"<([^>]+)>")

def _extract_titles(text: str):
    return _TITLE_RE.findall(text)

def load_phase1_records(rewritten_dir: str, cora_dir: str, seed: int = 42):
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

            if default_bucket is None:
                c1 = title2class.get(titles[1]) if len(titles) >= 2 else None
                if c1 is None:
                    bucket = "intra"
                else:
                    bucket = "intra" if c0 == c1 else "cross"
            else:
                bucket = default_bucket

            records.append({
                "query": item["query"],
                "answer": item["answer"],
                "community": c0,
                "bucket": bucket,
            })

    return records

def split_records(records, max_train_per_bucket, max_eval_per_bucket, seed=42):
    rng = random.Random(seed)
    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["bucket"]].append(r)

    train, eval_ = [], []
    for bucket, items in by_bucket.items():
        rng.shuffle(items)
        n_eval = min(max_eval_per_bucket, max(1, len(items) // 7))
        n_train = min(max_train_per_bucket, len(items) - n_eval)
        train.extend(items[n_eval: n_eval + n_train])
        eval_.extend(items[:n_eval])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_

class Phase1Dataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
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
            "community": rec["community"],
            "bucket": BUCKETS.index(rec["bucket"]) if rec["bucket"] in BUCKETS else 0,
        }

def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros(B, max_len, dtype=torch.long)
    comms = torch.zeros(B, dtype=torch.long)
    buckets = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attn_mask[i, :L] = 1
        comms[i] = b["community"]
        buckets[i] = b["bucket"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
        "community": comms,
        "bucket": buckets,
    }

def set_router_decision(moe_layers, rw, ri):
    for layer in moe_layers:
        layer._cached_rw = rw
        layer._cached_ri = ri

def clear_router_decision(moe_layers):
    for layer in moe_layers:
        layer._cached_rw = None
        layer._cached_ri = None

class _RecordDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
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
            "bucket": 0,
        }

def _run_expert_warmup(model, moe_layers, all_records, tokenizer, cfg, device):
    num_experts = cfg["num_local_experts"]
    warmup_steps = cfg.get("warmup_steps_per_expert", 50)
    batch_size = cfg["batch_size"]
    max_length = cfg["max_length"]
    lr = cfg["lr"]
    grad_clip = cfg["grad_clip"]

    print(f"\n[warmup] Expert warmup: {num_experts} experts × {warmup_steps} steps each")
    model.train()

    for eid in range(num_experts):
        comm_records = [r for r in all_records if r.get("community", -1) == eid]
        if not comm_records:
            print(f"[warmup] Expert {eid}: no samples, skipping")
            continue

        print(f"[warmup] Expert {eid} ({CLASS_NAMES[eid]}): {len(comm_records)} samples", flush=True)

        ds = _RecordDataset(comm_records, tokenizer, max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
        dl_iter = iter(dl)

        expert_params = []
        for layer in moe_layers:
            expert_params.append(layer.lora_A_local)
            expert_params.append(layer.lora_B_local)
        opt = torch.optim.AdamW(expert_params, lr=lr, weight_decay=0.01)

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

            rw = torch.ones(B, 1, device=device)
            ri = torch.full((B, 1), eid, dtype=torch.long, device=device)
            set_router_decision(moe_layers, rw, ri)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs.loss

            opt.zero_grad()
            loss.backward()

            for layer in moe_layers:
                for param in (layer.lora_A_local, layer.lora_B_local):
                    if param.grad is not None:
                        mask = torch.zeros_like(param.grad)
                        mask[eid] = 1.0
                        param.grad.mul_(mask)

            torch.nn.utils.clip_grad_norm_(expert_params, grad_clip)
            opt.step()
            clear_router_decision(moe_layers)

        print(f"[warmup] Expert {eid} done (last loss={loss.item():.4f})", flush=True)

    print("[warmup] All experts warmed up.\n", flush=True)

@torch.no_grad()
def _eval_bucket_losses(model, router, moe_layers, eval_records, tokenizer, device, batch_size):
    ds = Phase1Dataset(eval_records, tokenizer, CFG["max_length"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=0)

    bucket_losses = defaultdict(list)
    all_hits = []
    all_loads = torch.zeros(CFG["num_local_experts"])

    model.eval()
    router.eval()

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        community_labels = batch["community"].to(device)
        bucket_ids = batch["bucket"]

        embed = model.model.embed_tokens(input_ids)
        mask_f = attn_mask.unsqueeze(-1).float()
        query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        rw, ri, logits = router(query_repr.to(torch.float32))
        set_router_decision(moe_layers, rw, ri)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        shift_logits = outputs.logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_per = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = (shift_labels != -100).float()
        loss_per_sample = (loss_per.view(input_ids.shape[0], -1) * valid).sum(-1) / valid.sum(-1).clamp(min=1)

        clear_router_decision(moe_layers)

        for i in range(input_ids.shape[0]):
            bname = BUCKETS[bucket_ids[i].item()]
            bucket_losses[bname].append(loss_per_sample[i].item())

        for j in range(input_ids.shape[0]):
            gt_c = community_labels[j].item()
            selected = ri[j].tolist()
            all_hits.append(int(gt_c in selected))
            for eid in selected:
                if 0 <= eid < CFG["num_local_experts"]:
                    all_loads[eid] += 1

    overall_hr = sum(all_hits) / len(all_hits) if all_hits else 0.0
    per_bucket_loss = {b: (sum(v)/len(v) if v else float("nan")) for b, v in bucket_losses.items()}
    per_bucket_ppl = {b: math.exp(min(l, 20)) for b, l in per_bucket_loss.items() if not math.isnan(l)}
    load_dist = (all_loads / all_loads.sum().clamp(min=1)).tolist()

    return {
        "per_bucket_loss": per_bucket_loss,
        "per_bucket_ppl": per_bucket_ppl,
        "overall_hit_rate": overall_hr,
        "random_baseline": CFG["top_k"] / CFG["num_local_experts"],
        "load_distribution": load_dist,
        "dead_experts": int((all_loads == 0).sum().item()),
    }

def ablation_eval(model, router, moe_layers, eval_records, tokenizer, device, batch_size):
    for layer in moe_layers:
        layer.use_global_expert = True
    print("[ablation] Evaluating WITH global expert ...")
    result_with = _eval_bucket_losses(model, router, moe_layers, eval_records, tokenizer, device, batch_size)
    result_with["label"] = "WITH global expert"

    for layer in moe_layers:
        layer.use_global_expert = False
    print("[ablation] Evaluating WITHOUT global expert ...")
    result_without = _eval_bucket_losses(model, router, moe_layers, eval_records, tokenizer, device, batch_size)
    result_without["label"] = "WITHOUT global expert"

    for layer in moe_layers:
        layer.use_global_expert = True

    return result_with, result_without

def write_phase1_report(result_with, result_without, output_path):
    lines = []
    lines.append("# Phase 1 Report — Global-Local LoRA-MoE on Cora\n")
    lines.append("## Three-Bucket Evaluation\n")
    lines.append("| Bucket | PPL (with global) | PPL (without global) | Delta PPL | Direction |\n")
    lines.append("|--------|-------------------|----------------------|-----------|----------|\n")
    for b in BUCKETS:
        p_with = result_with["per_bucket_ppl"].get(b, float("nan"))
        p_without = result_without["per_bucket_ppl"].get(b, float("nan"))
        delta = p_without - p_with
        direction = "↑ worse (expected for cross/global)" if delta > 0 else "↓ better"
        lines.append(f"| {b:6s} | {p_with:17.2f} | {p_without:20.2f} | {delta:+9.2f} | {direction} |\n")
    lines.append("\n## Router Hit Rate (with global expert)\n")
    lines.append(f"- Overall hit rate: {result_with['overall_hit_rate']:.3f}\n")
    lines.append(f"- Random baseline:  {result_with['random_baseline']:.3f}\n")
    lines.append(f"- Dead experts:     {result_with['dead_experts']}\n")
    lines.append("\n## Load Distribution (with global expert)\n")
    for i, (name, load) in enumerate(zip(CLASS_NAMES, result_with["load_distribution"])):
        lines.append(f"- Expert {i} ({name:22s}): {load:.3f}\n")
    lines.append("\n## Conclusion\n")
    cross_delta = result_without["per_bucket_ppl"].get("cross", 0) - result_with["per_bucket_ppl"].get("cross", 0)
    intra_delta = result_without["per_bucket_ppl"].get("intra", 0) - result_with["per_bucket_ppl"].get("intra", 0)
    global_delta = result_without["per_bucket_ppl"].get("global", 0) - result_with["per_bucket_ppl"].get("global", 0)
    if cross_delta > intra_delta and global_delta > intra_delta:
        conclusion = "HYPOTHESIS CONFIRMED: Removing global expert hurts cross-community and global queries more than intra-community queries."
    elif cross_delta > intra_delta:
        conclusion = "PARTIAL: Cross-community queries suffer more than intra without global expert."
    else:
        conclusion = "INCONCLUSIVE: Further training or larger model may be needed."
    lines.append(conclusion + "\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"[phase1] Report saved → {output_path}")

def _init_tb(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None

def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase1] Device: {device}")

    print("[phase1] Loading Cora QA data ...")
    all_records = load_phase1_records(CFG["rewritten_dir"], CFG["cora_dir"], seed=CFG["seed"])
    print(f"[phase1] Total records: {len(all_records)}")

    from collections import Counter
    bucket_counts = Counter(r["bucket"] for r in all_records)
    comm_counts = Counter(r["community"] for r in all_records)
    print("[phase1] Bucket distribution:", dict(bucket_counts))
    print("[phase1] Community distribution:", dict(sorted(comm_counts.items())))

    train_records, eval_records = split_records(
        all_records,
        max_train_per_bucket=CFG["max_train_per_bucket"],
        max_eval_per_bucket=CFG["max_eval_per_bucket"],
        seed=CFG["seed"],
    )
    print(f"[phase1] Train: {len(train_records)}, Eval: {len(eval_records)}")

    num_experts = CFG["num_local_experts"]
    from collections import Counter as _Counter
    comm_freq = _Counter(r["community"] for r in train_records)
    raw_counts = torch.tensor(
        [comm_freq.get(i, 1) for i in range(num_experts)], dtype=torch.float32
    )
    inv_freq = 1.0 / raw_counts
    class_weights_cpu = inv_freq / inv_freq.mean()
    print("[phase1] Fix A — route_sup class weights (inv-freq, mean=1):")
    for i, (w, cnt) in enumerate(zip(class_weights_cpu.tolist(), raw_counts.tolist())):
        print(f"  Expert {i:2d} ({CLASS_NAMES[i]:22s}): count={int(cnt):4d}, weight={w:.3f}")

    print(f"[phase1] Loading model: {CFG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["base_model"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model, router = inject_moe_lora(
        model,
        rank=CFG["rank"],
        lora_alpha=CFG["lora_alpha"],
        num_local_experts=CFG["num_local_experts"],
        use_global_expert=CFG["use_global_expert"],
        top_k=CFG["top_k"],
    )
    router = router.to(device)
    moe_layers = get_moe_layers(model)
    print(f"[phase1] MoE layers: {len(moe_layers)}")

    _orig_forward = GlobalLocalLoraLinear.forward

    def _patched_forward(self, x, router_weights=None, router_indices=None):
        rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)
        ri = router_indices if router_indices is not None else getattr(self, "_cached_ri", None)
        return _orig_forward(self, x, rw, ri)

    GlobalLocalLoraLinear.forward = _patched_forward

    monitor = RouterMonitor(
        num_experts=CFG["num_local_experts"],
        use_tensorboard=True,
        tb_writer=_init_tb(os.path.join(CFG["output_dir"], "tb_logs")),
    )

    if CFG["expert_warmup"]:
        _run_expert_warmup(model, moe_layers, train_records, tokenizer, CFG, device)

    train_ds = Phase1Dataset(train_records, tokenizer, CFG["max_length"])
    train_dl = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[phase1] Train samples: {len(train_ds)}, batches: {len(train_dl)}")

    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(router.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[phase1] Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=CFG["lr"], weight_decay=0.01)
    total_steps = len(train_dl) * CFG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n[phase1] Starting training ({CFG['num_epochs']} epochs) ...")
    global_step = 0

    for epoch in range(CFG["num_epochs"]):
        model.train()
        router.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_dl):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            community_labels = batch["community"].to(device)

            with torch.no_grad():
                embed = model.model.embed_tokens(input_ids)
                mask_f = attn_mask.unsqueeze(-1).float()
                query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

            rw, ri, logits = router(query_repr.to(torch.float32))
            set_router_decision(moe_layers, rw, ri)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            lm_loss = outputs.loss

            frac = global_step / max(1, total_steps)
            lam_route = CFG["route_sup_weight"] * (1.0 - frac) if CFG["route_sup_anneal"] else CFG["route_sup_weight"]

            if CFG.get("class_weighted_route_sup", False):
                cw = class_weights_cpu.to(logits.device)
                route_sup_loss = F.cross_entropy(logits, community_labels, weight=cw)
            else:
                route_sup_loss = F.cross_entropy(logits, community_labels)

            if CFG.get("aux_underload_penalty", 1.0) > 1.0:
                probs = F.softmax(logits, dim=-1)
                P = probs.mean(0)
                one_hot = F.one_hot(ri, num_classes=CFG["num_local_experts"]).float()
                f = one_hot.sum(1).mean(0)
                target_load = CFG["top_k"] / CFG["num_local_experts"]
                pen = torch.where(f < target_load,
                                  torch.full_like(f, CFG["aux_underload_penalty"]),
                                  torch.ones_like(f))
                aux_loss = CFG["num_local_experts"] * (pen * f * P).sum()
                loss = lm_loss + CFG["aux_loss_weight"] * aux_loss + lam_route * route_sup_loss
            else:
                loss = compute_total_loss(
                    lm_loss, logits,
                    CFG["num_local_experts"], CFG["top_k"],
                    CFG["aux_loss_weight"],
                ) + lam_route * route_sup_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, CFG["grad_clip"])
            optimizer.step()
            scheduler.step()

            clear_router_decision(moe_layers)
            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                metrics = monitor.update(ri.detach(), logits.detach())
                print(
                    f"  step {global_step:4d} | loss={loss.item():.4f} "
                    f"| lm={lm_loss.item():.4f} "
                    f"| entropy={metrics['router/entropy']:.3f} "
                    f"| dead={metrics['router/dead_experts']}"
                )

        avg_loss = epoch_loss / len(train_dl)
        print(f"[phase1] Epoch {epoch+1}/{CFG['num_epochs']} avg_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(CFG["output_dir"], "phase1_checkpoint.pt")
    torch.save({
        "router": router.state_dict(),
        "router_config": {"num_experts": CFG["num_local_experts"], "top_k": CFG["top_k"]},
        "lora_state": {
            name: param.data
            for name, param in model.named_parameters()
            if param.requires_grad
        },
    }, ckpt_path)
    print(f"[phase1] Checkpoint saved → {ckpt_path}")

    print("\n[phase1] Running ablation evaluation ...")
    result_with, result_without = ablation_eval(
        model, router, moe_layers, eval_records, tokenizer, device, CFG["eval_batch_size"]
    )

    print(f"\n{'='*65}")
    print("  Phase 1 Ablation Results")
    print(f"{'='*65}")
    print(f"  {'Bucket':<8} {'PPL (with global)':>18} {'PPL (no global)':>16} {'Delta PPL':>10}")
    print(f"  {'-'*8} {'-'*18} {'-'*16} {'-'*10}")
    for b in BUCKETS:
        p_with = result_with["per_bucket_ppl"].get(b, float("nan"))
        p_without = result_without["per_bucket_ppl"].get(b, float("nan"))
        delta = p_without - p_with
        print(f"  {b:<8} {p_with:>18.2f} {p_without:>16.2f} {delta:>+10.2f}")
    print(f"{'='*65}")
    print(f"  Router hit rate (with global): {result_with['overall_hit_rate']:.3f}")
    print(f"  Random baseline:               {result_with['random_baseline']:.3f}")
    print(f"  Dead experts:                  {result_with['dead_experts']}")
    print(f"{'='*65}")

    write_phase1_report(
        result_with, result_without,
        output_path=os.path.join(CFG["output_dir"], "phase1_report.md"),
    )

    GlobalLocalLoraLinear.forward = _orig_forward

    return result_with, result_without

if __name__ == "__main__":
    main()
