
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

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
from src.eval.monitor import RouterMonitor
from src.eval.router_probe import (
    evaluate_router_hit_rate,
    plot_router_metrics,
    write_phase0_report,
)

CFG = {
    "base_model": "/home/USER/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/",
    "train_jsonl": "data/phase0_train.jsonl",
    "eval_jsonl": "data/phase0_eval.jsonl",
    "output_dir": "outputs/phase0",
    "rank": 8,
    "lora_alpha": 16.0,
    "num_local_experts": 8,
    "use_global_expert": False,
    "top_k": 2,
    "num_epochs": 3,
    "batch_size": 4,
    "lr": 5e-4,
    "max_length": 128,
    "aux_loss_weight": 0.01,
    "route_sup_weight": 0.5,
    "route_sup_anneal": True,
    "expert_warmup": True,
    "warmup_steps_per_expert": 50,
    "grad_clip": 1.0,
    "eval_batch_size": 16,
    "seed": 42,
}

class Phase0Dataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 128):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

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
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        assistant_ids = self.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        n = len(assistant_ids)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == assistant_ids:
                labels[:i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": rec.get("community", 0),
        }

def collate_fn(batch):
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
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
        "community": comms,
    }

def set_router_decision(moe_layers, rw, ri):
    for layer in moe_layers:
        layer._cached_rw = rw
        layer._cached_ri = ri

def clear_router_decision(moe_layers):
    for layer in moe_layers:
        layer._cached_rw = None
        layer._cached_ri = None

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

        print(f"[warmup] Expert {eid}: {len(comm_records)} samples", flush=True)

        ds = _RecordDataset(comm_records, tokenizer, max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)
        dl_iter = iter(dl)

        expert_params = []
        for layer in moe_layers:
            expert_params.append(layer.lora_A_local)
            expert_params.append(layer.lora_B_local)
        opt = torch.optim.AdamW(expert_params, lr=lr, weight_decay=0.01)

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

class _RecordDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        assistant_ids = self.tokenizer.encode(
            "<|im_start|>assistant", add_special_tokens=False
        )
        n = len(assistant_ids)
        ids = input_ids.tolist()
        for i in range(len(ids) - n):
            if ids[i: i + n] == assistant_ids:
                labels[:i + n] = -100
                break
        return {"input_ids": input_ids, "labels": labels, "community": rec.get("community", 0)}

def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase0] Device: {device}")

    print(f"[phase0] Loading model: {CFG['base_model']}")
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
    print(f"[phase0] MoE layers: {len(moe_layers)}")

    _orig_forward = GlobalLocalLoraLinear.forward

    def _patched_forward(self, x, router_weights=None, router_indices=None):
        rw = router_weights if router_weights is not None else getattr(self, '_cached_rw', None)
        ri = router_indices if router_indices is not None else getattr(self, '_cached_ri', None)
        return _orig_forward(self, x, rw, ri)

    GlobalLocalLoraLinear.forward = _patched_forward

    monitor = RouterMonitor(
        num_experts=CFG["num_local_experts"],
        use_tensorboard=True,
        tb_writer=_init_tb(os.path.join(CFG["output_dir"], "tb_logs")),
    )

    all_train_records = []
    with open(CFG["train_jsonl"], encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_train_records.append(json.loads(line))

    if CFG["expert_warmup"]:
        _run_expert_warmup(model, moe_layers, all_train_records, tokenizer, CFG, device)

    train_ds = Phase0Dataset(CFG["train_jsonl"], tokenizer, CFG["max_length"])
    train_dl = DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[phase0] Train samples: {len(train_ds)}, batches: {len(train_dl)}")

    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(router.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[phase0] Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=CFG["lr"], weight_decay=0.01)
    total_steps = len(train_dl) * CFG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n[phase0] Starting training ({CFG['num_epochs']} epochs) ...")
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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            )
            lm_loss = outputs.loss

            frac = global_step / max(1, total_steps)
            if CFG.get("route_sup_anneal", True):
                lam_route = CFG["route_sup_weight"] * (1.0 - frac)
            else:
                lam_route = CFG["route_sup_weight"]
            route_sup_loss = F.cross_entropy(logits, community_labels)

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
                    f"| entropy={metrics['router/entropy']:.3f} "
                    f"| dead={metrics['router/dead_experts']}"
                )
                if monitor.collapse_detected():
                    print("  ⚠️  Router collapse detected!")

        avg_loss = epoch_loss / len(train_dl)
        print(f"[phase0] Epoch {epoch+1}/{CFG['num_epochs']} avg_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(CFG["output_dir"], "router_checkpoint.pt")
    torch.save({
        "router": router.state_dict(),
        "router_config": {
            "num_experts": CFG["num_local_experts"],
            "top_k": CFG["top_k"],
        },
        "lora_state": {
            name: param.data
            for name, param in model.named_parameters()
            if param.requires_grad
        },
    }, ckpt_path)
    print(f"[phase0] Checkpoint saved → {ckpt_path}")

    print("\n[phase0] Evaluating router hit rate ...")
    eval_items = []
    with open(CFG["eval_jsonl"]) as f:
        for line in f:
            line = line.strip()
            if line:
                eval_items.append(json.loads(line))

    model.eval()
    router.eval()

    result = _eval_hit_rate(
        model, router, moe_layers, eval_items, tokenizer,
        device=device, batch_size=CFG["eval_batch_size"],
    )

    print(f"\n{'='*60}")
    print(f"  Phase 0 Router Probe Results")
    print(f"{'='*60}")
    print(f"  Overall hit rate : {result['overall_hit_rate']:.3f}")
    print(f"  Random baseline  : {result['random_baseline']:.3f}  (top_k/num_experts = {CFG['top_k']}/{CFG['num_local_experts']})")
    print(f"  Gate PASSED      : {result['hit_rate_above_random']}")
    print(f"  Dead experts     : {result['dead_experts']}")
    print(f"{'='*60}")

    plot_router_metrics(result, output_dir=CFG["output_dir"])
    write_phase0_report(result, output_path=os.path.join(CFG["output_dir"], "phase0_report.md"))

    GlobalLocalLoraLinear.forward = _orig_forward

    return result

@torch.no_grad()
def _eval_hit_rate(model, router, moe_layers, eval_items, tokenizer, device, batch_size):
    top_k = router.top_k
    num_experts = router.num_experts
    random_baseline = top_k / num_experts

    per_community_hits = defaultdict(list)
    all_indices = []
    all_logits = []

    for i in range(0, len(eval_items), batch_size):
        batch = eval_items[i: i + batch_size]
        queries = [item["query"] for item in batch]
        gt_comms = [item["community"] for item in batch]

        enc = tokenizer(
            queries, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(device)

        embed = model.model.embed_tokens(enc["input_ids"])
        mask_f = enc["attention_mask"].unsqueeze(-1).float()
        query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        rw, ri, logits = router(query_repr.to(torch.float32))

        for j, gt_c in enumerate(gt_comms):
            selected = ri[j].tolist()
            per_community_hits[gt_c].append(int(gt_c in selected))

        all_indices.append(ri.cpu())
        all_logits.append(logits.cpu())

    all_hits = [h for hits in per_community_hits.values() for h in hits]
    overall_hr = sum(all_hits) / len(all_hits) if all_hits else 0.0

    per_c_hr = {c: sum(h)/len(h) for c, h in per_community_hits.items()}

    all_idx = torch.cat(all_indices).reshape(-1)
    freq = torch.zeros(num_experts)
    for eid in all_idx:
        freq[eid] += 1
    freq = freq / freq.sum()

    dead = int((freq == 0).sum().item())

    return {
        "overall_hit_rate": overall_hr,
        "random_baseline": random_baseline,
        "hit_rate_above_random": overall_hr > random_baseline * 1.5,
        "per_community_hit_rate": per_c_hr,
        "load_distribution": freq.tolist(),
        "num_eval_items": len(eval_items),
        "top_k": top_k,
        "num_experts": num_experts,
        "dead_experts": dead,
    }

def _init_tb(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None

if __name__ == "__main__":
    result = main()
    if result["hit_rate_above_random"]:
        print("\n✅ Phase 0 PASSED — proceed to Phase 1.")
    else:
        print("\n❌ Phase 0 FAILED — router cannot learn community structure from text.")
        print("   Action: diagnose router, consider hybrid routing before Phase 1.")
