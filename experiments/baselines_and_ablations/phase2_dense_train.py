
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, get_moe_layers
from src.eval.monitor import RouterMonitor
from phase1_train import (
    CLASS_NAMES, BUCKETS, CFG as _CFG_BASE,
    build_title2class, load_phase1_records, split_records,
    Phase1Dataset, collate_fn,
    set_router_decision, clear_router_decision,
    _run_expert_warmup,
    _eval_bucket_losses, ablation_eval, write_phase1_report,
    _init_tb,
)

CFG = dict(_CFG_BASE)
CFG.update({
    "output_dir": "outputs/phase2_dense",
    "routing_mode": "dense",
    "entropy_aux_weight": 0.05,
    "class_weighted_route_sup": True,
    "aux_underload_penalty": 1.0,
    "top_k": 2,
    "lr": 2e-4,
    "num_epochs": 3,
})

class DenseSoftRouter(nn.Module):

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        inner = max(256, num_experts * 4)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, inner, bias=True),
            nn.GELU(),
            nn.Linear(inner, num_experts, bias=True),
        )

    def forward(self, query_repr: torch.Tensor):
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
        lora_out_g = ((x_lora @ A_g.T) @ B_g.T).to(orig_dtype)
        h = h + lora_out_g * self.scaling

    if rw is not None:
        B_batch = rw.shape[0]
        E = rw.shape[1]
        tokens_per_seq = T // B_batch if T != B_batch else 1

        rw_exp = rw.to(dev).unsqueeze(1).expand(B_batch, tokens_per_seq, E)
        rw_exp = rw_exp.reshape(T, E)

        expert_delta = torch.zeros_like(h)
        for eid in range(E):
            w = rw_exp[:, eid]
            A = self.lora_A_local[eid].to(dev)
            B = self.lora_B_local[eid].to(dev)
            out = ((x_lora @ A.T) @ B.T).to(orig_dtype) * self.scaling
            expert_delta += w.to(orig_dtype).unsqueeze(-1) * out

        h = h + expert_delta

    return h.reshape(orig_shape[:-1] + (self.out_features,))

def entropy_aux_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    E = gate_weights.shape[-1]
    entropy = -(gate_weights * (gate_weights.clamp(min=1e-9).log())).sum(-1)
    max_entropy = math.log(E)
    return max_entropy - entropy.mean()

def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])

    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase2_dense] Device: {device}")
    print(f"[phase2_dense] Routing mode: {CFG['routing_mode']}")

    print("[phase2_dense] Loading Cora QA data ...")
    all_records = load_phase1_records(CFG["rewritten_dir"], CFG["cora_dir"], seed=CFG["seed"])
    print(f"[phase2_dense] Total records: {len(all_records)}")

    from collections import Counter
    print("[phase2_dense] Community distribution:",
          dict(sorted(Counter(r["community"] for r in all_records).items())))

    train_records, eval_records = split_records(
        all_records,
        max_train_per_bucket=CFG["max_train_per_bucket"],
        max_eval_per_bucket=CFG["max_eval_per_bucket"],
        seed=CFG["seed"],
    )
    print(f"[phase2_dense] Train: {len(train_records)}, Eval: {len(eval_records)}")

    num_experts = CFG["num_local_experts"]
    from collections import Counter as _Counter
    comm_freq = _Counter(r["community"] for r in train_records)
    raw_counts = torch.tensor(
        [comm_freq.get(i, 1) for i in range(num_experts)], dtype=torch.float32
    )
    inv_freq = 1.0 / raw_counts
    class_weights_cpu = inv_freq / inv_freq.mean()
    print("[phase2_dense] Fix A — route_sup class weights (inv-freq, mean=1):")
    for i, (w, cnt) in enumerate(zip(class_weights_cpu.tolist(), raw_counts.tolist())):
        print(f"  Expert {i:2d} ({CLASS_NAMES[i]:22s}): count={int(cnt):4d}, weight={w:.3f}")

    print(f"[phase2_dense] Loading model: {CFG['base_model']}")
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

    model, _sparse_router = inject_moe_lora(
        model,
        rank=CFG["rank"],
        lora_alpha=CFG["lora_alpha"],
        num_local_experts=num_experts,
        use_global_expert=CFG["use_global_expert"],
        top_k=CFG["top_k"],
    )
    moe_layers = get_moe_layers(model)
    print(f"[phase2_dense] MoE layers: {len(moe_layers)}")

    hidden_size = _sparse_router.gate[0].in_features
    router = DenseSoftRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=CFG["top_k"],
    ).to(device)
    model.router = router

    GlobalLocalLoraLinear.forward = _dense_forward
    print("[phase2_dense] Patched GlobalLocalLoraLinear → dense soft routing")

    monitor = RouterMonitor(
        num_experts=num_experts,
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
    print(f"[phase2_dense] Train samples: {len(train_ds)}, batches: {len(train_dl)}")

    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(router.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[phase2_dense] Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=CFG["lr"], weight_decay=0.01)
    total_steps = len(train_dl) * CFG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n[phase2_dense] Starting training ({CFG['num_epochs']} epochs) ...")
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

            weights, topk_idx, logits = router(query_repr.to(torch.float32))

            set_router_decision(moe_layers, weights, topk_idx)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            lm_loss = outputs.loss

            frac = global_step / max(1, total_steps)
            lam_route = CFG["route_sup_weight"] * (1.0 - frac) if CFG["route_sup_anneal"] else CFG["route_sup_weight"]
            cw = class_weights_cpu.to(logits.device)
            route_sup_loss = F.cross_entropy(logits, community_labels, weight=cw)

            ent_loss = entropy_aux_loss(weights)

            loss = lm_loss + lam_route * route_sup_loss + CFG["entropy_aux_weight"] * ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, CFG["grad_clip"])
            optimizer.step()
            scheduler.step()
            clear_router_decision(moe_layers)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                metrics = monitor.update(topk_idx.detach(), logits.detach())
                with torch.no_grad():
                    min_expert_load = weights.mean(0).min().item()
                print(
                    f"  step {global_step:4d} | loss={loss.item():.4f} "
                    f"| lm={lm_loss.item():.4f} "
                    f"| ent_loss={ent_loss.item():.4f} "
                    f"| entropy={metrics['router/entropy']:.3f} "
                    f"| min_gate={min_expert_load:.4f}"
                )

        avg_loss = epoch_loss / len(train_dl)
        print(f"[phase2_dense] Epoch {epoch+1}/{CFG['num_epochs']} avg_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(CFG["output_dir"], "dense_checkpoint.pt")
    torch.save({
        "router": router.state_dict(),
        "router_config": {
            "type": "DenseSoftRouter",
            "num_experts": num_experts,
            "top_k": CFG["top_k"],
        },
        "lora_state": {
            name: param.data
            for name, param in model.named_parameters()
            if param.requires_grad
        },
    }, ckpt_path)
    print(f"[phase2_dense] Checkpoint saved → {ckpt_path}")

    print("\n[phase2_dense] Running ablation evaluation ...")
    result_with, result_without = ablation_eval(
        model, router, moe_layers, eval_records, tokenizer, device,
        CFG["eval_batch_size"],
    )

    print(f"\n{'='*65}")
    print("  Phase 2 Dense-Routing Ablation Results")
    print(f"{'='*65}")
    print(f"  {'Bucket':<8} {'PPL (with global)':>18} {'PPL (no global)':>16} {'Delta PPL':>10}")
    print(f"  {'-'*8} {'-'*18} {'-'*16} {'-'*10}")
    for b in BUCKETS:
        p_with    = result_with["per_bucket_ppl"].get(b, float("nan"))
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
        output_path=os.path.join(CFG["output_dir"], "dense_report.md"),
    )

if __name__ == "__main__":
    main()
