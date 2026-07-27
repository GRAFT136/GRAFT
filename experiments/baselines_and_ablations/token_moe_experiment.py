
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from phase1_train import BUCKETS, Phase1Dataset, collate_fn, load_phase1_records, split_records
from phase1_single_lora import _classify, check_em
from src.data.other_sft_loader import load_other_sft_records, num_communities as other_num_communities
from src.model.token_injection import inject_token_moe_lora
from src.model.token_moe_lora import (
    clear_token_router_caches,
    token_load_balancing_loss,
    token_route_supervision_loss,
)

BASE_MODEL = (
    "/home/USER/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
)

def resolve_model_path(pattern: str) -> str:
    path = Path(pattern)
    if path.is_dir() and any(path.iterdir()):
        subdirs = [child for child in path.iterdir() if child.is_dir()]
        if subdirs:
            return str(subdirs[0])
    return pattern

@torch.no_grad()
def token_em_eval(model, token_layers, eval_records, tokenizer, device, batch_size=16, max_new=60):
    eos_list = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_list[0] if eos_list else tokenizer.eos_token_id
    bucket_results = defaultdict(list)

    for start in range(0, len(eval_records), batch_size):
        batch = eval_records[start:start + batch_size]
        prompts = [
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n<|im_start|>assistant\n"
            for rec in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=280).to(device)
        prompt_len = enc["input_ids"].shape[1]

        gen_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
        )
        clear_token_router_caches(token_layers)

        for idx, rec in enumerate(batch):
            new_tokens = gen_ids[idx][prompt_len:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            q_type = _classify(rec["query"])
            bucket_results[rec["bucket"]].append(check_em(generated, rec["answer"], q_type))

        if (start // batch_size) % 5 == 0:
            print(f"  [em] {min(start + batch_size, len(eval_records))}/{len(eval_records)}", flush=True)

    stats = {}
    for bucket, results in bucket_results.items():
        valid = [value for value in results if value is not None]
        stats[bucket] = {
            "acc": sum(valid) / len(valid) if valid else float("nan"),
            "n_valid": len(valid),
            "n_total": len(results),
            "pct_valid": len(valid) / len(results) if results else 0,
        }
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--dataset", default="cora", choices=[
        "cora", "citeseer", "amazon-photo", "amazon-computers", "wn18rr", "fb15k-237", "pubmed"
    ])
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=float, default=4.0)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--no_global_expert", action="store_true")
    parser.add_argument("--router_hidden", type=int, default=None)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--route_sup_weight", type=float, default=0.1)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_per_bucket", type=int, default=800)
    parser.add_argument("--max_eval_per_bucket", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--other_sft_root", default="../other_sft_data")
    parser.add_argument("--other_graph_root", default="../other_graph_dataset")
    parser.add_argument("--fb15k_min_rel_freq", type=int, default=0)
    parser.add_argument("--rewritten_dir", default="../Cora/sft_data/rewritten")
    parser.add_argument("--cora_dir", default="../Cora/cora_dataset")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_path = resolve_model_path(args.model_path or BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[token_moe] Device: {device}  Model: {model_path}")

    if args.dataset == "cora":
        all_records = load_phase1_records(args.rewritten_dir, args.cora_dir, seed=args.seed)
        num_experts = args.num_experts or 7
    else:
        all_records = load_other_sft_records(
            args.dataset, args.other_sft_root, args.other_graph_root, seed=args.seed,
            fb15k_min_rel_freq=args.fb15k_min_rel_freq,
        )
        num_experts = args.num_experts or other_num_communities(
            args.dataset, args.other_graph_root, sft_root=args.other_sft_root,
            fb15k_min_rel_freq=args.fb15k_min_rel_freq,
        )

    train_records, eval_records = split_records(
        all_records, args.max_train_per_bucket, args.max_eval_per_bucket, seed=args.seed
    )
    print(f"[token_moe] Train: {len(train_records)}  Eval: {len(eval_records)}")
    print("[token_moe] Train bucket dist:", dict(Counter(rec["bucket"] for rec in train_records)))
    print(f"[token_moe] num_experts={num_experts} top_k={args.top_k} rank={args.rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model, token_layers = inject_token_moe_lora(
        model,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        num_experts=num_experts,
        top_k=args.top_k,
        use_global_expert=not args.no_global_expert,
        router_hidden=args.router_hidden,
        router_temperature=args.router_temperature,
    )

    train_ds = Phase1Dataset(train_records, tokenizer, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    trainable = [param for param in model.parameters() if param.requires_grad]
    print(f"[token_moe] Trainable params: {sum(param.numel() for param in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            community = batch["community"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs.loss
            route_loss = token_route_supervision_loss(token_layers, community, attention_mask)
            aux_loss = token_load_balancing_loss(token_layers, attention_mask)
            loss = lm_loss + args.route_sup_weight * route_loss + args.aux_loss_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            clear_token_router_caches(token_layers)

            epoch_loss += float(loss.detach().cpu())
            global_step += 1
            if global_step % 20 == 0:
                print(
                    f"  step {global_step:4d} | loss={loss.item():.4f} | "
                    f"lm={lm_loss.item():.4f} | route={route_loss.item():.4f}",
                    flush=True,
                )

        print(f"[token_moe] Epoch {epoch + 1}/{args.num_epochs} avg_loss={epoch_loss / len(train_dl):.4f}")

    model.eval()
    print("\n[token_moe] Running EM eval ...")
    em_stats = token_em_eval(
        model, token_layers, eval_records, tokenizer, device,
        batch_size=args.eval_batch_size,
    )

    out_name = "token_moe_lora" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path("outputs/token_moe") / args.dataset / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": "token_moe_lora",
        "dataset": args.dataset,
        "num_experts": num_experts,
        "top_k": args.top_k,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "route_sup_weight": args.route_sup_weight,
        "aux_loss_weight": args.aux_loss_weight,
        "fb15k_min_rel_freq": args.fb15k_min_rel_freq,
        "buckets": {},
    }
    print(f"\n{'=' * 55}\n  token_moe_lora - EM Accuracy\n{'=' * 55}")
    for bucket in BUCKETS:
        stats = em_stats.get(bucket, {})
        result["buckets"][bucket] = stats
        print(
            f"  {bucket:8s} EM={stats.get('acc', float('nan')):.3f}  "
            f"(n_valid={stats.get('n_valid', 0)}, {stats.get('pct_valid', 0):.0%})"
        )
    print(f"{'=' * 55}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[token_moe] Results saved -> {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()