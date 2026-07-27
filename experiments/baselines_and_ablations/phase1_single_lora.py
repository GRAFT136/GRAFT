
import math, os, re, sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from phase1_train import (
    load_phase1_records, split_records,
    Phase1Dataset, collate_fn,
    CLASS_NAMES, BUCKETS, CFG,
)

SL_CFG = dict(CFG)
SL_CFG["rank"]           = 128
SL_CFG["lora_alpha"]     = 256.0
SL_CFG["output_dir"]     = "outputs/phase1/single_lora"
SL_CFG["expert_warmup"]  = False
SL_CFG["route_sup_weight"] = 0.0
SL_CFG["lr"]             = 1e-4

TARGET_MODULES = {"gate_proj", "up_proj", "down_proj"}

class SingleLoraLinear(nn.Module):

    def __init__(self, in_features, out_features, base_weight,
                 base_bias=None, rank=128, lora_alpha=256.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.scaling = lora_alpha / rank

        self.base_weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        self.base_bias   = (nn.Parameter(base_bias.clone(), requires_grad=False)
                            if base_bias is not None else None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, self.in_features)

        h = F.linear(x_flat, self.base_weight, self.base_bias)

        dev   = x_flat.device
        x32   = x_flat.to(torch.float32)
        delta = ((x32 @ self.lora_A.to(dev).T) @ self.lora_B.to(dev).T).to(orig_dtype)
        h = h + delta * self.scaling
        return h.reshape(orig_shape[:-1] + (self.out_features,))

def inject_single_lora(model, rank, lora_alpha):
    for p in model.parameters():
        p.requires_grad_(False)

    targets = []
    for full_name, mod in model.named_modules():
        if full_name.split(".")[-1] in TARGET_MODULES and isinstance(mod, nn.Linear):
            targets.append(full_name)

    for full_name in targets:
        parts  = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child = parts[-1]
        mod   = getattr(parent, child)
        dev   = next(mod.parameters()).device

        new_layer = SingleLoraLinear(
            mod.in_features, mod.out_features,
            mod.weight.data,
            mod.bias.data if mod.bias is not None else None,
            rank=rank, lora_alpha=lora_alpha,
        ).to(dev)
        setattr(parent, child, new_layer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[single_lora] {len(targets)} layers replaced | "
          f"Total: {total:,} | Trainable: {trainable:,}")
    return model

def _classify(q: str) -> str:
    q = q.lower()
    if "how many" in q or "in-degree" in q or "out-degree" in q:
        return "counting"
    return "binary"

_POS = ("yes", "indeed", "correct", "a direct reference exists",
        "there is a direct", "is included", "the cora citation graph shows")
_NEG = ("no.", "negative.", "not", "does not", "no,",
        "within the cora citation network", "no direct",
        "does not participate", "provides no direct")

def _binary_label(text: str) -> str:
    t = text.lower().strip()
    if any(t.startswith(p) for p in _POS): return "pos"
    if any(t.startswith(p) for p in _NEG): return "neg"
    w = t.split()[0] if t else ""
    if w in ("yes", "indeed", "correct"): return "pos"
    if w in ("no", "negative", "not"):    return "neg"
    return "unk"

def _first_int(text: str):
    m = re.search(r'\b(\d+)\b', text)
    return int(m.group(1)) if m else None

def check_em(generated: str, ground_truth: str, q_type: str):
    if q_type == "counting":
        gt_n = _first_int(ground_truth)
        gn_n = _first_int(generated)
        if gt_n is None or gn_n is None: return None
        return gt_n == gn_n
    else:
        gt_l  = _binary_label(ground_truth)
        gen_l = _binary_label(generated)
        if gt_l == "unk" or gen_l == "unk": return None
        return gt_l == gen_l

@torch.no_grad()
def em_eval(model, eval_records, tokenizer, device, batch_size=4, max_new=60):
    eos_list = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id   = eos_list[0] if eos_list else tokenizer.eos_token_id
    bucket_results = defaultdict(list)

    for i in range(0, len(eval_records), batch_size):
        batch = eval_records[i:i+batch_size]
        prompts = [
            f"<|im_start|>user\n{r['query']}<|im_end|>\n<|im_start|>assistant\n"
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=280).to(device)
        prompt_len = enc["input_ids"].shape[1]

        gen_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
        )
        for j, rec in enumerate(batch):
            new_toks  = gen_ids[j][prompt_len:]
            generated = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            q_type    = _classify(rec["query"])
            correct   = check_em(generated, rec["answer"], q_type)
            bucket_results[rec["bucket"]].append(correct)

        if (i // batch_size) % 10 == 0:
            print(f"  [em] {min(i+batch_size,len(eval_records))}/{len(eval_records)}", flush=True)

    stats = {}
    for b, results in bucket_results.items():
        valid = [r for r in results if r is not None]
        pct   = len(valid) / len(results) if results else 0
        stats[b] = {
            "acc":       sum(valid)/len(valid) if valid else float("nan"),
            "n_valid":   len(valid),
            "pct_valid": pct,
        }
    return stats

def main():
    cfg = SL_CFG
    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[single_lora] Device: {device}")

    print("[single_lora] Loading Cora data ...")
    all_records = load_phase1_records(cfg["rewritten_dir"], cfg["cora_dir"], seed=cfg["seed"])
    train_records, eval_records = split_records(
        all_records,
        max_train_per_bucket=cfg["max_train_per_bucket"],
        max_eval_per_bucket=cfg["max_eval_per_bucket"],
        seed=cfg["seed"],
    )
    print(f"[single_lora] Train: {len(train_records)}, Eval: {len(eval_records)}")

    print(f"[single_lora] Loading {cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = inject_single_lora(model, rank=cfg["rank"], lora_alpha=cfg["lora_alpha"])

    train_ds = Phase1Dataset(train_records, tokenizer, cfg["max_length"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=collate_fn, num_workers=0)
    print(f"[single_lora] Batches/epoch: {len(train_dl)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_dl) * cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"\n[single_lora] Training ({cfg['num_epochs']} epochs, rank={cfg['rank']}) ...")
    global_step = 0
    for epoch in range(cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1
            if global_step % 100 == 0:
                print(f"  step {global_step:4d} | loss={loss.item():.4f}", flush=True)

        avg = epoch_loss / len(train_dl)
        print(f"[single_lora] Epoch {epoch+1}/{cfg['num_epochs']} avg_loss={avg:.4f}")

    ckpt_path = os.path.join(cfg["output_dir"], "single_lora_ckpt.pt")
    torch.save(
        {"lora_state": {n: p.data for n, p in model.named_parameters() if p.requires_grad}},
        ckpt_path,
    )
    print(f"[single_lora] Checkpoint → {ckpt_path}")

    print("\n[single_lora] EM evaluation ...")
    model.eval()
    em_stats = em_eval(model, eval_records, tokenizer, device)

    print(f"\n{'='*55}")
    print("  Single LoRA Baseline — EM Accuracy")
    print(f"{'='*55}")
    for b in BUCKETS:
        s = em_stats.get(b, {})
        acc = s.get("acc", float("nan"))
        n   = s.get("n_valid", 0)
        pct = s.get("pct_valid", 0)
        print(f"  {b:<8} EM={acc:.3f}  (n={n}, {pct:.0%} valid)")
    print(f"{'='*55}")

    report_path = os.path.join(cfg["output_dir"], "single_lora_report.md")
    with open(report_path, "w") as f:
        f.write(f"# Single LoRA Baseline — Phase 1\n\n")
        f.write(f"rank={cfg['rank']}, lora_alpha={cfg['lora_alpha']}, "
                f"epochs={cfg['num_epochs']}, trainable≈242M\n\n")
        f.write("## Exact-Match Accuracy per Bucket\n\n")
        f.write("| Bucket | EM Accuracy | N valid | % valid |\n")
        f.write("|--------|:-----------:|:-------:|:-------:|\n")
        for b in BUCKETS:
            s   = em_stats.get(b, {})
            acc = s.get("acc", float("nan"))
            n   = s.get("n_valid", 0)
            pct = s.get("pct_valid", 0)
            f.write(f"| {b:6s} | {acc:.3f} | {n} | {pct:.0%} |\n")
    print(f"[single_lora] Report → {report_path}")

if __name__ == "__main__":
    main()
