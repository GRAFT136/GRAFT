
import json, math, os, re, sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.injection import inject_moe_lora, get_moe_layers
from phase1_train import (
    load_phase1_records, split_records,
    CLASS_NAMES, BUCKETS, CFG,
    set_router_decision, clear_router_decision,
)

CKPT_PATH = "outputs/phase1/phase1_checkpoint.pt"
OUTPUT_DIR = "outputs/phase1"

def classify_question(q: str) -> str:
    q = q.lower()
    if "how many" in q or "in-degree" in q or "out-degree" in q:
        return "counting"
    if "shortest path" in q or "path between" in q or "path from" in q:
        return "path"
    return "binary"

_POS = ("yes", "indeed", "correct", "a direct reference exists",
        "there is a direct", "is included", "the cora citation graph shows")
_NEG = ("no.", "negative.", "not", "does not", "no,",
        "within the cora citation network", "no direct",
        "does not participate", "provides no direct")

def _binary_label(text: str) -> str:
    t = text.lower().strip()
    if any(t.startswith(p) for p in _POS):
        return "pos"
    if any(t.startswith(p) for p in _NEG):
        return "neg"
    w = t.split()[0] if t else ""
    if w in ("yes", "indeed", "correct", "true"):
        return "pos"
    if w in ("no", "negative", "false", "not"):
        return "neg"
    return "unk"

def _first_int(text: str):
    m = re.search(r'\b(\d+)\b', text)
    return int(m.group(1)) if m else None

def check_em(generated: str, ground_truth: str, q_type: str):
    if q_type == "counting":
        gt_n = _first_int(ground_truth)
        gen_n = _first_int(generated)
        if gt_n is None or gen_n is None:
            return None
        return gt_n == gen_n
    else:
        gt_l  = _binary_label(ground_truth)
        gen_l = _binary_label(generated)
        if gt_l == "unk" or gen_l == "unk":
            return None
        return gt_l == gen_l

def load_checkpoint(model, router, path):
    ckpt = torch.load(path, map_location="cpu")
    router.load_state_dict(ckpt["router"])
    pmap = dict(model.named_parameters())
    missing = 0
    for name, data in ckpt["lora_state"].items():
        if name in pmap:
            pmap[name].data.copy_(data)
        else:
            missing += 1
    if missing:
        print(f"[ckpt] WARNING: {missing} param names not found in model")
    print(f"[ckpt] Loaded {len(ckpt['lora_state'])} LoRA params from {path}")
    return model, router

@torch.no_grad()
def per_community_hit_rate(model, router, eval_records, tokenizer, device, bs=16):
    per_c = defaultdict(list)
    for i in range(0, len(eval_records), bs):
        batch = eval_records[i:i+bs]
        queries = [r["query"] for r in batch]
        enc = tokenizer(queries, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        embed = model.model.embed_tokens(enc["input_ids"])
        mask_f = enc["attention_mask"].unsqueeze(-1).float()
        qr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        _, ri, _ = router(qr.to(torch.float32))
        for j, rec in enumerate(batch):
            gt_c = rec["community"]
            per_c[gt_c].append(int(gt_c in ri[j].tolist()))
    return {c: (sum(v)/len(v), len(v)) for c, v in per_c.items()}

@torch.no_grad()
def em_eval(model, router, moe_layers, eval_records, tokenizer, device,
            use_global: bool, batch_size=4, max_new=60):
    for layer in moe_layers:
        layer.use_global_expert = use_global

    tag = "WITH" if use_global else "WITHOUT"
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

        embed = model.model.embed_tokens(enc["input_ids"])
        mask_f = enc["attention_mask"].unsqueeze(-1).float()
        qr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        rw, ri, _ = router(qr.to(torch.float32))
        set_router_decision(moe_layers, rw, ri)

        gen_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
        )
        clear_router_decision(moe_layers)

        for j, rec in enumerate(batch):
            new_toks = gen_ids[j][prompt_len:]
            generated = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
            q_type    = classify_question(rec["query"])
            correct   = check_em(generated, rec["answer"], q_type)
            bucket_results[rec["bucket"]].append(correct)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(eval_records))
            print(f"  [{tag}] {done}/{len(eval_records)}", flush=True)

    for layer in moe_layers:
        layer.use_global_expert = True

    stats = {}
    for b, results in bucket_results.items():
        valid = [r for r in results if r is not None]
        pct_valid = len(valid) / len(results) if results else 0
        stats[b] = {
            "acc":       sum(valid) / len(valid) if valid else float("nan"),
            "n_valid":   len(valid),
            "n_total":   len(results),
            "pct_valid": pct_valid,
        }
    return stats

def write_report(per_comm, em_with, em_without, path):
    lines = []
    lines.append("# Phase 1 Post-hoc Evaluation\n\n")

    lines.append("## 1. Per-Community Router Hit Rates\n\n")
    lines.append("| Expert | Topic Class | Hit Rate | N |\n")
    lines.append("|--------|-------------|:--------:|---|\n")
    total_w, total_n = 0.0, 0
    for c in range(len(CLASS_NAMES)):
        hr, n = per_comm.get(c, (0.0, 0))
        dead = " **(dead)**" if hr == 0.0 else ""
        lines.append(f"| {c} | {CLASS_NAMES[c]}{dead} | {hr:.3f} | {n} |\n")
        total_w += hr * n; total_n += n
    overall = total_w / total_n if total_n else 0.0
    rand_bl  = CFG["top_k"] / CFG["num_local_experts"]
    lines.append(f"\n- **Weighted overall**: {overall:.3f}\n")
    lines.append(f"- **Random baseline**: {rand_bl:.3f}  ({CFG['top_k']}/{CFG['num_local_experts']})\n")
    lines.append(f"- **Overall / random**: {overall/rand_bl:.2f}×\n\n")

    lines.append("## 2. EM Ablation (with / without global expert)\n\n")
    lines.append("| Bucket | EM (with) | EM (without) | Δ EM | N valid |\n")
    lines.append("|--------|:---------:|:------------:|:----:|--------|\n")
    for b in BUCKETS:
        w  = em_with.get(b, {})
        wo = em_without.get(b, {})
        acc_w  = w.get("acc", float("nan"))
        acc_wo = wo.get("acc", float("nan"))
        delta  = acc_wo - acc_w
        n = w.get("n_valid", 0)
        pct = w.get("pct_valid", 0)
        warning = " ⚠️ low" if pct < 0.5 else ""
        lines.append(f"| {b:6s} | {acc_w:.3f} | {acc_wo:.3f} | {delta:+.3f} | {n} ({pct:.0%}){warning} |\n")

    lines.append("\n## 3. Interpretation\n\n")
    cross_d  = em_without.get("cross",  {}).get("acc", 0) - em_with.get("cross",  {}).get("acc", 0)
    intra_d  = em_without.get("intra",  {}).get("acc", 0) - em_with.get("intra",  {}).get("acc", 0)
    global_d = em_without.get("global", {}).get("acc", 0) - em_with.get("global", {}).get("acc", 0)
    if cross_d < -0.02 and global_d < -0.02 and (cross_d < intra_d or global_d < intra_d):
        msg = "✅ HYPOTHESIS CONFIRMED: cross and global queries degrade more when global expert removed."
    elif cross_d < intra_d - 0.02 or global_d < intra_d - 0.02:
        msg = "⚠️ PARTIAL: at least one non-intra bucket degrades more than intra."
    elif max(abs(cross_d), abs(intra_d), abs(global_d)) < 0.02:
        msg = ("❌ NEGLIGIBLE: global expert has no differential effect "
               "(consistent with universal shared-adapter behavior; ablation with EM is inconclusive).")
    else:
        msg = "❓ MIXED: results inconclusive."
    lines.append(msg + "\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"[eval] Report saved → {path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Device: {device}")

    print("[eval] Loading Cora data ...")
    all_records = load_phase1_records(CFG["rewritten_dir"], CFG["cora_dir"], seed=CFG["seed"])
    _, eval_records = split_records(
        all_records,
        max_train_per_bucket=CFG["max_train_per_bucket"],
        max_eval_per_bucket=CFG["max_eval_per_bucket"],
        seed=CFG["seed"],
    )
    print(f"[eval] Eval records: {len(eval_records)}")
    from collections import Counter
    print("[eval] Bucket dist:", dict(Counter(r["bucket"] for r in eval_records)))

    print(f"[eval] Loading {CFG['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["base_model"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )

    model, router = inject_moe_lora(
        model,
        rank=CFG["rank"], lora_alpha=CFG["lora_alpha"],
        num_local_experts=CFG["num_local_experts"],
        use_global_expert=CFG["use_global_expert"],
        top_k=CFG["top_k"],
    )
    router = router.to(device)
    moe_layers = get_moe_layers(model)

    _orig = GlobalLocalLoraLinear.forward
    def _patched(self, x, router_weights=None, router_indices=None):
        rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)
        ri = router_indices if router_indices is not None else getattr(self, "_cached_ri", None)
        return _orig(self, x, rw, ri)
    GlobalLocalLoraLinear.forward = _patched

    model, router = load_checkpoint(model, router, CKPT_PATH)
    model.eval(); router.eval()

    print("\n[eval] Step 1: per-community hit rates ...")
    per_comm = per_community_hit_rate(model, router, eval_records, tokenizer, device)
    print(f"\n  {'Expert':>6} | {'Topic Class':<26} | {'Hit Rate':>8} | N")
    print(f"  {'------':>6}-+-{'--------':<26}-+-{'--------':>8}-+----")
    for c in range(len(CLASS_NAMES)):
        hr, n = per_comm.get(c, (0.0, 0))
        dead = " ← DEAD" if hr == 0.0 else ""
        print(f"  {c:>6} | {CLASS_NAMES[c]:<26} | {hr:>8.3f} | {n}{dead}")
    rand_bl = CFG["top_k"] / CFG["num_local_experts"]
    total_w = sum(hr * n for hr, n in per_comm.values())
    total_n = sum(n for _, n in per_comm.values())
    overall = total_w / total_n if total_n else 0.0
    print(f"\n  Overall (weighted): {overall:.3f}   Random baseline: {rand_bl:.3f}   Ratio: {overall/rand_bl:.2f}×")

    print("\n[eval] Step 2: EM evaluation WITH global expert ...")
    em_with    = em_eval(model, router, moe_layers, eval_records, tokenizer, device, use_global=True)
    print("[eval] Step 2: EM evaluation WITHOUT global expert ...")
    em_without = em_eval(model, router, moe_layers, eval_records, tokenizer, device, use_global=False)

    print(f"\n{'='*65}")
    print("  EM Ablation Results")
    print(f"{'='*65}")
    print(f"  {'Bucket':<8} {'EM(with)':>9} {'EM(w/o)':>9} {'Δ':>7}  N_valid")
    print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*7}  {'-'*7}")
    for b in BUCKETS:
        w  = em_with.get(b, {}).get("acc", float("nan"))
        wo = em_without.get(b, {}).get("acc", float("nan"))
        n  = em_with.get(b, {}).get("n_valid", 0)
        print(f"  {b:<8} {w:>9.3f} {wo:>9.3f} {wo-w:>+7.3f}  {n}")
    print(f"{'='*65}")

    write_report(
        per_comm, em_with, em_without,
        path=os.path.join(OUTPUT_DIR, "eval_em_report.md"),
    )
    GlobalLocalLoraLinear.forward = _orig

if __name__ == "__main__":
    main()
