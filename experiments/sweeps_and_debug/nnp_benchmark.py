
from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from phase1_single_lora import inject_single_lora
from phase1_train import clear_router_decision, set_router_decision
from src.data.other_sft_loader import load_other_sft_records
from src.model.injection import get_moe_layers, inject_moe_lora
from src.model.losses import compute_total_loss
from src.model.moe_lora import GlobalLocalLoraLinear
from src.model.token_injection import inject_token_moe_lora
from src.model.token_moe_lora import (
    clear_token_router_caches,
    token_load_balancing_loss,
    token_route_supervision_loss,
)

MODEL_0P5B = (
    "/home/USER/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
)
MODEL_7B = (
    "/home/USER/.cache/huggingface/hub/"
    "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

DATASET_NAMES = ("children", "stack_elec", "fb")
BASELINE_METHODS = {"zero_shot", "full_context", "subgraphrag", "graphtoken", "gnp"}
ADAPTER_METHODS = {"single_lora", "old_moe", "token_moe"}
ALL_METHODS = tuple(sorted(BASELINE_METHODS | ADAPTER_METHODS))

ENTITY_RE = re.compile(r"<([^>]+)>")
AND_MORE_RE = re.compile(r"\band\s+\d+\s+more\b", re.I)
FB_SIMPLE_ANSWER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,&\-'/()]+")

STACK_SFT_FILES = [
    ("01_existence_qa.jsonl", "existence"),
    ("02_counting_qa.jsonl", "counting"),
    ("03_traversal_qa.jsonl", "traversal"),
    ("04_substructure_qa.jsonl", "substructure"),
    ("05_multihop_qa.jsonl", "multihop"),
    ("06_node_info_qa.jsonl", "node_info"),
    ("07_edge_info_qa.jsonl", "edge_info"),
    ("08_node_basic_qa.jsonl", "node_basic"),
]

def resolve_model_path(pattern: str) -> str:
    path = Path(pattern)
    if path.is_dir() and any(path.iterdir()):
        subdirs = [child for child in path.iterdir() if child.is_dir()]
        if subdirs:
            return str(sorted(subdirs)[0])
        return str(path)
    return pattern

def _clean_text(text: str) -> str:
    return " ".join(html.unescape(str(text)).split())

def _stable_hash_int(text: str) -> int:
    total = 1469598103934665603
    for byte in text.encode("utf-8", errors="ignore"):
        total ^= byte
        total *= 1099511628211
        total &= (1 << 64) - 1
    return total

def _norm_item(text: str) -> str:
    value = _clean_text(text).casefold()
    value = value.strip(" <>.,:;\"'`()[]{}")
    value = re.sub(r"\s+", " ", value)
    return value

def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        cleaned = _clean_text(item)
        key = _norm_item(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out

def _wrap_item(item: str) -> str:
    cleaned = _clean_text(item).strip("<>")
    return f"<{cleaned}>"

def _canonical_answer(items: Sequence[str]) -> str:
    canonical = _dedupe_preserve(items)
    if not canonical:
        return "NONE"
    return "; ".join(_wrap_item(item) for item in canonical)

def _parse_generated_items(text: str) -> List[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    upper = cleaned.upper()
    if upper.startswith("OOC") or upper == "NONE":
        return []

    angle_items = _dedupe_preserve(ENTITY_RE.findall(cleaned))
    if angle_items:
        return angle_items

    tail = cleaned
    if ":" in tail and not tail.strip().startswith("<"):
        head, rest = tail.split(":", 1)
        if len(head.split()) <= 8:
            tail = rest.strip()

    pieces = re.split(r";|\n", tail)
    if len(pieces) == 1 and "," in tail:
        pieces = re.split(r",", tail)
    return _dedupe_preserve(piece.strip(" <>.,:;\"'") for piece in pieces)

def _dataset_index(name: str) -> int:
    return DATASET_NAMES.index(name)

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def _load_children_records(root: Path) -> List[Dict[str, Any]]:
    path = root / "children" / "sft" / "09_neighbor_prediction_qa.jsonl"
    records: List[Dict[str, Any]] = []
    for item in _read_jsonl(path):
        query = _clean_text(item.get("query", ""))
        answer = _clean_text(item.get("answer", ""))
        if AND_MORE_RE.search(answer):
            continue
        anchor_titles = ENTITY_RE.findall(query)
        gold_items = _dedupe_preserve(ENTITY_RE.findall(answer))
        if not anchor_titles or not gold_items:
            continue
        anchor = anchor_titles[0]
        records.append({
            "dataset": "children",
            "query": query,
            "original_answer": answer,
            "canonical_answer": _canonical_answer(gold_items),
            "gold_items": gold_items,
            "anchor": anchor,
            "relation": "co_purchase",
            "expert_key": anchor,
            "source_file": path.name,
        })
    return records

def _stack_query_kind(query: str) -> Optional[str]:
    lowered = query.lower()
    if (
        "which users responded to the post" in lowered
        or "who responded to the post" in lowered
        or "who replied to the post" in lowered
    ):
        return "all_responders"
    if "which users commented on the post" in lowered or "who commented on the post" in lowered:
        return "all_commenters"
    return None

def _stack_parse_items(kind: str, answer: str) -> List[str]:
    text = _clean_text(answer)

    if kind == "all_commenters":
        names = _dedupe_preserve(re.findall(r"(.+?) left \d+ comments?", text))
        if names:
            return names

    patterns = [
        r"(?i)the responders were (.+?)(?:\.|$)",
        r"(?i)unique users responded:\s*(.+?)(?:\.|$)",
        r"(?i)users responded:\s*(.+?)(?:\.|$)",
        r"(?i)the commenters were (.+?)(?:\.|$)",
        r"(?i)users commented:\s*(.+?)(?:\.|$)",
        r"(?i)the only responder(?: was|,)\s*([^.,]+)",
        r"(?i)the only commenter(?: was|,)\s*([^.,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        tail = match.group(1)
        tail = re.sub(r"(?i)both responses.*$", "", tail).strip()
        tail = re.sub(r"(?i)for a total of .*", "", tail).strip()
        pieces = re.split(r",| and ", tail)
        items = _dedupe_preserve(piece.strip(" <>.,:;\"'") for piece in pieces)
        if items:
            return items
    return []

def _load_stack_records(root: Path) -> List[Dict[str, Any]]:
    path = root / "Stack_elec" / "09_neighbor_qa_augment(2).jsonl"
    records: List[Dict[str, Any]] = []
    for item in _read_jsonl(path):
        query = _clean_text(item.get("query", ""))
        answer = _clean_text(item.get("answer", ""))
        kind = _stack_query_kind(query)
        if kind is None:
            continue
        gold_items = _stack_parse_items(kind, answer)
        if not gold_items:
            continue
        anchor = _clean_text(item.get("_entity", "")) or query
        records.append({
            "dataset": "stack_elec",
            "query": query,
            "original_answer": answer,
            "canonical_answer": _canonical_answer(gold_items),
            "gold_items": gold_items,
            "anchor": anchor,
            "relation": kind,
            "expert_key": kind,
            "source_file": path.name,
        })
    return records

def _fb_answer_items(answer: str) -> List[str]:
    text = _clean_text(answer)
    if not text or text.lower().startswith(("yes", "no")):
        return []
    if FB_SIMPLE_ANSWER_RE.fullmatch(text):
        return [text]
    angle_items = _dedupe_preserve(ENTITY_RE.findall(text))
    if angle_items and not AND_MORE_RE.search(text):
        return angle_items
    return []

def _load_fb_records(root: Path) -> List[Dict[str, Any]]:
    path = root / "fb" / "09_neighbor_qa_aug.jsonl"
    records: List[Dict[str, Any]] = []
    for item in _read_jsonl(path):
        if str(item.get("_task")) != "09":
            continue
        key = str(item.get("_key", "09__unknown__0"))
        parts = key.split("__")
        relation = parts[1] if len(parts) >= 3 else key
        for idx, qa in enumerate(item.get("qa_pairs", [])):
            query = _clean_text(qa.get("query", ""))
            answer = _clean_text(qa.get("answer", ""))
            gold_items = _fb_answer_items(answer)
            if not query or not gold_items:
                continue
            anchor = f"fb_fact::{parts[-1] if parts else '0'}::{qa.get('fact_id', idx)}::{_stable_hash_int(query)}"
            records.append({
                "dataset": "fb",
                "query": query,
                "original_answer": answer,
                "canonical_answer": _canonical_answer(gold_items),
                "gold_items": gold_items,
                "anchor": anchor,
                "relation": relation,
                "expert_key": relation,
                "source_file": path.name,
            })
    return records

def load_nnp_records(nnp_root: str, dataset: str) -> List[Dict[str, Any]]:
    root = Path(nnp_root)
    if dataset == "children":
        return _load_children_records(root)
    if dataset == "stack_elec":
        return _load_stack_records(root)
    if dataset == "fb":
        return _load_fb_records(root)
    raise ValueError(f"Unknown dataset: {dataset}")

def _aux_anchor(query: str, answer: str, fallback: str) -> str:
    entities = ENTITY_RE.findall(query)
    if entities:
        return _clean_text(entities[0])
    entities = ENTITY_RE.findall(answer)
    if entities:
        return _clean_text(entities[0])
    query_text = _clean_text(query)
    if query_text:
        return query_text[:96]
    return fallback

def _make_aux_record(
    dataset: str,
    query: str,
    answer: str,
    relation: str,
    expert_key: str,
    source_file: str,
    anchor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    query_text = _clean_text(query)
    answer_text = _clean_text(answer)
    if not query_text or not answer_text:
        return None
    anchor_text = anchor or _aux_anchor(query_text, answer_text, source_file)
    return {
        "dataset": dataset,
        "query": query_text,
        "original_answer": answer_text,
        "canonical_answer": answer_text,
        "train_answer": answer_text,
        "train_prompt_style": "generic",
        "train_source": "aux_sft",
        "gold_items": _dedupe_preserve(ENTITY_RE.findall(answer_text)),
        "anchor": anchor_text,
        "relation": relation,
        "expert_key": expert_key,
        "source_file": source_file,
    }

def _load_children_aux_records(nnp_root: str) -> List[Dict[str, Any]]:
    path = Path(nnp_root) / "children" / "sft" / "augmented" / "09_neighbor_prediction_qa_aug.jsonl"
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for item in _read_jsonl(path):
        query = item.get("query", "")
        answer = item.get("answer", "")
        anchor_titles = ENTITY_RE.findall(str(query)) or ENTITY_RE.findall(str(answer))
        anchor = _clean_text(anchor_titles[0]) if anchor_titles else None
        rec = _make_aux_record(
            dataset="children",
            query=str(query),
            answer=str(answer),
            relation="aux_neighbor_prediction",
            expert_key=anchor or path.name,
            source_file=path.name,
            anchor=anchor,
        )
        if rec is not None:
            records.append(rec)
    return records

def _load_stack_aux_records(stack_root: str) -> List[Dict[str, Any]]:
    base = Path(stack_root)
    roots = [(base / "sft_data", ""), (base / "sft_data" / "augmented", "_aug")]
    records: List[Dict[str, Any]] = []
    for root, suffix in roots:
        for fname, relation_name in STACK_SFT_FILES:
            path = root / (fname.replace(".jsonl", f"{suffix}.jsonl") if suffix else fname)
            if not path.exists():
                continue
            for item in _read_jsonl(path):
                query = item.get("query", "")
                answer = item.get("answer", "")
                rec = _make_aux_record(
                    dataset="stack_elec",
                    query=str(query),
                    answer=str(answer),
                    relation=f"aux::{relation_name}",
                    expert_key=f"aux::{path.name}",
                    source_file=path.name,
                    anchor=path.name,
                )
                if rec is not None:
                    records.append(rec)
    return records

def _load_fb_aux_records(args) -> List[Dict[str, Any]]:
    raw_records = load_other_sft_records(
        "fb15k-237",
        sft_root=args.other_sft_root,
        graph_root=args.other_graph_root,
        seed=args.seed,
        use_v2=True,
        v2_root=args.other_sft_v2_root,
    )
    records: List[Dict[str, Any]] = []
    for item in raw_records:
        relation = f"aux::{item.get('bucket', 'unknown')}::{item.get('source_file', 'unknown')}"
        rec = _make_aux_record(
            dataset="fb",
            query=str(item.get("query", "")),
            answer=str(item.get("answer", "")),
            relation=relation,
            expert_key=f"aux_comm::{item.get('community', 0)}",
            source_file=str(item.get("source_file", "unknown")),
        )
        if rec is not None:
            records.append(rec)
    return records

def load_related_sft_train_records(args, dataset: str) -> List[Dict[str, Any]]:
    if dataset == "children":
        records = _load_children_aux_records(args.nnp_root)
    elif dataset == "stack_elec":
        records = _load_stack_aux_records(args.stack_root)
    elif dataset == "fb":
        records = _load_fb_aux_records(args)
    else:
        records = []

    if args.max_aux_train_records > 0 and len(records) > args.max_aux_train_records:
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: args.max_aux_train_records]
    return records

def summarise_train_pool(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_records": len(records),
        "source_files": Counter(rec.get("source_file", "unknown") for rec in records).most_common(20),
        "relations": Counter(rec.get("relation", "unknown") for rec in records).most_common(20),
        "train_sources": Counter(rec.get("train_source", "nnp") for rec in records).most_common(),
    }

def assign_communities(records: List[Dict[str, Any]], num_experts: int) -> None:
    for rec in records:
        rec["community"] = _stable_hash_int(f"{rec['dataset']}::{rec['expert_key']}") % max(1, num_experts)

def split_records(
    records: List[Dict[str, Any]],
    seed: int,
    eval_ratio: float,
    max_train_records: int,
    max_eval_records: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_relation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_relation[str(rec.get("relation", "unknown"))].append(dict(rec))

    train_records: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    for group in by_relation.values():
        rng.shuffle(group)
        if len(group) == 1:
            train_records.extend(group)
            continue
        n_eval = max(1, int(round(len(group) * eval_ratio)))
        n_eval = min(n_eval, len(group) - 1)
        eval_records.extend(group[:n_eval])
        train_records.extend(group[n_eval:])

    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    if max_train_records > 0:
        train_records = train_records[: max_train_records]
    if max_eval_records > 0:
        eval_records = eval_records[: max_eval_records]
    return train_records, eval_records

def build_corpus_docs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for rec in records:
        docs.append({
            "text": (
                f"dataset={rec['dataset']} relation={rec['relation']} community={rec['community']}\n"
                f"question: {rec['query']}\n"
                f"answer: {rec['canonical_answer']}"
            ),
            "dataset": rec["dataset"],
            "relation": rec["relation"],
            "community": rec["community"],
        })
    return docs

def build_feature_lookup(records: List[Dict[str, Any]], num_experts: int) -> Dict[str, Any]:
    anchor_freq = Counter(rec["anchor"] for rec in records)
    relation_freq = Counter(rec["relation"] for rec in records)
    relation_sizes: Dict[str, List[int]] = defaultdict(list)
    unique_items = set()
    for rec in records:
        relation_sizes[rec["relation"]].append(len(rec["gold_items"]))
        for item in rec["gold_items"]:
            unique_items.add(_norm_item(item))
    relation_avg_size = {
        relation: sum(vals) / len(vals)
        for relation, vals in relation_sizes.items()
        if vals
    }
    overall_avg_size = (
        sum(len(rec["gold_items"]) for rec in records) / len(records)
        if records
        else 0.0
    )
    overall_max_size = max((len(rec["gold_items"]) for rec in records), default=0)
    return {
        "num_experts": num_experts,
        "num_train_records": len(records),
        "anchor_freq": anchor_freq,
        "relation_freq": relation_freq,
        "relation_avg_size": relation_avg_size,
        "overall_avg_size": overall_avg_size,
        "overall_max_size": overall_max_size,
        "num_unique_items": len(unique_items),
    }

def classify_size_bucket(avg_size: float) -> int:
    if avg_size <= 1.2:
        return 0
    if avg_size <= 3.0:
        return 1
    return 2

def record_feature_vector(record: Dict[str, Any], lookup: Dict[str, Any], variant: str) -> torch.Tensor:
    dataset_onehot = torch.zeros(3, dtype=torch.float32)
    dataset_onehot[_dataset_index(record["dataset"])] = 1.0

    avg_size = float(lookup["relation_avg_size"].get(record["relation"], lookup["overall_avg_size"]))
    size_onehot = torch.zeros(3, dtype=torch.float32)
    size_onehot[classify_size_bucket(avg_size)] = 1.0

    node_feat = torch.tensor([
        math.log1p(float(lookup["anchor_freq"].get(record["anchor"], 0))),
        math.log1p(float(lookup["relation_freq"].get(record["relation"], 0))),
    ], dtype=torch.float32)

    if variant == "graphtoken":
        graph_stats = torch.zeros(4, dtype=torch.float32)
        comm = torch.zeros(1, dtype=torch.float32)
    else:
        graph_stats = torch.tensor([
            math.log1p(float(lookup["num_train_records"])),
            math.log1p(float(lookup["overall_avg_size"])),
            math.log1p(float(lookup["overall_max_size"])),
            math.log1p(float(lookup["num_unique_items"])),
        ], dtype=torch.float32)
        denom = max(1, int(lookup["num_experts"]) - 1)
        comm = torch.tensor([float(record["community"]) / float(denom)], dtype=torch.float32)

    bucket = torch.tensor([math.log1p(avg_size)], dtype=torch.float32)
    return torch.cat([node_feat, graph_stats, comm, bucket, dataset_onehot, size_onehot], dim=0)

class SimpleRetriever:
    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        self.doc_tokens: List[Counter] = []
        self.doc_lens: List[int] = []
        df = Counter()
        for doc in docs:
            tokens = Counter(re.findall(r"[a-z0-9_]+", _norm_item(doc["text"])))
            self.doc_tokens.append(tokens)
            self.doc_lens.append(sum(tokens.values()))
            for token in tokens:
                df[token] += 1
        self.n_docs = max(1, len(docs))
        self.idf = {tok: math.log((self.n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = Counter(re.findall(r"[a-z0-9_]+", _norm_item(query)))
        d_tokens = self.doc_tokens[doc_idx]
        score = 0.0
        for token, q_count in q_tokens.items():
            if token not in d_tokens:
                continue
            score += self.idf.get(token, 1.0) * min(q_count, d_tokens[token])
        return score / math.sqrt(max(1, self.doc_lens[doc_idx]))

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        scored = [(self.score(query, idx), idx) for idx in range(len(self.docs))]
        scored.sort(reverse=True)
        out: List[Dict[str, Any]] = []
        for score, idx in scored[:top_k]:
            doc = dict(self.docs[idx])
            doc["score"] = float(score)
            out.append(doc)
        return out

class GraphPromptEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, prefix_len: int) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prefix_len * hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(features.shape[0], self.prefix_len, self.hidden_size)

def _system_prompt() -> str:
    return (
        "You are a graph neighbor prediction assistant. "
        "Return only the final neighbor set as a semicolon-separated list of items wrapped in angle brackets, "
        "for example <A>; <B>. "
        "If there is exactly one answer, still wrap it in angle brackets. "
        "Do not include explanations. If the answer cannot be determined from the provided information, answer OOC."
    )

def _generic_system_prompt() -> str:
    return (
        "You are a graph reasoning assistant. "
        "Answer the user's question directly based on graph structure and entity relations learned during training. "
        "Be concise and do not add extra commentary."
    )

def build_prompt(
    query: str,
    graph_context: Optional[str] = None,
    retrieved: Optional[List[Dict[str, Any]]] = None,
) -> str:
    user_parts: List[str] = []
    if graph_context:
        user_parts.append("Training-corpus facts:\n" + graph_context)
    if retrieved:
        lines = [f"[{idx}] {doc['text']}" for idx, doc in enumerate(retrieved[:50], start=1)]
        user_parts.append("Retrieved facts:\n" + "\n".join(lines))
    user_parts.append(f"Question: {query}")
    user_body = "\n\n".join(user_parts)
    return (
        "<|im_start|>system\n"
        f"{_system_prompt()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_body}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_generic_prompt(query: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{_generic_system_prompt()}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{query}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def build_training_text(record: Dict[str, Any]) -> str:
    prompt_style = record.get("train_prompt_style", "nnp")
    if prompt_style == "generic":
        prompt = build_generic_prompt(record["query"])
        answer = _clean_text(record.get("train_answer") or record.get("original_answer") or record.get("canonical_answer", ""))
    else:
        prompt = build_prompt(record["query"])
        answer = str(record.get("train_answer") or record.get("canonical_answer", "NONE"))
    return prompt + answer + "<|im_end|>"

class PromptTuningDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int, feature_lookup: Dict[str, Any], variant: str) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_lookup = feature_lookup
        self.variant = variant
        self._assistant_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = build_training_text(rec)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        ids = input_ids.tolist()
        n = len(self._assistant_ids)
        for pos in range(len(ids) - n):
            if ids[pos:pos + n] == self._assistant_ids:
                labels[: pos + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": int(rec["community"]),
            "features": record_feature_vector(rec, self.feature_lookup, self.variant),
        }

def collate_prompt_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    communities = torch.zeros(len(batch), dtype=torch.long)
    features = torch.stack([item["features"] for item in batch], dim=0)
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
        "features": features,
    }

class CanonicalSFTDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._assistant_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        text = build_training_text(rec)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        ids = input_ids.tolist()
        n = len(self._assistant_ids)
        for pos in range(len(ids) - n):
            if ids[pos:pos + n] == self._assistant_ids:
                labels[: pos + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": int(rec["community"]),
        }

def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
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

def build_inference_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    return tokenizer, model

def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)

def model_device(model) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")

def model_input_device(model) -> torch.device:
    return model.model.embed_tokens.weight.device

def pooled_query_repr(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, router=None) -> torch.Tensor:
    embed_layer = model.model.embed_tokens
    embed_device = embed_layer.weight.device
    embed = embed_layer(input_ids.to(embed_device))
    mask_f = attention_mask.to(embed_device).unsqueeze(-1).float()
    query_repr = (embed * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
    if router is not None:
        router_device = next(router.parameters()).device
        return query_repr.to(device=router_device, dtype=torch.float32)
    return query_repr.to(torch.float32)

def generate_greedy(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    prefix_embeds: Optional[torch.Tensor] = None,
    max_input_tokens: int = 4096,
) -> str:
    device = model_device(model)
    enc = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"][:, -max_input_tokens:].to(device)
    attention_mask = enc["attention_mask"][:, -max_input_tokens:].to(device)

    if prefix_embeds is not None:
        prefix_embeds = prefix_embeds.to(device=device, dtype=model.model.embed_tokens.weight.dtype)
        input_embeds = model.model.embed_tokens(input_ids)
        input_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        prefix_mask = torch.ones(
            (attention_mask.shape[0], prefix_embeds.shape[1]),
            device=device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=True)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

    past = outputs.past_key_values
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        outputs = model(input_ids=next_token, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def chunk_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def available_context_tokens(context_budget: int) -> int:
    return max(256, context_budget - 256)

def serialise_docs(docs: List[Dict[str, Any]], tokenizer, max_tokens: int) -> Tuple[str, bool]:
    pieces = []
    used = 0
    for doc in docs:
        text = doc["text"]
        n_tokens = chunk_tokens(text, tokenizer)
        if used + n_tokens > max_tokens:
            return "\n".join(pieces), True
        pieces.append(text)
        used += n_tokens
    return "\n".join(pieces), False

def train_prompt_encoder(
    model,
    tokenizer,
    train_records: List[Dict[str, Any]],
    feature_lookup: Dict[str, Any],
    variant: str,
    prompt_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
) -> GraphPromptEncoder:
    device = model_device(model)
    hidden_size = model.config.hidden_size
    feature_dim = 14
    encoder = GraphPromptEncoder(feature_dim, hidden_size, prompt_len).to(device)
    train_ds = PromptTuningDataset(train_records, tokenizer, max_length, feature_lookup, variant)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_prompt_batch(batch, pad_token_id),
        num_workers=0,
    )
    trainable = list(encoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = max(1, len(dl) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    freeze_model(model)
    model.eval()
    encoder.train()
    for epoch in range(epochs):
        running = 0.0
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device)
            prefix = encoder(features)
            text_embeds = model.model.embed_tokens(input_ids)
            prefix = prefix.to(dtype=text_embeds.dtype)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
            prefix_mask = torch.ones((input_ids.shape[0], prompt_len), device=device, dtype=attention_mask.dtype)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            prefix_labels = torch.full((input_ids.shape[0], prompt_len), -100, dtype=labels.dtype, device=device)
            full_labels = torch.cat([prefix_labels, labels], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu())
        print(f"[prompt:{variant}] epoch={epoch + 1}/{epochs} avg_loss={running / max(1, len(dl)):.4f}")

    encoder.eval()
    return encoder

def score_prediction(record: Dict[str, Any], generated: str) -> Dict[str, Any]:
    gold_items = list(record["gold_items"])
    pred_items = _parse_generated_items(generated)

    gold_map = {_norm_item(item): item for item in gold_items}
    pred_map = {_norm_item(item): item for item in pred_items}
    gold_keys = set(gold_map)
    pred_keys = set(pred_map)
    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)
    parsed = bool(pred_items) or generated.strip().upper().startswith(("OOC", "NONE"))
    return {
        "gold_items": gold_items,
        "pred_items": [pred_map[key] for key in pred_keys],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact_match": gold_keys == pred_keys,
        "parsed": parsed,
    }

def aggregate_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tp = sum(item["tp"] for item in details)
    total_fp = sum(item["fp"] for item in details)
    total_fn = sum(item["fn"] for item in details)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    n_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    enm = sum(1 for item in details if item["exact_match"]) / max(1, len(details))
    her = total_fp / max(1, total_tp + total_fp)
    parsed_rate = sum(1 for item in details if item["parsed"]) / max(1, len(details))
    avg_gold = sum(len(item["gold_items"]) for item in details) / max(1, len(details))
    avg_pred = sum(len(item["pred_items"]) for item in details) / max(1, len(details))
    return {
        "n_records": len(details),
        "precision": precision,
        "recall": recall,
        "n_f1": n_f1,
        "enm": enm,
        "her": her,
        "parsed_rate": parsed_rate,
        "avg_gold_size": avg_gold,
        "avg_pred_size": avg_pred,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

def eval_baseline_records(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    method: str,
    graph_context: Optional[str],
    graph_context_truncated: bool,
    retriever: Optional[SimpleRetriever],
    prompt_encoder: Optional[GraphPromptEncoder],
    feature_lookup: Optional[Dict[str, Any]],
    context_budget: int,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    if prompt_encoder is not None:
        prompt_encoder.eval()

    details: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    ooc_count = 0
    for idx, rec in enumerate(records):
        prefix_embeds = None
        retrieved_docs = None
        ooc = False
        if method == "zero_shot":
            prompt = build_prompt(rec["query"])
        elif method == "full_context":
            if graph_context_truncated or chunk_tokens(graph_context or "", tokenizer) > available_context_tokens(context_budget):
                prompt = build_prompt(rec["query"])
                generated = "OOC"
                ooc = True
            else:
                prompt = build_prompt(rec["query"], graph_context=graph_context)
        elif method == "subgraphrag":
            assert retriever is not None
            retrieved_docs = retriever.retrieve(rec["query"], top_k=50)
            retrieved_text = "\n".join(doc["text"] for doc in retrieved_docs)
            if chunk_tokens(retrieved_text, tokenizer) > available_context_tokens(context_budget):
                prompt = build_prompt(rec["query"])
                generated = "OOC"
                ooc = True
            else:
                prompt = build_prompt(rec["query"], retrieved=retrieved_docs)
        elif method in {"graphtoken", "gnp"}:
            assert prompt_encoder is not None and feature_lookup is not None
            device = model_device(model)
            features = record_feature_vector(rec, feature_lookup, method).unsqueeze(0).to(device)
            prefix_embeds = prompt_encoder(features).to(device)
            prompt = build_prompt(rec["query"])
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        if not ooc:
            generated = generate_greedy(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                prefix_embeds=prefix_embeds,
                max_input_tokens=context_budget,
            )
        detail = score_prediction(rec, generated)
        details.append(detail)
        predictions.append({
            "idx": idx,
            "dataset": rec["dataset"],
            "method": method,
            "query": rec["query"],
            "original_answer": rec["original_answer"],
            "canonical_answer": rec["canonical_answer"],
            "generated": generated,
            "gold_items": detail["gold_items"],
            "pred_items": detail["pred_items"],
            "tp": detail["tp"],
            "fp": detail["fp"],
            "fn": detail["fn"],
            "exact_match": detail["exact_match"],
            "parsed": detail["parsed"],
            "ooc": ooc,
        })
        if ooc:
            ooc_count += 1
    result = aggregate_metrics(details)
    result["ooc_rate"] = ooc_count / max(1, len(records))
    return result, predictions

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def patch_old_moe_forward() -> None:
    if getattr(GlobalLocalLoraLinear, "_nnp_cache_patch", False):
        return
    original_forward = GlobalLocalLoraLinear.forward

    def _forward_with_cache(self, x, router_weights=None, router_indices=None):
        rw = router_weights if router_weights is not None else getattr(self, "_cached_rw", None)
        ri = router_indices if router_indices is not None else getattr(self, "_cached_ri", None)
        return original_forward(self, x, rw, ri)

    GlobalLocalLoraLinear.forward = _forward_with_cache
    GlobalLocalLoraLinear._nnp_cache_patch = True

def unique_trainable_params(model, router=None) -> List[torch.nn.Parameter]:
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

def build_train_model_and_tokenizer(args, method: str):
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
    old_moe_layers: List[Any] = []
    token_layers: List[Any] = []
    if method == "single_lora":
        model = inject_single_lora(model, rank=args.single_rank, lora_alpha=args.single_alpha)
    elif method == "old_moe":
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
        patch_old_moe_forward()
    elif method == "token_moe":
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
        raise ValueError(f"Unknown adapter method: {method}")
    return tokenizer, model, router, old_moe_layers, token_layers

def train_adapter(
    args,
    method: str,
    model,
    router,
    old_moe_layers,
    token_layers,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    if len(dataloader) == 0 or args.max_steps == 0 or args.epochs == 0:
        return {"steps": 0, "avg_loss": float("nan"), "avg_lm_loss": float("nan")}

    trainable = unique_trainable_params(model, router)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    planned_steps = min(args.max_steps, len(dataloader) * args.epochs) if args.max_steps > 0 else len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, planned_steps // 10),
        num_training_steps=max(1, planned_steps),
    )

    model.train()
    if router is not None:
        router.train()

    total_loss = 0.0
    total_lm = 0.0
    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            batch_device = model_input_device(model)
            input_ids = batch["input_ids"].to(batch_device)
            attention_mask = batch["attention_mask"].to(batch_device)
            labels = batch["labels"].to(batch_device)
            community = batch["community"].to(device)

            logits = None
            if method == "old_moe":
                with torch.no_grad():
                    query_repr = pooled_query_repr(model, input_ids, attention_mask, router=router)
                weights, indices, logits = router(query_repr)
                set_router_decision(old_moe_layers, weights, indices)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            lm_loss = outputs.loss
            loss = lm_loss
            if method == "old_moe" and logits is not None:
                if args.route_sup_weight:
                    loss = loss + args.route_sup_weight * F.cross_entropy(logits, community)
                if args.aux_loss_weight:
                    loss = compute_total_loss(loss, logits, args.num_experts, args.top_k, args.aux_loss_weight)
            elif method == "token_moe":
                if args.route_sup_weight:
                    route_loss = token_route_supervision_loss(token_layers, community, attention_mask).to(loss.device)
                    loss = loss + args.route_sup_weight * route_loss
                if args.aux_loss_weight:
                    aux_loss = token_load_balancing_loss(token_layers, attention_mask).to(loss.device)
                    loss = loss + args.aux_loss_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()

            if method == "old_moe":
                clear_router_decision(old_moe_layers)
            elif method == "token_moe":
                clear_token_router_caches(token_layers)

            step += 1
            total_loss += float(loss.detach().cpu())
            total_lm += float(lm_loss.detach().cpu())
            if step % args.log_every == 0 or step == 1:
                print(f"[{method}] step={step}/{planned_steps} loss={loss.item():.4f} lm={lm_loss.item():.4f}", flush=True)
            if step >= planned_steps:
                return {
                    "steps": step,
                    "avg_loss": total_loss / max(1, step),
                    "avg_lm_loss": total_lm / max(1, step),
                }

    return {
        "steps": step,
        "avg_loss": total_loss / max(1, step),
        "avg_lm_loss": total_lm / max(1, step),
    }

@torch.no_grad()
def generate_adapter_predictions(
    args,
    method: str,
    model,
    router,
    old_moe_layers,
    token_layers,
    records: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
) -> List[str]:
    eos_list = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = eos_list[0] if eos_list else tokenizer.eos_token_id
    model.eval()
    if router is not None:
        router.eval()

    outputs_text: List[str] = []
    batch_device = model_input_device(model)
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start:start + args.eval_batch_size]
        prompts = [build_prompt(rec["query"]) for rec in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to(batch_device)
        prompt_len = enc["input_ids"].shape[1]

        if method == "old_moe":
            query_repr = pooled_query_repr(model, enc["input_ids"], enc["attention_mask"], router=router)
            weights, indices, _ = router(query_repr)
            set_router_decision(old_moe_layers, weights, indices)

        gen_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_id,
        )
        if method == "old_moe":
            clear_router_decision(old_moe_layers)
        elif method == "token_moe":
            clear_token_router_caches(token_layers)

        for row_idx in range(len(batch)):
            generated = tokenizer.decode(gen_ids[row_idx][prompt_len:], skip_special_tokens=True).strip()
            outputs_text.append(generated)
    return outputs_text

def save_trainable_state(path: Path, model, router, args, train_metrics, eval_metrics) -> None:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad or "lora_" in name or "router" in name or "token_router" in name:
            state[name] = param.detach().cpu()
    if router is not None:
        for name, param in router.named_parameters():
            state[f"router.{name}"] = param.detach().cpu()
    torch.save({
        "args": vars(args),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "trainable_state": state,
    }, path)

def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_baseline_suite(
    args,
    dataset: str,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    methods: List[str],
    dataset_dir: Path,
) -> List[Dict[str, Any]]:
    if not methods:
        return []

    tokenizer, model = build_inference_model(args.model_path)
    docs = build_corpus_docs(train_records)
    retriever = SimpleRetriever(docs)
    feature_lookup = build_feature_lookup(train_records, args.num_experts)
    graph_context, graph_context_truncated = serialise_docs(
        docs,
        tokenizer,
        max_tokens=available_context_tokens(args.max_context_tokens),
    )

    results = []
    for method in methods:
        print(f"[baseline] dataset={dataset} method={method}")
        prompt_encoder = None
        if method in {"graphtoken", "gnp"}:
            prompt_encoder = train_prompt_encoder(
                model=model,
                tokenizer=tokenizer,
                train_records=train_records,
                feature_lookup=feature_lookup,
                variant=method,
                prompt_len=args.prompt_len if method == "graphtoken" else args.prompt_len * 2,
                epochs=args.prompt_epochs,
                batch_size=args.batch_size,
                lr=args.prompt_lr,
                max_length=args.max_length,
            )

        result, predictions = eval_baseline_records(
            model=model,
            tokenizer=tokenizer,
            records=eval_records,
            method=method,
            graph_context=graph_context if method == "full_context" else None,
            graph_context_truncated=graph_context_truncated,
            retriever=retriever if method == "subgraphrag" else None,
            prompt_encoder=prompt_encoder,
            feature_lookup=feature_lookup if method in {"graphtoken", "gnp"} else None,
            context_budget=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
        )

        out_dir = dataset_dir / method
        save_json(out_dir / "results.json", {
            "dataset": dataset,
            "method": method,
            "model_path": args.model_path,
            "n_train": len(train_records),
            "n_eval": len(eval_records),
            "result": result,
        })
        save_jsonl(out_dir / "predictions.jsonl", predictions)
        print(
            f"[baseline-done] dataset={dataset} method={method} "
            f"N-F1={result['n_f1']:.3f} ENM={result['enm']:.3f} HER={result['her']:.3f}"
        )
        results.append({"dataset": dataset, "method": method, **result})

    del model
    clear_cuda()
    return results

def run_adapter_method(
    args,
    dataset: str,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    method: str,
    dataset_dir: Path,
) -> Dict[str, Any]:
    tokenizer, model, router, old_moe_layers, token_layers = build_train_model_and_tokenizer(args, method)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = model_input_device(model)

    train_ds = CanonicalSFTDataset(train_records, tokenizer, args.max_length)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id),
        num_workers=0,
    )

    train_metrics = train_adapter(args, method, model, router, old_moe_layers, token_layers, train_dl, device)
    generated = generate_adapter_predictions(args, method, model, router, old_moe_layers, token_layers, eval_records, tokenizer, device)

    details: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for idx, (rec, text) in enumerate(zip(eval_records, generated)):
        detail = score_prediction(rec, text)
        details.append(detail)
        predictions.append({
            "idx": idx,
            "dataset": rec["dataset"],
            "method": method,
            "query": rec["query"],
            "original_answer": rec["original_answer"],
            "canonical_answer": rec["canonical_answer"],
            "generated": text,
            "gold_items": detail["gold_items"],
            "pred_items": detail["pred_items"],
            "tp": detail["tp"],
            "fp": detail["fp"],
            "fn": detail["fn"],
            "exact_match": detail["exact_match"],
            "parsed": detail["parsed"],
        })
    eval_metrics = aggregate_metrics(details)

    out_dir = dataset_dir / method
    save_json(out_dir / "results.json", {
        "dataset": dataset,
        "method": method,
        "model_path": args.model_path,
        "n_train": len(train_records),
        "n_train_nnp": sum(1 for rec in train_records if rec.get("train_source", "nnp") == "nnp"),
        "n_train_aux": sum(1 for rec in train_records if rec.get("train_source", "nnp") != "nnp"),
        "n_eval": len(eval_records),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    })
    save_jsonl(out_dir / "predictions.jsonl", predictions)
    save_trainable_state(out_dir / "adapter_state.pt", model, router, args, train_metrics, eval_metrics)

    print(
        f"[adapter-done] dataset={dataset} method={method} "
        f"N-F1={eval_metrics['n_f1']:.3f} ENM={eval_metrics['enm']:.3f} HER={eval_metrics['her']:.3f}"
    )

    del model
    clear_cuda()
    return {"dataset": dataset, "method": method, **eval_metrics}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnp_root", default="../next_neighbor_prediction(nnp)")
    parser.add_argument("--stack_root", default="../Stack_elec_dataset")
    parser.add_argument("--other_sft_root", default="../other_sft_data")
    parser.add_argument("--other_sft_v2_root", default="../other_sft_data_v2")
    parser.add_argument("--other_graph_root", default="../other_graph_dataset")
    parser.add_argument("--datasets", nargs="+", default=["children", "stack_elec", "fb"])
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    parser.add_argument("--model_path", default=MODEL_0P5B)
    parser.add_argument("--output_root", default="outputs/nnp_benchmark")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_records", type=int, default=2000)
    parser.add_argument("--max_eval_records", type=int, default=300)
    parser.add_argument("--include_related_sft_train", action="store_true")
    parser.add_argument("--max_aux_train_records", type=int, default=0)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--prompt_epochs", type=int, default=1)
    parser.add_argument("--prompt_lr", type=float, default=3e-4)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--single_rank", type=int, default=8)
    parser.add_argument("--single_alpha", type=float, default=16.0)
    parser.add_argument("--moe_rank", type=int, default=4)
    parser.add_argument("--moe_alpha", type=float, default=8.0)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--route_sup_weight", type=float, default=0.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.model_path = resolve_model_path(args.model_path)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    methods = list(args.methods)
    invalid = [method for method in methods if method not in BASELINE_METHODS and method not in ADAPTER_METHODS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")

    run_name = args.tag or "default"
    output_root = Path(args.output_root) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        if dataset not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset: {dataset}")
        raw_records = load_nnp_records(args.nnp_root, dataset)
        if not raw_records:
            raise RuntimeError(f"No evaluable records found for {dataset}")
        assign_communities(raw_records, args.num_experts)
        train_records, eval_records = split_records(
            raw_records,
            seed=args.seed,
            eval_ratio=args.eval_ratio,
            max_train_records=args.max_train_records,
            max_eval_records=args.max_eval_records,
        )
        if not train_records or not eval_records:
            raise RuntimeError(f"Insufficient split for {dataset}: train={len(train_records)} eval={len(eval_records)}")

        dataset_dir = output_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        split_summary = {
            "dataset": dataset,
            "raw_records": len(raw_records),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "avg_train_gold_size": sum(len(rec["gold_items"]) for rec in train_records) / max(1, len(train_records)),
            "avg_eval_gold_size": sum(len(rec["gold_items"]) for rec in eval_records) / max(1, len(eval_records)),
            "top_relations": Counter(rec["relation"] for rec in raw_records).most_common(10),
        }
        save_json(dataset_dir / "split_summary.json", split_summary)
        print(
            f"[data] dataset={dataset} raw={len(raw_records)} train={len(train_records)} eval={len(eval_records)}"
        )

        baseline_methods = [method for method in methods if method in BASELINE_METHODS]
        adapter_methods = [method for method in methods if method in ADAPTER_METHODS]

        adapter_train_records = [dict(rec) for rec in train_records]
        aux_train_records: List[Dict[str, Any]] = []
        if adapter_methods and args.include_related_sft_train:
            aux_train_records = load_related_sft_train_records(args, dataset)
            adapter_train_records.extend(aux_train_records)
            assign_communities(adapter_train_records, args.num_experts)
            save_json(dataset_dir / "adapter_train_summary.json", {
                "dataset": dataset,
                "nnp_train_records": len(train_records),
                "aux_train_records": len(aux_train_records),
                "total_adapter_train_records": len(adapter_train_records),
                "nnp_summary": summarise_train_pool(train_records),
                "aux_summary": summarise_train_pool(aux_train_records),
                "adapter_summary": summarise_train_pool(adapter_train_records),
            })
            print(
                f"[adapter-train-pool] dataset={dataset} nnp={len(train_records)} aux={len(aux_train_records)} total={len(adapter_train_records)}"
            )

        all_results.extend(run_baseline_suite(args, dataset, train_records, eval_records, baseline_methods, dataset_dir))
        for method in adapter_methods:
            all_results.append(run_adapter_method(args, dataset, adapter_train_records, eval_records, method, dataset_dir))

    save_json(output_root / "summary.json", {"results": all_results})
    print(f"[nnp] summary -> {output_root / 'summary.json'}")

if __name__ == "__main__":
    main()