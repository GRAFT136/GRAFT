
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from phase1_single_lora import _binary_label, _classify, _first_int, check_em
from phase1_train import BUCKETS, Phase1Dataset, collate_fn, load_phase1_records, split_records
from src.data.graph_loader import load_cora
from src.data.other_graph_loader import load_other_graph
from src.data.other_sft_loader import (
    build_lookup,
    load_other_sft_records,
    _extract_entity_refs,
)

MODEL_7B = (
    "/home/USER/.cache/huggingface/hub/"
    "models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)

DATASET_DEFAULTS = {
    "cora": {
        "cora_dir": "../Cora/cora_dataset",
        "rewritten_dir": "../Cora/sft_data/rewritten",
        "use_v2": False,
    },
    "citeseer": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": False},
    "wn18rr": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": False},
    "amazon-computers": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": True},
    "amazon-photo": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": True},
    "pubmed": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": True},
    "fb15k-237": {"graph_root": "../other_graph_dataset", "sft_root": "../other_sft_data", "use_v2": True},
}

STRUCTURAL_BUCKETS = {"existence", "counting", "node_basic", "node_info", "edge_info", "classification", "relation"}
PROMPT_OVERHEAD_TOKENS = 256
REASONING_SOURCE_FILES = {
    "01_existence_qa.jsonl",
    "02_counting_qa.jsonl",
    "03_traversal_qa.jsonl",
    "04_substructure_qa.jsonl",
    "05_multihop_qa.jsonl",
    "05_neighbor_qa.jsonl",
    "07_relation_qa.jsonl",
}
_QUOTE_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")

def _norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

def _chunk_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def _available_context_tokens(context_budget: int) -> int:
    return max(256, context_budget - PROMPT_OVERHEAD_TOKENS)

def _extract_titles(query: str) -> List[str]:
    return re.findall(r"<([^>]+)>", query)

def _guess_task_bucket(record: Dict[str, Any]) -> str:
    return str(record.get("bucket", "unknown"))

def _record_is_reasoning(record: Dict[str, Any]) -> bool:
    source_file = record.get("source_file")
    if source_file:
        return source_file in REASONING_SOURCE_FILES
    return True

def _is_structural_record(record: Dict[str, Any]) -> bool:
    source_file = record.get("source_file")
    if source_file:
        return source_file in REASONING_SOURCE_FILES
    bucket = _guess_task_bucket(record)
    if bucket in STRUCTURAL_BUCKETS:
        return True
    query = str(record.get("query", "")).lower()
    return query.startswith(("did ", "has ", "was ", "is ")) or "how many" in query or "degree" in query

def _infer_answer_mode(query: str, answer: str = "") -> str:
    q = query.lower().strip()
    if any(key in q for key in ("how many", "total count", "sum of", "in-degree", "out-degree")):
        return "counting"
    if q.startswith(("does ", "do ", "is ", "are ", "can ", "has ", "have ")):
        return "binary"
    if any(key in q for key in ("directly connected", "direct link", "direct edge", "reference one another", "connected (directly or indirectly)")):
        return "binary"
    if "what is the semantic relation" in q or "what is the relationship" in q:
        return "relation"
    if "what entities does" in q or "which entity do you reach" in q or re.search(r'what is the ".+" of', q):
        return "entity"
    if "what research area" in q or "which category" in q or "belongs to" in q:
        return "label"
    if _binary_label(answer) != "unk":
        return "binary"
    if _first_int(answer) is not None:
        return "counting"
    if _extract_entity_refs(answer):
        return "entity"
    if _QUOTE_RE.search(answer):
        return "label"
    return "open"

def _normalize_free_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

def _extract_quoted_values(text: str) -> List[str]:
    values: List[str] = []
    for left, right in _QUOTE_RE.findall(text):
        value = left or right
        if value:
            values.append(value)
    return values

def _check_answer(generated: str, ground_truth: str, query: str) -> Optional[bool]:
    mode = _infer_answer_mode(query, ground_truth)
    if mode == "binary":
        return check_em(generated, ground_truth, "binary")
    if mode == "counting":
        return check_em(generated, ground_truth, "counting")
    if mode == "relation":
        gt_values = _extract_quoted_values(ground_truth)
        gen_values = _extract_quoted_values(generated)
        gt_key = _normalize_free_text(gt_values[0] if gt_values else ground_truth)
        gen_key = _normalize_free_text(gen_values[0] if gen_values else generated)
        if not gt_key or not gen_key:
            return None
        return gt_key == gen_key or gt_key in gen_key or gen_key in gt_key
    if mode == "entity":
        gt_entities = [str(x) for x in _extract_entity_refs(ground_truth) if not isinstance(x, int)]
        if len(gt_entities) > 1:
            gt_entities = gt_entities[1:]
        if not gt_entities:
            gt_entities = [ground_truth]
        gen_entities = [str(x) for x in _extract_entity_refs(generated) if not isinstance(x, int)]
        gt_norms = [_normalize_free_text(x) for x in gt_entities if _normalize_free_text(x)]
        if gen_entities:
            gen_norms = [_normalize_free_text(x) for x in gen_entities if _normalize_free_text(x)]
        else:
            gen_text = _normalize_free_text(generated)
            gen_norms = [gen_text] if gen_text else []
        if not gt_norms or not gen_norms:
            return None
        return any(gt == gen or gt in gen or gen in gt for gt in gt_norms for gen in gen_norms)
    if mode == "label":
        gt_values = _extract_quoted_values(ground_truth)
        gt_key = _normalize_free_text(gt_values[0] if gt_values else ground_truth)
        gen_values = _extract_quoted_values(generated)
        gen_key = _normalize_free_text(gen_values[0] if gen_values else generated)
        if not gt_key or not gen_key:
            return None
        return gt_key == gen_key or gt_key in gen_key or gen_key in gt_key
    gt_key = _normalize_free_text(ground_truth)
    gen_key = _normalize_free_text(generated)
    if not gt_key or not gen_key:
        return None
    return gt_key == gen_key or gt_key in gen_key or gen_key in gt_key

@dataclass
class DatasetBundle:
    name: str
    records: List[Dict[str, Any]]
    train_records: List[Dict[str, Any]]
    eval_records: List[Dict[str, Any]]
    graph: Optional[Dict[str, Any]]
    graph_stats: Dict[str, float]
    feature_lookup: Dict[str, Any]
    corpus_docs: List[Dict[str, Any]]

class SimpleRetriever:

    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self.docs = docs
        self.doc_tokens: List[Counter] = []
        self.doc_lens: List[int] = []
        df = Counter()
        for doc in docs:
            tokens = Counter(re.findall(r"[a-z0-9_]+", _norm_text(doc["text"])))
            self.doc_tokens.append(tokens)
            self.doc_lens.append(sum(tokens.values()))
            for tok in tokens:
                df[tok] += 1
        self.n_docs = max(1, len(docs))
        self.idf = {tok: math.log((self.n_docs + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = Counter(re.findall(r"[a-z0-9_]+", _norm_text(query)))
        d_tokens = self.doc_tokens[doc_idx]
        score = 0.0
        for tok, q_count in q_tokens.items():
            if tok not in d_tokens:
                continue
            score += self.idf.get(tok, 1.0) * min(q_count, d_tokens[tok])
        return score / math.sqrt(max(1, self.doc_lens[doc_idx]))

    def retrieve(self, query: str, top_k: int = 50, exclude: Optional[set] = None) -> List[Dict[str, Any]]:
        scored = []
        exclude = exclude or set()
        for idx, doc in enumerate(self.docs):
            if idx in exclude:
                continue
            scored.append((self.score(query, idx), idx))
        scored.sort(reverse=True)
        out = []
        for score, idx in scored[:top_k]:
            item = dict(self.docs[idx])
            item["score"] = float(score)
            out.append(item)
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
        out = self.net(features)
        return out.view(features.shape[0], self.prefix_len, self.hidden_size)

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
        text = (
            f"<|im_start|>user\n{rec['query']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['answer']}<|im_end|>"
        )
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        ids = input_ids.tolist()
        n = len(self._assistant_ids)
        for i in range(len(ids) - n):
            if ids[i:i + n] == self._assistant_ids:
                labels[: i + n] = -100
                break
        return {
            "input_ids": input_ids,
            "labels": labels,
            "community": int(rec.get("community", 0)),
            "features": _entity_feature_vector(rec, self.feature_lookup, self.variant),
        }

def _collate_prompt_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
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

def _build_model(model_path: str):
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

def _freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)

def _model_device(model) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")

def _generate_greedy(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 64,
    prefix_embeds: Optional[torch.Tensor] = None,
    max_input_tokens: int = 4096,
) -> str:
    device = _model_device(model)
    enc = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"][:, -max_input_tokens:].to(device)
    attn = enc["attention_mask"][:, -max_input_tokens:].to(device)

    if prefix_embeds is not None:
        prefix_embeds = prefix_embeds.to(device=device, dtype=model.model.embed_tokens.weight.dtype)
        input_embeds = model.model.embed_tokens(input_ids)
        input_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        prefix_mask = torch.ones(input_embeds.shape[:2], device=device, dtype=attn.dtype)
        attn = torch.cat([prefix_mask, attn], dim=1)
        outputs = model(inputs_embeds=input_embeds, attention_mask=attn, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated: List[int] = []
        for _ in range(max_new_tokens):
            generated.append(int(next_token.item()))
            if next_token.item() == tokenizer.eos_token_id:
                break
            attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
            outputs = model(input_ids=next_token, attention_mask=attn, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = []
    for _ in range(max_new_tokens):
        generated.append(int(next_token.item()))
        if next_token.item() == tokenizer.eos_token_id:
            break
        attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
        outputs = model(input_ids=next_token, attention_mask=attn, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def _answer_prompt(query: str, graph_context: Optional[str] = None, retrieved: Optional[List[Dict[str, Any]]] = None) -> str:
    mode = _infer_answer_mode(query)
    if mode == "counting":
        answer_format = "Return only the final integer."
    elif mode == "binary":
        answer_format = "Return only Yes or No."
    elif mode == "relation":
        answer_format = "Return only the relation name."
    elif mode == "entity":
        answer_format = "Return only the final entity name or names."
    elif mode == "label":
        answer_format = "Return only the final category or label name."
    else:
        answer_format = "Return only one short answer sentence."
    user_parts: List[str] = []
    if graph_context:
        user_parts.append("Graph facts:\n" + graph_context)
    if retrieved:
        fact_lines = [f"[{idx}] {item['text']}" for idx, item in enumerate(retrieved[:50], start=1)]
        user_parts.append("Retrieved graph facts:\n" + "\n".join(fact_lines))
    user_parts.append(f"Question: {query}")
    user_parts.append(answer_format)
    user_body = "\n\n".join(user_parts)
    return (
        "<|im_start|>system\n"
        "You are a graph reasoning assistant. Use only the provided graph information. "
        "Do not restate the facts. If the answer cannot be determined from the provided graph information, answer OOC.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_body}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _serialise_docs(docs: List[Dict[str, Any]], tokenizer, max_tokens: int) -> Tuple[str, bool]:
    pieces = []
    used = 0
    for item in docs:
        text = item["text"]
        n = _chunk_tokens(text, tokenizer)
        if used + n > max_tokens:
            return "\n".join(pieces), True
        pieces.append(text)
        used += n
    return "\n".join(pieces), False

def _graph_stats_from_graph(graph: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if graph is None:
        return {}
    stats: Dict[str, float] = {}
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    stats["num_nodes"] = float(len(nodes))
    stats["num_edges"] = float(len(edges))
    if nodes and "nx_graph" in graph:
        G = graph["nx_graph"]
        degrees = [int(G.degree(nid)) for nid in G.nodes()]
        stats["avg_degree"] = float(sum(degrees) / len(degrees))
        stats["max_degree"] = float(max(degrees))
    return stats

def _build_feature_lookup(dataset: str, graph: Optional[Dict[str, Any]], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lookup: Dict[str, Any] = {"dataset": dataset}
    if graph is not None:
        lookup["graph"] = graph
        if "nx_graph" in graph:
            G = graph["nx_graph"]
            deg = dict(G.degree())
            try:
                import networkx as nx
                pagerank = nx.pagerank(G, alpha=0.85)
            except Exception:
                pagerank = {nid: 0.0 for nid in G.nodes()}
            lookup["node_features"] = {
                int(nid): torch.tensor([
                    math.log1p(deg.get(nid, 0)),
                    float(pagerank.get(nid, 0.0)),
                ], dtype=torch.float32)
                for nid in G.nodes()
            }
    if dataset in {"wn18rr", "fb15k-237"}:
        lookup["lookup"] = build_lookup(
            dataset,
            graph_root=DATASET_DEFAULTS[dataset]["graph_root"],
            sft_root=DATASET_DEFAULTS[dataset]["sft_root"],
            use_v2=DATASET_DEFAULTS[dataset]["use_v2"],
        )
    lookup["records_by_community"] = defaultdict(list)
    for rec in records:
        lookup["records_by_community"][int(rec.get("community", 0))].append(rec)
    return lookup

def _entity_feature_vector(record: Dict[str, Any], lookup: Dict[str, Any], variant: str = "gnp") -> torch.Tensor:
    dataset = lookup["dataset"]
    query = str(record.get("query", ""))
    q_type = _classify(query)
    q_type_onehot = torch.zeros(6, dtype=torch.float32)
    q_map = {"binary": 0, "counting": 1, "path": 2, "multihop": 3, "node": 4, "other": 5}
    q_type_onehot[q_map.get(q_type, 5)] = 1.0

    entities = _extract_entity_refs(query)
    feats: List[torch.Tensor] = []
    node_features = lookup.get("node_features", {})
    graph = lookup.get("graph")
    if entities:
        for ent in entities[:3]:
            nid = None
            if isinstance(ent, int):
                nid = ent
            elif dataset == "cora":
                title2id = {v["text"].lower(): k for k, v in graph["nodes"].items()} if graph and "nodes" in graph else {}
                nid = title2id.get(str(ent).lower())
            else:
                entity2id = lookup.get("lookup", {}).get("name2id", {})
                nid = entity2id.get(str(ent).lower())
            if nid is not None and nid in node_features:
                feats.append(node_features[int(nid)])
    if feats:
        node_feat = torch.stack(feats, dim=0).mean(dim=0)
    else:
        node_feat = torch.zeros(2, dtype=torch.float32)

    if variant == "graphtoken":
        graph_stats = torch.zeros(4, dtype=torch.float32)
        comm = torch.zeros(1, dtype=torch.float32)
    else:
        graph_stats = torch.tensor([
            float(lookup.get("graph_stats", {}).get("num_nodes", 0.0)),
            float(lookup.get("graph_stats", {}).get("num_edges", 0.0)),
            float(lookup.get("graph_stats", {}).get("avg_degree", 0.0)),
            float(lookup.get("graph_stats", {}).get("max_degree", 0.0)),
        ], dtype=torch.float32)
        comm = torch.tensor([float(record.get("community", 0))], dtype=torch.float32)
    bucket = torch.tensor([float(BUCKETS.index(record.get("bucket", BUCKETS[0])) if record.get("bucket") in BUCKETS else 0)], dtype=torch.float32)
    return torch.cat([node_feat, graph_stats, comm, bucket, q_type_onehot], dim=0)

def _graph_to_fact_docs(dataset: str, graph: Optional[Dict[str, Any]], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if graph is None:
        for rec in records:
            docs.append({
                "text": f"fact: {rec['query']} => {rec['answer']}",
                "type": "qa_fact",
                "community": int(rec.get("community", 0)),
            })
        return docs

    if dataset == "cora":
        nodes = graph["nodes"]
        for nid, node in nodes.items():
            docs.append({
                "text": f"node {nid}: title={node.get('text', '')}; class={node.get('class', '')}",
                "type": "node",
                "community": int(node.get("label") or 0),
            })
        for edge in graph["edges"]:
            docs.append({
                "text": f"edge: {edge['src']} cites {edge['tgt']}",
                "type": "edge",
                "community": 0,
            })
        return docs

    task = graph.get("task")
    if task == "node_classification":
        nodes = graph["nodes"]
        class_names = graph.get("class_names", [])
        community_map = graph.get("community_map", {})
        for nid, node in nodes.items():
            cid = int(community_map.get(nid, 0))
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            docs.append({
                "text": f"node {nid}: name={node.get('name') or node.get('title') or ''}; class={cname}",
                "type": "node",
                "community": cid,
            })
        for src, dst in graph["edges"]:
            docs.append({
                "text": f"edge: {src} linked_to {dst}",
                "type": "edge",
                "community": int(community_map.get(src, 0)),
            })
        return docs

    if task == "kg":
        rel_names = graph.get("relations", [])
        nodes = graph["nodes"]
        for src, rel, dst in graph["edges"]:
            rel_name = rel_names[int(rel)] if isinstance(rel, int) and 0 <= int(rel) < len(rel_names) else str(rel)
            src_name = nodes.get(int(src), {}).get("name", f"node_{src}")
            dst_name = nodes.get(int(dst), {}).get("name", f"node_{dst}")
            docs.append({
                "text": f"triple: <{src_name}> --{rel_name}--> <{dst_name}>",
                "type": "triple",
                "community": int(rel) if isinstance(rel, int) else 0,
            })
        return docs

    for rec in records:
        docs.append({
            "text": f"fact: {rec['query']} => {rec['answer']}",
            "type": "qa_fact",
            "community": int(rec.get("community", 0)),
        })
    return docs

def _build_bundle(dataset: str, seed: int, train_budget: int, eval_budget: int) -> DatasetBundle:
    cfg = DATASET_DEFAULTS[dataset]
    if dataset == "cora":
        records = load_phase1_records(cfg["rewritten_dir"], cfg["cora_dir"], seed=seed)
        train_records, eval_records = split_records(records, train_budget, eval_budget, seed=seed)
        graph = load_cora(cfg["cora_dir"])
    else:
        records = load_other_sft_records(
            dataset,
            sft_root=cfg["sft_root"],
            graph_root=cfg["graph_root"],
            seed=seed,
            use_v2=cfg["use_v2"],
        )
        records = [record for record in records if _record_is_reasoning(record)]
        train_records, eval_records = split_records(records, train_budget, eval_budget, seed=seed)
        graph = load_other_graph(dataset, root=cfg["graph_root"]) if dataset not in {"pubmed", "fb15k-237"} else None

    graph_stats = _graph_stats_from_graph(graph)
    feature_lookup = _build_feature_lookup(dataset, graph, records)
    feature_lookup["graph_stats"] = graph_stats

    corpus_docs = _graph_to_fact_docs(dataset, graph, train_records)

    return DatasetBundle(
        name=dataset,
        records=records,
        train_records=train_records,
        eval_records=eval_records,
        graph=graph,
        graph_stats=graph_stats,
        feature_lookup=feature_lookup,
        corpus_docs=corpus_docs,
    )

def _eval_records(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    baseline_name: str,
    graph_context: Optional[str] = None,
    retriever: Optional[SimpleRetriever] = None,
    prompt_encoder: Optional[GraphPromptEncoder] = None,
    feature_lookup: Optional[Dict[str, Any]] = None,
    context_budget: int = 8192,
    max_new_tokens: int = 64,
    iterative_retrieval: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    if prompt_encoder is not None:
        prompt_encoder.eval()

    bucket_results: Dict[str, List[Optional[bool]]] = defaultdict(list)
    fidelity_results: List[Optional[bool]] = []
    predictions: List[Dict[str, Any]] = []
    total_ooc = 0

    for rec in records:
        query = rec["query"]
        prompt = _answer_prompt(query)
        ooc = False
        retrieved_docs = None
        prefix_embeds = None

        if baseline_name == "zero_shot":
            prompt = _answer_prompt(query)

        elif baseline_name == "full_context":
            context_text = graph_context or ""
            token_count = _chunk_tokens(context_text, tokenizer)
            if context_text.endswith("[TRUNCATED]") or token_count > _available_context_tokens(context_budget):
                ooc = True
                generated = "OOC"
            else:
                prompt = _answer_prompt(query, graph_context=context_text)

        elif baseline_name == "subgraphrag":
            assert retriever is not None
            retrieved_docs = retriever.retrieve(query, top_k=50)
            if iterative_retrieval and any(key in query.lower() for key in ("multihop", "other posts", "share", "path")):
                hop_prompt = query + "\n" + "\n".join(doc["text"] for doc in retrieved_docs[:10])
                retrieved_docs = retriever.retrieve(hop_prompt, top_k=50)
            retrieved_text = "\n".join(doc["text"] for doc in retrieved_docs)
            if _chunk_tokens(retrieved_text, tokenizer) > _available_context_tokens(context_budget):
                ooc = True
                generated = "OOC"
            else:
                prompt = _answer_prompt(query, retrieved=retrieved_docs)

        elif baseline_name in {"graphtoken", "gnp"}:
            assert prompt_encoder is not None and feature_lookup is not None
            features = _entity_feature_vector(rec, feature_lookup, baseline_name).unsqueeze(0)
            device = _model_device(model)
            prefix_embeds = prompt_encoder(features.to(device)).to(device)
            prompt = _answer_prompt(query)

        if not ooc:
            generated = _generate_greedy(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                prefix_embeds=prefix_embeds,
                max_input_tokens=context_budget,
            )

        q_type = _classify(query)
        em = _check_answer(generated, rec["answer"], query)
        parsed = em is not None
        correct = bool(em) if parsed else False
        bucket = str(rec.get("bucket", "unknown"))
        bucket_results[bucket].append(correct)
        fidelity_results.append(correct if _is_structural_record(rec) else None)
        predictions.append({
            "query": query,
            "answer": rec["answer"],
            "generated": generated,
            "bucket": bucket,
            "community": int(rec.get("community", 0)),
            "correct": correct,
            "parsed": parsed,
            "ooc": ooc,
        })
        if ooc:
            total_ooc += 1

    def _pack(values: List[Optional[bool]]) -> Dict[str, Any]:
        valid = [v for v in values if v is not None]
        return {
            "acc": sum(valid) / len(valid) if valid else float("nan"),
            "n_valid": len(valid),
            "n_total": len(values),
            "pct_valid": len(valid) / len(values) if values else 0.0,
        }

    fidelity_valid = [v for v in fidelity_results if v is not None]
    result = {
        "baseline": baseline_name,
        "overall": _pack([v for vals in bucket_results.values() for v in vals]),
        "by_bucket": {b: _pack(vals) for b, vals in sorted(bucket_results.items())},
        "fidelity": {
            "acc": sum(fidelity_valid) / len(fidelity_valid) if fidelity_valid else float("nan"),
            "n_valid": len(fidelity_valid),
            "n_total": len(fidelity_results),
            "pct_valid": len(fidelity_valid) / len(fidelity_results) if fidelity_results else 0.0,
        },
        "ooc_rate": total_ooc / max(1, len(records)),
    }
    return result, predictions

def _train_graph_prompt(
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
    device = _model_device(model)
    hidden_size = model.config.hidden_size
    feature_dim = 2 + 4 + 1 + 1 + 6
    encoder = GraphPromptEncoder(feature_dim, hidden_size, prompt_len).to(device)
    train_ds = PromptTuningDataset(train_records, tokenizer, max_length, feature_lookup, variant)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_prompt_batch, num_workers=0)
    trainable = list(encoder.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = max(1, len(dl) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"[prompt] trainable params={sum(p.numel() for p in trainable):,} prompt_len={prompt_len} epochs={epochs}")
    _freeze_model(model)
    model.eval()
    encoder.train()

    for epoch in range(epochs):
        running = 0.0
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            feat_tensor = batch["features"].to(device)
            prefix = encoder(feat_tensor)
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
        print(f"[prompt] epoch {epoch + 1}/{epochs} avg_loss={running / max(1, len(dl)):.4f}")

    encoder.eval()
    return encoder

def _build_graph_context(docs: List[Dict[str, Any]], tokenizer, max_tokens: int = 4096) -> str:
    text, truncated = _serialise_docs(docs, tokenizer, max_tokens)
    if truncated:
        return text + "\n[TRUNCATED]"
    return text

def _prepare_dataset(dataset: str, seed: int, train_budget: int, eval_budget: int) -> DatasetBundle:
    bundle = _build_bundle(dataset, seed, train_budget, eval_budget)
    print(
        f"[data] {dataset}: total={len(bundle.records)} train={len(bundle.train_records)} eval={len(bundle.eval_records)} "
        f"graph_nodes={len(bundle.graph['nodes']) if bundle.graph and 'nodes' in bundle.graph else 'n/a'}"
    )
    return bundle

def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_7B)
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "wn18rr", "amazon-computers", "amazon-photo", "pubmed", "fb15k-237"])
    parser.add_argument("--baselines", nargs="+", default=["zero_shot", "full_context", "subgraphrag", "graphtoken", "gnp"])
    parser.add_argument("--train_budget", type=int, default=800)
    parser.add_argument("--eval_budget", type=int, default=150)
    parser.add_argument("--max_context_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prompt_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterative_retrieval", action="store_true")
    parser.add_argument("--output_dir", default="outputs/baselines")
    parser.add_argument("--max_datasets", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[baseline] device={device} model_path={args.model_path}")

    tokenizer, model = _build_model(args.model_path)
    all_results = []

    datasets = args.datasets[: args.max_datasets] if args.max_datasets > 0 else args.datasets
    for dataset in datasets:
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(f"Unknown dataset: {dataset}")

        bundle = _prepare_dataset(dataset, args.seed, args.train_budget, args.eval_budget)
        retriever = SimpleRetriever(bundle.corpus_docs)
        graph_context = _build_graph_context(
            bundle.corpus_docs,
            tokenizer,
            max_tokens=_available_context_tokens(args.max_context_tokens),
        )
        feature_lookup = bundle.feature_lookup

        dataset_dir = Path(args.output_dir) / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for baseline in args.baselines:
            if baseline not in {"zero_shot", "full_context", "subgraphrag", "graphtoken", "gnp"}:
                raise ValueError(f"Unknown baseline: {baseline}")

            print(f"\n[run] dataset={dataset} baseline={baseline}")
            prompt_encoder = None
            if baseline in {"graphtoken", "gnp"}:
                prompt_encoder = _train_graph_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    train_records=bundle.train_records,
                    feature_lookup=feature_lookup,
                    variant=baseline,
                    prompt_len=args.prompt_len if baseline == "graphtoken" else args.prompt_len * 2,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    max_length=256,
                )
                prompt_encoder = prompt_encoder.to(_model_device(model))

            result, predictions = _eval_records(
                model=model,
                tokenizer=tokenizer,
                records=bundle.eval_records,
                baseline_name=baseline,
                graph_context=graph_context if baseline == "full_context" else None,
                retriever=retriever if baseline == "subgraphrag" else None,
                prompt_encoder=prompt_encoder,
                feature_lookup=feature_lookup if baseline in {"graphtoken", "gnp"} else None,
                context_budget=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                iterative_retrieval=args.iterative_retrieval,
            )

            out_dir = dataset_dir / baseline
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_json(out_dir / "results.json", {
                "dataset": dataset,
                "baseline": baseline,
                "train_budget": args.train_budget,
                "eval_budget": args.eval_budget,
                "max_context_tokens": args.max_context_tokens,
                "max_new_tokens": args.max_new_tokens,
                "prompt_len": args.prompt_len,
                "result": result,
            })
            with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
                for item in predictions:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(
                f"[done] {dataset}/{baseline} overall={result['overall']['acc']:.3f} "
                f"fidelity={result['fidelity']['acc']:.3f} ooc={result['ooc_rate']:.2%}"
            )
            all_results.append({"dataset": dataset, "baseline": baseline, **result})

    summary_path = Path(args.output_dir) / "summary.json"
    _save_json(summary_path, {"results": all_results})
    print(f"[baseline] summary -> {summary_path}")

if __name__ == "__main__":
    main()