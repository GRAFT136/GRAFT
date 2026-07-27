# GRAFT — Scalable Graph Internalization for Large Language Models

## Abstract

Internalizing graph knowledge directly into language model parameters eliminates runtime dependence on explicit graph context, yet existing methods suffer from two fundamental limitations. First, they rely on hand-crafted rules to translate graph structures into question-answer training signals, causing the model to absorb only a lossy, rule-constrained projection of the graph, a problem we term *semantic distortion*. Second, their use of a single low-rank adapter creates a *capacity saturation* bottleneck that prevents faithful encoding of large-scale graphs. We propose GRAFT, a framework that addresses both limitations. First, **Next-Neighbor Prediction** trains the model to directly reproduce each node's neighbors from the raw edge set, bypassing all rule-based intermediaries and preserving the graph's complete relational information. Second, **Topology-Aware Routing** partitions the graph into communities, assigns each to a dedicated LoRA expert alongside a global expert for cross-community knowledge, and routes queries via a structure-grounded mechanism that matches query embeddings against GNN-derived community embeddings, allowing parametric capacity to scale linearly with graph size while maintaining topology-informed expert composition. Extensive experiments on 9 real-world graphs and a synthetic suite spanning three orders of magnitude in size demonstrate that GRAFT consistently outperforms context-based, GNN-adapted, and existing in-parameter baselines, achieving neighbor-set F1 above 0.76 and up to 5.9-point gains on graph reasoning, with especially large advantages on graphs containing up to 1 million edges.

## Environment setup

Use the conda environment named **`GRAFT`**:

```
conda activate GRAFT
pip install -r requirements.txt
pip install -e .
```

Verified working versions: Python 3.12, `torch` 2.9 (GPU), `transformers`
4.49, `networkx` 3.3. Notably **`peft` and `python-louvain` are NOT
required** — the MoE-LoRA layers are implemented from scratch and community
detection uses networkx's built-in `louvain_communities` (see
`requirements.txt` for the full rationale).

## Running the pipeline

```powershell
python scripts/run_graft_pipeline.py --config configs/phase0_synthetic.yaml
```

This downloads/generates a small synthetic SBM graph, runs all 3 stages on
`meta-llama/Meta-Llama-3.1-8B-Instruct`, saves a checkpoint, and prints a demo
generation. `configs/phase1_graft.yaml` targets the full Cora pipeline on
`meta-llama/Meta-Llama-3.1-8B-Instruct` — it requires the raw Cora dataset
(`all.csv`/`edges.csv`) under `data/cora_dataset/`, which is **not included**
in this repository (see the comment at the top of that config file).

## Running tests

```powershell
python -m pytest tests -q
```

The suite validates every core component — NNP construction, the
GNN encoder, the structure-grounded router, `CommunityMoELoraLinear`
composition math, the losses, MoE-LoRA injection into a real
`Llama3ForCausalLM`, and a full 3-stage pipeline + inference test.
