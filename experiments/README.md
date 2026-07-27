# experiments/ — archival reference scripts (not part of the core GRAFT package)

Everything under this folder is copied from the original `graph-moe-lora/`
and `Cora/` directories for historical reference (only anonymized — see
below — otherwise unmodified). None of it has been rewritten to match the
paper or to import from the new `graft` package, and none of it is required
for (or exercised by) the tests in `fixed_project/tests/`. Per the cleanup
scope, only the core pipeline (`src/graft/`, `scripts/run_graft_pipeline.py`,
`configs/`, `tests/`) was verified to run and to align with
`methodology.tex`.

If you want to reuse any of these scripts going forward, treat them as a
starting point only — they use the *old* module names (`GlobalLocalLoraLinear`,
`SharedGlobalRouter`, `MoEForwardContext`, the "GRIP-style" 2-stage trainer,
hardcoded local Qwen2.5 snapshot paths, etc.), which the core package has
since replaced/renamed.

**Anonymization note**: hardcoded absolute paths that referenced the
original author's cluster username/home directory and conda environment
name have been replaced with generic placeholders (`/home/USER/...`,
`graft-env`) so this folder can be shared alongside an anonymous submission.

## legacy_rule_based_qa/
Rule-based Cora QA generators (existence / counting / traversal / substructure
/ multi-hop). The paper's Next-Neighbor Prediction (NNP) objective
(`src/graft/data/nnp_dataset.py`) is explicitly designed to *replace* this
kind of hand-crafted, rule-derived QA — methodology.tex frames NNP as
requiring "no rule design, no derived-property computation, and no
LLM-based augmentation." These generators therefore contradict that framing
and are kept only as a historical artifact / possible baseline for an
ablation such as "rule-based QA pretraining vs. raw NNP."

## baselines_and_ablations/
Scripts that formed the closest thing to a "core" training pipeline in the
old codebase, plus architectural ablations:
- `phase0_train.py`, `phase1_train.py`: old 2-stage trainer entry points
  (synthetic graph / real Cora) — superseded by
  `scripts/run_graft_pipeline.py` + `src/graft/train/` (now a genuine 3-stage
  pipeline with a Graph Preprocessing stage and a structure-grounded router).
- `phase1_single_lora.py`: single-LoRA (no MoE) baseline for comparison.
- `phase2_dense_train.py`: dense (non-sparse) soft-routing ablation.
- `token_moe_experiment.py`: token-level (rather than sequence-level) routing
  ablation.
- `topo_aware_moe.py`, `cora_topo_moe.py`: an earlier topology-aware router
  design using Laplacian positional encodings + FiLM conditioning, distinct
  from the dot-product `StructureGroundedRouter` the paper specifies.
- `baseline_benchmark.py`, `eval_phase1.py`: comparison baselines / post-hoc
  evaluation harnesses.

## sweeps_and_debug/
Parameter sweeps, capacity-scaling experiments, and one-off
debugging/smoke-test scripts (`capacity_scaling_sweep.py`, `scale_sweep.py`,
`real_scale_sweep.py`, `matched_param_experiment.py`, `fair_moe_test.py`,
`community_label_test.py`, `node_code_test.py`, `stack_elec_experiment.py`,
`nnp_train_pool_sweep.py`, `nnp_benchmark.py`,
`merge_capacity_scaling_results.py`). Useful for reproducing specific
experiments from the project's research history, not part of the paper's
core methodology.
