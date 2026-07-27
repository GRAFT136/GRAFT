
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graft.data import load_synthetic, load_cora
from graft.data.synthetic_graph import generate_sbm
from graft.model import inject_moe_lora, get_moe_layers, StructureGroundedRouter, get_hidden_size
from graft.train import run_stage1_preprocessing, run_stage2_warmup, run_stage3_joint
from graft.eval import RouterMonitor, GraftInference

def load_graph(config: dict):
    graph_cfg = config["graph"]
    if graph_cfg["type"] == "synthetic_sbm":
        out_path = graph_cfg.get("output_path", "data/synthetic_sbm.json")
        if not os.path.exists(out_path):
            generate_sbm(
                num_communities=graph_cfg["num_communities"],
                nodes_per_community=graph_cfg["nodes_per_community"],
                p_in=graph_cfg["p_in"],
                p_out=graph_cfg["p_out"],
                bridge_edges=graph_cfg["bridge_edges"],
                seed=graph_cfg["seed"],
                output_path=out_path,
            )
        return load_synthetic(out_path)
    elif graph_cfg["type"] == "cora":
        return load_cora(graph_cfg["data_dir"])
    raise ValueError(f"Unknown graph.type: {graph_cfg['type']!r}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device
    torch.manual_seed(config["train"].get("seed", 42))

    print(f"[pipeline] loading base model: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"]).to(device)

    moe_cfg = config["moe_lora"]
    model = inject_moe_lora(
        model,
        rank=moe_cfg["rank"],
        lora_alpha=moe_cfg["alpha"],
        num_communities=moe_cfg["num_communities"],
        use_global_expert=moe_cfg["use_global_expert"],
        target_modules=moe_cfg["target_modules"],
    )
    moe_layers = get_moe_layers(model)

    router = StructureGroundedRouter(
        hidden_size=get_hidden_size(model),
        community_dim=config["gnn"]["out_dim"],
        routing_dim=moe_cfg["router"]["routing_dim"],
        top_kappa=moe_cfg["router"]["top_kappa"],
    ).to(device)

    graph_data = load_graph(config)
    part_cfg = config["partition"]
    node_class_map = None
    ground_truth_map = graph_data.get("community_map") if part_cfg["method"] == "ground_truth" else None
    if part_cfg["method"] == "topic_class":
        node_class_map = {nid: attrs.get("class", "") for nid, attrs in graph_data["nodes"].items()}

    print("[pipeline] Stage 1: Graph Preprocessing")
    stage1 = run_stage1_preprocessing(
        graph_data,
        model,
        tokenizer,
        partition_method=part_cfg["method"],
        min_community_size=part_cfg.get("min_community_size", 10),
        node_class_map=node_class_map,
        ground_truth_map=ground_truth_map,
        d_max=config["nnp"]["d_max"],
        gnn_hidden_dim=config["gnn"]["hidden_dim"],
        gnn_out_dim=config["gnn"]["out_dim"],
        gnn_num_layers=config["gnn"]["num_layers"],
        gnn_pretrain_epochs=config["gnn"]["pretrain_epochs"],
        gnn_lr=config["gnn"]["lr"],
        seed=config["train"].get("seed", 42),
        device=device,
    )

    print("[pipeline] Stage 2: Expert Warmup")
    warmup_cfg = config["train"]["warmup"]
    run_stage2_warmup(
        model,
        tokenizer,
        moe_layers,
        stage1.nnp_instances,
        stage1.nnp_instances_per_community,
        device=device,
        global_warmup_epochs=warmup_cfg["global_epochs"],
        expert_warmup_epochs=warmup_cfg["expert_epochs"],
        batch_size=warmup_cfg["batch_size"],
        lr=warmup_cfg["lr"],
        max_length=config["train"]["max_length"],
        grad_clip=config["train"]["grad_clip"],
    )

    print("[pipeline] Stage 3: Joint Optimization")
    joint_cfg = config["train"]["joint"]
    monitor = RouterMonitor(num_communities=stage1.num_communities)
    run_stage3_joint(
        model,
        router,
        moe_layers,
        stage1.gnn,
        stage1.h0,
        stage1.a_hat,
        stage1.communities,
        stage1.node_index,
        tokenizer,
        stage1.nnp_instances,
        device=device,
        num_epochs=joint_cfg["epochs"],
        batch_size=joint_cfg["batch_size"],
        lr=joint_cfg["lr"],
        max_length=config["train"]["max_length"],
        alpha=joint_cfg["alpha"],
        beta=joint_cfg["beta"],
        top_kappa=moe_cfg["router"]["top_kappa"],
        grad_clip=config["train"]["grad_clip"],
        monitor=monitor,
    )

    output_dir = config["train"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "graft_checkpoint.pt")
    torch.save(
        {
            "moe_state_dict": {n: p for n, p in model.state_dict().items() if "lora_" in n},
            "router_state_dict": router.state_dict(),
            "gnn_state_dict": stage1.gnn.state_dict(),
            "community_embeddings": stage1.community_embeddings,
            "config": config,
        },
        ckpt_path,
    )
    print(f"[pipeline] Saved checkpoint to {ckpt_path}")

    from graft.model import GNNEncoder

    with torch.no_grad():
        z = stage1.gnn(stage1.h0, stage1.a_hat)
        community_embeddings = GNNEncoder.community_embeddings(z, stage1.communities, stage1.node_index)

    inference = GraftInference(model, router, community_embeddings, tokenizer, device=device)
    sample_query = (
        stage1.nnp_instances[0].prompt
        if stage1.nnp_instances
        else "Given the entity [example], list its directly connected neighbors."
    )
    print("[pipeline] sample inference query:", sample_query)
    print("[pipeline] sample generation:", inference.generate(sample_query, max_new_tokens=32))

if __name__ == "__main__":
    main()
