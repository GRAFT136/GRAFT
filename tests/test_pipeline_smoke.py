
from __future__ import annotations

import torch

from graft.data.synthetic_graph import generate_sbm
from graft.eval.inference import GraftInference
from graft.eval.monitor import RouterMonitor
from graft.model.gnn_encoder import GNNEncoder
from graft.model.injection import get_hidden_size, get_moe_layers, inject_moe_lora
from graft.model.router import StructureGroundedRouter
from graft.train import run_stage1_preprocessing, run_stage2_warmup, run_stage3_joint

def test_full_pipeline_smoke(tiny_model_factory, tiny_tokenizer):
    graph_data = generate_sbm(
        num_communities=3,
        nodes_per_community=5,
        p_in=0.6,
        p_out=0.05,
        bridge_edges=4,
        seed=1,
    )

    model = tiny_model_factory(hidden_size=32, num_hidden_layers=2)
    inject_moe_lora(model, rank=4, lora_alpha=8, num_communities=3, use_global_expert=True)
    moe_layers = get_moe_layers(model)
    assert len(moe_layers) > 0

    stage1 = run_stage1_preprocessing(
        graph_data,
        model,
        tiny_tokenizer,
        partition_method="ground_truth",
        min_community_size=1,
        ground_truth_map=graph_data["community_map"],
        d_max=32,
        gnn_hidden_dim=16,
        gnn_out_dim=16,
        gnn_num_layers=2,
        gnn_pretrain_epochs=2,
        device="cpu",
    )
    assert stage1.num_communities == 3
    assert stage1.community_embeddings.shape == (3, 16)
    assert len(stage1.nnp_instances) > 0
    assert isinstance(stage1.gnn, GNNEncoder)

    router = StructureGroundedRouter(
        hidden_size=get_hidden_size(model),
        community_dim=16,
        routing_dim=16,
        top_kappa=2,
    )

    run_stage2_warmup(
        model,
        tiny_tokenizer,
        moe_layers,
        stage1.nnp_instances,
        stage1.nnp_instances_per_community,
        device="cpu",
        global_warmup_epochs=1,
        expert_warmup_epochs=1,
        batch_size=2,
        max_length=64,
    )
    for layer in moe_layers:
        assert layer.lora_A_local.requires_grad
        assert layer.lora_A_global.requires_grad

    monitor = RouterMonitor(num_communities=3, window=10)
    run_stage3_joint(
        model,
        router,
        moe_layers,
        stage1.gnn,
        stage1.h0,
        stage1.a_hat,
        stage1.communities,
        stage1.node_index,
        tiny_tokenizer,
        stage1.nnp_instances,
        device="cpu",
        num_epochs=1,
        batch_size=2,
        max_length=64,
        alpha=1.0,
        beta=0.01,
        top_kappa=2,
        monitor=monitor,
    )

    with torch.no_grad():
        z = stage1.gnn(stage1.h0, stage1.a_hat)
        final_community_embeddings = GNNEncoder.community_embeddings(
            z, stage1.communities, stage1.node_index
        )

    inference = GraftInference(
        model,
        router,
        final_community_embeddings,
        tiny_tokenizer,
        device="cpu",
    )
    sample_query = stage1.nnp_instances[0].prompt
    output_text = inference.generate(sample_query, max_new_tokens=8, do_sample=False)
    assert isinstance(output_text, str)
    assert len(output_text) > 0
