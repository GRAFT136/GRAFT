
from __future__ import annotations

import torch

from graft.model.injection import RouterDecisionContext, get_hidden_size, get_moe_layers, inject_moe_lora
from graft.model.moe_lora import CommunityMoELoraLinear
from graft.model.router import StructureGroundedRouter

def test_inject_moe_lora_replaces_mlp_projections_only(tiny_model_factory):
    model = tiny_model_factory(hidden_size=32, num_hidden_layers=2)
    num_layers = model.config.num_hidden_layers
    inject_moe_lora(model, rank=4, lora_alpha=8, num_communities=3, use_global_expert=True)

    moe_layers = get_moe_layers(model)
    assert len(moe_layers) == num_layers * 3
    assert all(isinstance(m, CommunityMoELoraLinear) for m in moe_layers)

    attn_linears = [
        m for name, m in model.named_modules()
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
    ]
    assert len(attn_linears) > 0
    assert all(not isinstance(m, CommunityMoELoraLinear) for m in attn_linears)

def test_inject_moe_lora_freezes_base_and_leaves_lora_trainable(tiny_model_factory):
    model = tiny_model_factory()
    inject_moe_lora(model, rank=4, num_communities=2)

    moe_layers = get_moe_layers(model)
    for layer in moe_layers:
        assert layer.base_weight.requires_grad is False
        assert layer.lora_A_local.requires_grad is True
        assert layer.lora_B_local.requires_grad is True

    non_lora_params = [
        p for n, p in model.named_parameters() if "lora_" not in n
    ]
    assert all(not p.requires_grad for p in non_lora_params)

def test_model_forward_runs_after_injection(tiny_model_factory, tiny_tokenizer):
    model = tiny_model_factory()
    inject_moe_lora(model, rank=4, num_communities=2)

    enc = tiny_tokenizer(["hello graph world"], return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    assert out.logits.shape[0] == 1
    assert out.logits.shape[-1] == model.config.vocab_size
    assert torch.isfinite(out.logits).all()

def test_router_decision_context_changes_output_when_experts_nonzero(tiny_model_factory, tiny_tokenizer):
    model = tiny_model_factory()
    inject_moe_lora(model, rank=4, num_communities=3, use_global_expert=True)
    moe_layers = get_moe_layers(model)

    with torch.no_grad():
        for layer in moe_layers:
            layer.lora_B_local[1].normal_(std=1.0)

    enc = tiny_tokenizer(["hello graph world"], return_tensors="pt")

    with torch.no_grad():
        out_no_route = model(**enc).logits

        weights = torch.ones(1, 1)
        indices_expert1 = torch.tensor([[1]])
        with RouterDecisionContext(moe_layers, weights, indices_expert1):
            out_routed = model(**enc).logits

        out_after_context = model(**enc).logits

    assert not torch.allclose(out_no_route, out_routed, atol=1e-4)
    assert torch.allclose(out_no_route, out_after_context, atol=1e-5)

def test_get_hidden_size_matches_config(tiny_model_factory):
    model = tiny_model_factory(hidden_size=32)
    assert get_hidden_size(model) == 32

def test_router_end_to_end_with_hidden_states(tiny_model_factory, tiny_tokenizer):
    model = tiny_model_factory()
    inject_moe_lora(model, rank=4, num_communities=3)
    hidden_size = get_hidden_size(model)
    router = StructureGroundedRouter(hidden_size=hidden_size, community_dim=6, routing_dim=8, top_kappa=2)
    community_embeddings = torch.randn(3, 6)

    enc = tiny_tokenizer(["a query about the graph"], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
        mid = len(outputs.hidden_states) // 2
        u = outputs.hidden_states[mid][:, -1, :]
        weights, indices, scores = router(u, community_embeddings)

    assert weights.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert scores.shape == (1, 3)
