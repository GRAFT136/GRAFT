
from __future__ import annotations

import torch

from graft.model.moe_lora import CommunityMoELoraLinear

def _make_layer(in_f=8, out_f=6, rank=4, num_communities=3, use_global_expert=True, seed=0):
    torch.manual_seed(seed)
    base_weight = torch.randn(out_f, in_f)
    base_bias = torch.randn(out_f)
    layer = CommunityMoELoraLinear(
        in_features=in_f,
        out_features=out_f,
        base_weight=base_weight,
        base_bias=base_bias,
        rank=rank,
        lora_alpha=2 * rank,
        num_communities=num_communities,
        use_global_expert=use_global_expert,
    )
    return layer, base_weight, base_bias

def test_base_weight_and_bias_are_frozen():
    layer, base_weight, base_bias = _make_layer()
    assert layer.base_weight.requires_grad is False
    assert layer.base_bias.requires_grad is False
    assert torch.allclose(layer.base_weight, base_weight)
    assert torch.allclose(layer.base_bias, base_bias)

def test_lora_params_are_trainable_by_default():
    layer, *_ = _make_layer()
    assert layer.lora_A_global.requires_grad
    assert layer.lora_B_global.requires_grad
    assert layer.lora_A_local.requires_grad
    assert layer.lora_B_local.requires_grad

def test_forward_with_no_routing_equals_base_plus_zero_init_global():
    layer, base_weight, base_bias = _make_layer()
    x = torch.randn(3, 8)
    out = layer(x)
    expected = torch.nn.functional.linear(x, base_weight, base_bias)
    assert torch.allclose(out, expected, atol=1e-5)

def test_global_expert_changes_output_once_nonzero():
    layer, base_weight, base_bias = _make_layer()
    x = torch.randn(3, 8)
    base_out = layer(x)

    with torch.no_grad():
        layer.lora_B_global.normal_(mean=0.0, std=1.0)
    new_out = layer(x)
    assert not torch.allclose(base_out, new_out, atol=1e-4)

    layer.zero_global_expert()
    restored = layer(x)
    assert torch.allclose(restored, base_out, atol=1e-5)

def test_forced_single_expert_routing_only_activates_selected_expert():
    layer, base_weight, base_bias = _make_layer(use_global_expert=False, num_communities=3)
    x = torch.randn(4, 8)

    with torch.no_grad():
        layer.lora_B_local.normal_(mean=0.0, std=1.0)

    weights = torch.ones(4, 1)
    idx0 = torch.zeros(4, 1, dtype=torch.long)
    idx1 = torch.ones(4, 1, dtype=torch.long)

    out_expert0 = layer(x, router_weights=weights, router_indices=idx0)
    out_expert1 = layer(x, router_weights=weights, router_indices=idx1)
    out_base = layer(x)

    assert not torch.allclose(out_expert0, out_base, atol=1e-4)
    assert not torch.allclose(out_expert1, out_base, atol=1e-4)
    assert not torch.allclose(out_expert0, out_expert1, atol=1e-4)

    layer.zero_local_expert(0)
    out_expert0_zeroed = layer(x, router_weights=weights, router_indices=idx0)
    assert torch.allclose(out_expert0_zeroed, out_base, atol=1e-5)

def test_top_kappa_multi_expert_composition_matches_manual_sum():
    layer, base_weight, base_bias = _make_layer(use_global_expert=True, num_communities=4, rank=4)
    with torch.no_grad():
        layer.lora_B_global.normal_(std=0.5)
        layer.lora_B_local.normal_(std=0.5)

    x = torch.randn(2, 8)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    indices = torch.tensor([[0, 2], [1, 3]])

    out = layer(x, router_weights=weights, router_indices=indices)

    x32 = x.to(torch.float32)
    h = torch.nn.functional.linear(x, base_weight, base_bias)
    g_out = (x32 @ layer.lora_A_global.T) @ layer.lora_B_global.T
    h = h + g_out * layer.scaling
    for b in range(2):
        for k_slot in range(2):
            k = indices[b, k_slot].item()
            g = weights[b, k_slot].item()
            A = layer.lora_A_local[k]
            B = layer.lora_B_local[k]
            delta = (x32[b] @ A.T) @ B.T * layer.scaling
            h[b] = h[b] + g * delta

    assert torch.allclose(out, h, atol=1e-4)

def test_zero_all_local_experts_disables_all_community_deltas():
    layer, base_weight, base_bias = _make_layer(use_global_expert=False, num_communities=3)
    with torch.no_grad():
        layer.lora_B_local.normal_(std=1.0)
    layer.zero_all_local_experts()

    x = torch.randn(4, 8)
    weights = torch.ones(4, 2)
    indices = torch.tensor([[0, 1]] * 4)
    out = layer(x, router_weights=weights, router_indices=indices)
    expected = torch.nn.functional.linear(x, base_weight, base_bias)
    assert torch.allclose(out, expected, atol=1e-5)

def test_gradients_only_flow_to_lora_params_not_base():
    layer, *_ = _make_layer()
    x = torch.randn(3, 8, requires_grad=True)
    weights = torch.ones(3, 1)
    indices = torch.zeros(3, 1, dtype=torch.long)
    out = layer(x, router_weights=weights, router_indices=indices)
    out.sum().backward()

    assert layer.base_weight.grad is None
    assert layer.base_bias.grad is None
    assert layer.lora_A_local.grad is not None
    assert layer.lora_B_local.grad is not None
    assert layer.lora_A_global.grad is not None
    assert layer.lora_B_global.grad is not None

def test_forward_preserves_leading_batch_shape():
    layer, *_ = _make_layer(in_f=8, out_f=6)
    x = torch.randn(2, 5, 8)
    out = layer(x)
    assert out.shape == (2, 5, 6)
