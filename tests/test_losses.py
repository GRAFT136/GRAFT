
from __future__ import annotations

import pytest
import torch

from graft.model.losses import compute_total_loss, load_balancing_loss, routing_supervision_loss

def test_routing_supervision_loss_is_zero_for_perfect_confident_logits():
    B, K = 4, 3
    labels = torch.tensor([0, 1, 2, 0])
    scores = torch.full((B, K), -100.0)
    for i, lbl in enumerate(labels):
        scores[i, lbl] = 100.0
    loss = routing_supervision_loss(scores, labels)
    assert loss.item() < 1e-3

def test_routing_supervision_loss_matches_manual_cross_entropy():
    scores = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    labels = torch.tensor([0, 2])
    expected = torch.nn.functional.cross_entropy(scores, labels)
    got = routing_supervision_loss(scores, labels)
    assert torch.allclose(got, expected)

def test_load_balancing_loss_minimal_when_uniform():
    K = 4
    B = K
    scores = torch.zeros(B, K)
    loss = load_balancing_loss(scores, num_communities=K, top_kappa=1)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)

def test_load_balancing_loss_higher_when_collapsed_to_one_expert():
    K = 4
    B = 8
    scores = torch.full((B, K), -10.0)
    scores[:, 0] = 10.0
    collapsed = load_balancing_loss(scores, num_communities=K, top_kappa=1).item()

    uniform_scores = torch.zeros(B, K)
    uniform = load_balancing_loss(uniform_scores, num_communities=K, top_kappa=1).item()

    assert collapsed > uniform

def test_compute_total_loss_falls_back_to_nnp_loss_only_when_scores_none():
    nnp_loss = torch.tensor(2.5)
    total = compute_total_loss(nnp_loss, None, None, num_communities=4, top_kappa=2)
    assert torch.equal(total, nnp_loss)

def test_compute_total_loss_skips_route_term_without_labels():
    nnp_loss = torch.tensor(1.0)
    scores = torch.randn(3, 4)
    total_no_labels = compute_total_loss(nnp_loss, scores, None, num_communities=4, top_kappa=2, alpha=1.0, beta=0.0)
    bal = load_balancing_loss(scores, 4, 2)
    assert torch.allclose(total_no_labels, nnp_loss + 0.0 * bal)

def test_compute_total_loss_includes_route_and_balance_terms():
    nnp_loss = torch.tensor(1.0)
    scores = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])
    alpha, beta = 0.5, 0.1
    total = compute_total_loss(nnp_loss, scores, labels, num_communities=4, top_kappa=2, alpha=alpha, beta=beta)
    expected = (
        nnp_loss
        + beta * load_balancing_loss(scores, 4, 2)
        + alpha * routing_supervision_loss(scores, labels)
    )
    assert torch.allclose(total, expected)
