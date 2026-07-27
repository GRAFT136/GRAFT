
from __future__ import annotations

import networkx as nx
import torch

from graft.model.gnn_encoder import (
    GNNEncoder,
    GNNLayer,
    build_normalized_adjacency,
    init_node_features_from_llm,
)
from graft.model.router import StructureGroundedRouter

def _line_graph(n=5):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    node_index = {i: i for i in range(n)}
    return G, node_index

def test_build_normalized_adjacency_is_symmetric_and_self_looped():
    G, node_index = _line_graph(5)
    a_hat = build_normalized_adjacency(G, num_nodes=5, node_index=node_index)
    assert a_hat.shape == (5, 5)
    assert torch.allclose(a_hat, a_hat.T, atol=1e-6)
    assert torch.all(torch.diagonal(a_hat) > 0)

def test_gnn_layer_forward_shape_and_gradient_flow():
    layer = GNNLayer(in_dim=8, out_dim=6)
    h = torch.randn(5, 8, requires_grad=True)
    a_hat = torch.eye(5)
    out = layer(h, a_hat)
    assert out.shape == (5, 6)
    out.sum().backward()
    assert h.grad is not None
    assert layer.linear.weight.grad is not None

def test_gnn_encoder_forward_shape():
    gnn = GNNEncoder(in_dim=8, hidden_dim=16, out_dim=4, num_layers=2)
    G, node_index = _line_graph(6)
    a_hat = build_normalized_adjacency(G, num_nodes=6, node_index=node_index)
    h0 = torch.randn(6, 8)
    z = gnn(h0, a_hat)
    assert z.shape == (6, 4)

def test_gnn_encoder_community_embeddings_mean_pools_members():
    node_embeddings = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 6.0]]
    )
    node_index = {10: 0, 11: 1, 20: 2, 21: 3}
    communities = [[10, 11], [20, 21]]
    c = GNNEncoder.community_embeddings(node_embeddings, communities, node_index)
    assert c.shape == (2, 2)
    assert torch.allclose(c[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(c[1], torch.tensor([0.0, 4.0]))

def test_init_node_features_from_llm_shape(tiny_model, tiny_tokenizer):
    embedding_layer = tiny_model.get_input_embeddings()
    texts = ["paper about neural nets", "paper about bayesian priors", "isolated stub"]
    h0 = init_node_features_from_llm(texts, embedding_layer, tiny_tokenizer, device=torch.device("cpu"))
    assert h0.shape == (3, tiny_model.config.hidden_size)
    assert torch.isfinite(h0).all()

def test_router_scores_shape_and_topk_selection():
    router = StructureGroundedRouter(hidden_size=16, community_dim=8, routing_dim=8, top_kappa=2)
    u = torch.randn(5, 16)
    community_embeddings = torch.randn(4, 8)
    scores = router.scores(u, community_embeddings)
    assert scores.shape == (5, 4)

    weights, indices, all_scores = router(u, community_embeddings)
    assert weights.shape == (5, 2)
    assert indices.shape == (5, 2)
    assert torch.equal(all_scores, scores)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-5)
    top2_expected = scores.topk(2, dim=-1).indices
    assert torch.equal(indices.sort(dim=-1).values, top2_expected.sort(dim=-1).values)

def test_router_top_kappa_capped_at_num_communities():
    router = StructureGroundedRouter(hidden_size=8, community_dim=4, routing_dim=4, top_kappa=10)
    u = torch.randn(2, 8)
    community_embeddings = torch.randn(3, 4)
    weights, indices, scores = router(u, community_embeddings)
    assert weights.shape == (2, 3)
    assert indices.shape == (2, 3)

def test_router_gradients_flow_to_projections():
    router = StructureGroundedRouter(hidden_size=8, community_dim=4, routing_dim=4, top_kappa=2)
    u = torch.randn(3, 8)
    community_embeddings = torch.randn(4, 4)
    weights, indices, scores = router(u, community_embeddings)
    scores.sum().backward()
    assert router.W_q.weight.grad is not None
    assert router.W_c.weight.grad is not None
