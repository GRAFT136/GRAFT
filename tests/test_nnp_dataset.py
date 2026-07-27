
from __future__ import annotations

import math

import networkx as nx
import pytest

from graft.data.nnp_dataset import (
    PROMPT_TEMPLATE,
    build_nnp_instances,
    build_nnp_instances_for_communities,
    nnp_prompt,
)
from graft.train.dataset import _mask_prompt

def _small_graph():
    G = nx.DiGraph()
    nodes = {
        0: {"text": "paper A"},
        1: {"text": "paper B"},
        2: {"text": "paper C"},
        3: {"text": "paper D"},
    }
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)
    G.add_edge(0, 1)
    G.add_edge(2, 0)
    G.add_edge(0, 2)
    return nodes, G

def test_nnp_prompt_matches_template():
    text = "paper A"
    assert nnp_prompt(text) == PROMPT_TEMPLATE.format(text=text)
    assert nnp_prompt(text) == "Given the entity [paper A], list its directly connected neighbors."

def test_isolated_node_gets_explicit_empty_target():
    nodes, G = _small_graph()
    instances = build_nnp_instances(nodes, G, seed=0)
    by_node = {inst.node_id: inst for inst in instances if inst.node_id == 3}
    assert 3 in by_node
    assert by_node[3].target == "(no directly connected neighbors)"
    assert by_node[3].neighbor_ids == []

def test_neighbor_set_is_undirected_union_and_deduped():
    nodes, G = _small_graph()
    instances = build_nnp_instances(nodes, G, seed=0)
    node0 = [inst for inst in instances if inst.node_id == 0]
    assert len(node0) == 1
    inst = node0[0]
    assert sorted(inst.neighbor_ids) == [1, 2]
    assert set(inst.target.split(" | ")) == {"paper B", "paper C"}

def test_permutation_is_seed_reproducible():
    nodes, G = _small_graph()
    a = build_nnp_instances(nodes, G, seed=7)
    b = build_nnp_instances(nodes, G, seed=7)
    c = build_nnp_instances(nodes, G, seed=8)
    targets_a = [i.target for i in a]
    targets_b = [i.target for i in b]
    assert targets_a == targets_b
    assert isinstance(c, list)

def test_subsampling_produces_ceil_d_over_dmax_instances():
    G = nx.DiGraph()
    nodes = {0: {"text": "hub"}}
    for i in range(1, 11):
        nodes[i] = {"text": f"n{i}"}
        G.add_edge(0, i)
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)

    instances = build_nnp_instances(nodes, G, d_max=3, seed=0)
    hub_instances = [i for i in instances if i.node_id == 0]
    assert len(hub_instances) == math.ceil(10 / 3)
    for m, inst in enumerate(hub_instances):
        assert inst.subsample_index == m
        assert inst.num_subsamples == len(hub_instances)
        assert len(inst.neighbor_ids) == 3

def test_build_nnp_instances_for_communities_restricts_by_membership():
    nodes, G = _small_graph()
    node_to_community = {0: 0, 1: 0, 2: 1, 3: 1}
    communities = [[0, 1], [2, 3]]
    per_comm = build_nnp_instances_for_communities(nodes, G, node_to_community, communities, seed=0)
    assert set(per_comm.keys()) == {0, 1}
    ids_in_comm0 = {inst.node_id for inst in per_comm[0]}
    ids_in_comm1 = {inst.node_id for inst in per_comm[1]}
    assert ids_in_comm0 == {0, 1}
    assert ids_in_comm1 == {2, 3}

def test_as_dict_keys():
    nodes, G = _small_graph()
    instances = build_nnp_instances(nodes, G, seed=0, node_to_community={0: 5})
    inst0 = [i for i in instances if i.node_id == 0][0]
    d = inst0.as_dict()
    assert set(d.keys()) == {
        "node_id",
        "query",
        "answer",
        "neighbor_ids",
        "community",
        "subsample_index",
        "num_subsamples",
    }
    assert d["community"] == 5

@pytest.mark.parametrize(
    "labels_text,marker_text",
    [
        ("A B C assistant D E", "assistant"),
    ],
)
def test_mask_prompt_masks_everything_up_to_and_including_marker(labels_text, marker_text):
    vocab = {w: i for i, w in enumerate(set(labels_text.split()))}
    ids = [vocab[w] for w in labels_text.split()]
    marker_ids = [vocab[w] for w in marker_text.split()]

    import torch

    labels = torch.tensor(ids, dtype=torch.long)
    masked = _mask_prompt(labels.clone(), marker_ids)

    marker_end = ids.index(marker_ids[0]) + len(marker_ids)
    assert (masked[:marker_end] == -100).all()
    assert (masked[marker_end:] == labels[marker_end:]).all()
