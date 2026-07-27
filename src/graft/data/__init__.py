
from .graph_loader import load_cora, load_synthetic
from .partition import detect_communities
from .nnp_dataset import (
    NNPInstance,
    build_nnp_instances,
    build_nnp_instances_for_communities,
    nnp_prompt,
)

__all__ = [
    "load_cora",
    "load_synthetic",
    "detect_communities",
    "NNPInstance",
    "build_nnp_instances",
    "build_nnp_instances_for_communities",
    "nnp_prompt",
]
