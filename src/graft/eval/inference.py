
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from ..model.injection import RouterDecisionContext, get_moe_layers
from ..model.router import StructureGroundedRouter

@dataclass
class RoutingDecision:
    weights: torch.Tensor
    indices: torch.Tensor
    scores: torch.Tensor

class GraftInference:

    def __init__(
        self,
        model,
        router: StructureGroundedRouter,
        community_embeddings: torch.Tensor,
        tokenizer,
        routing_layer_index: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.router = router.to(device)
        self.community_embeddings = community_embeddings.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.routing_layer_index = routing_layer_index
        self.moe_layers = get_moe_layers(model)

    @torch.no_grad()
    def route(self, queries: List[str], max_length: int = 256) -> RoutingDecision:
        enc = self.tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.device)

        outputs = self.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        layer_idx = self.routing_layer_index
        if layer_idx is None:
            layer_idx = len(hidden_states) // 2
        h = hidden_states[layer_idx]

        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        u = h[torch.arange(h.shape[0]), seq_lens]

        weights, indices, scores = self.router(u, self.community_embeddings)
        return RoutingDecision(weights=weights, indices=indices, scores=scores)

    @torch.no_grad()
    def generate(self, query: str, max_new_tokens: int = 64, **gen_kwargs) -> str:
        decision = self.route([query])
        enc = self.tokenizer(query, return_tensors="pt").to(self.device)

        with RouterDecisionContext(self.moe_layers, decision.weights, decision.indices):
            out_ids = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
