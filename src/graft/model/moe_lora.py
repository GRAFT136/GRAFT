
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class CommunityMoELoraLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,
        base_bias: Optional[torch.Tensor] = None,
        rank: int = 16,
        lora_alpha: float = 32.0,
        num_communities: int = 8,
        use_global_expert: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.num_communities = num_communities
        self.use_global_expert = use_global_expert

        self.base_weight = nn.Parameter(base_weight.clone(), requires_grad=False)
        if base_bias is not None:
            self.base_bias = nn.Parameter(base_bias.clone(), requires_grad=False)
        else:
            self.base_bias = None

        if use_global_expert:
            self.lora_A_global = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B_global = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A_global, a=math.sqrt(5))

        self.lora_A_local = nn.Parameter(torch.empty(num_communities, rank, in_features))
        self.lora_B_local = nn.Parameter(torch.zeros(num_communities, out_features, rank))
        for i in range(num_communities):
            nn.init.kaiming_uniform_(self.lora_A_local[i], a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        router_weights: Optional[torch.Tensor] = None,
        router_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, self.in_features)
        T = x_flat.shape[0]
        dev = x_flat.device

        h = F.linear(x_flat, self.base_weight, self.base_bias)

        x_lora = x_flat.to(torch.float32)

        if self.use_global_expert:
            A_g = self.lora_A_global.to(dev)
            B_g = self.lora_B_global.to(dev)
            lora_out_g = ((x_lora @ A_g.T) @ B_g.T).to(orig_dtype)
            h = h + lora_out_g * self.scaling

        if router_weights is not None and router_indices is not None:
            h = self._apply_community_experts(h, x_lora, router_weights, router_indices, T, dev, orig_dtype)

        return h.reshape(orig_shape[:-1] + (self.out_features,))

    def _apply_community_experts(
        self,
        h: torch.Tensor,
        x_lora: torch.Tensor,
        router_weights: torch.Tensor,
        router_indices: torch.Tensor,
        T: int,
        dev: torch.device,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        B_batch = router_weights.shape[0]
        kappa = router_weights.shape[1]
        tokens_per_seq = 1 if T == B_batch else T // B_batch
        assert T == B_batch * tokens_per_seq, (
            f"Token count {T} must be divisible by batch size {B_batch}."
        )

        rw = router_weights.to(dev).unsqueeze(1).expand(B_batch, tokens_per_seq, kappa).reshape(T, kappa)
        ri = router_indices.to(dev).unsqueeze(1).expand(B_batch, tokens_per_seq, kappa).reshape(T, kappa)

        K = self.num_communities
        weight_te = x_lora.new_zeros(T, K)
        weight_te.scatter_add_(1, ri, rw.to(weight_te.dtype))

        expert_delta = torch.zeros_like(h)
        for k in range(K):
            w = weight_te[:, k]
            if not torch.any(w != 0):
                continue
            A = self.lora_A_local[k].to(dev)
            B = self.lora_B_local[k].to(dev)
            out = ((x_lora @ A.T) @ B.T).to(orig_dtype) * self.scaling
            expert_delta = expert_delta + w.to(orig_dtype).unsqueeze(-1) * out

        return h + expert_delta

    def zero_local_expert(self, idx: int) -> None:
        with torch.no_grad():
            self.lora_A_local[idx].zero_()
            self.lora_B_local[idx].zero_()

    def zero_all_local_experts(self) -> None:
        with torch.no_grad():
            self.lora_A_local.zero_()
            self.lora_B_local.zero_()

    def zero_global_expert(self) -> None:
        if self.use_global_expert:
            with torch.no_grad():
                self.lora_A_global.zero_()
                self.lora_B_global.zero_()
