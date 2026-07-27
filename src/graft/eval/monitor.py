
from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import torch

class RouterMonitor:

    def __init__(
        self,
        num_communities: int,
        window: int = 100,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        tb_writer=None,
    ) -> None:
        self.num_communities = num_communities
        self.window = window
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.tb_writer = tb_writer

        self._freq_history: deque = deque(maxlen=window)
        self._step = 0

    def update(
        self,
        router_indices: torch.Tensor,
        router_scores: Optional[torch.Tensor] = None,
        prefix: str = "router",
    ) -> Dict[str, float]:
        self._step += 1

        one_hot = torch.zeros(self.num_communities, device=router_indices.device)
        for eid in router_indices.reshape(-1):
            if 0 <= eid < self.num_communities:
                one_hot[eid] += 1.0
        total_routes = router_indices.numel()
        freq = (one_hot / max(total_routes, 1)).cpu()
        self._freq_history.append(freq)

        stacked = torch.stack(list(self._freq_history), dim=0)
        avg_freq = stacked.mean(dim=0)
        entropy = _entropy(avg_freq)
        dead = int((avg_freq == 0).sum().item())

        metrics = {
            f"{prefix}/entropy": entropy,
            f"{prefix}/dead_experts": dead,
            f"{prefix}/max_freq": float(avg_freq.max().item()),
            f"{prefix}/min_freq": float(avg_freq.min().item()),
        }
        for i in range(self.num_communities):
            metrics[f"{prefix}/freq_community_{i}"] = float(avg_freq[i].item())

        if router_scores is not None:
            import torch.nn.functional as F

            probs = F.softmax(router_scores.float(), dim=-1).mean(dim=0)
            metrics[f"{prefix}/score_entropy"] = _entropy(probs.cpu())

        self._write(metrics)
        return metrics

    def _write(self, metrics: Dict[str, float]) -> None:
        if self.use_wandb:
            try:
                import wandb

                wandb.log(metrics, step=self._step)
            except Exception:
                pass
        if self.use_tensorboard and self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, self._step)

    def collapse_detected(self, entropy_threshold: float = 0.1) -> bool:
        if not self._freq_history:
            return False
        stacked = torch.stack(list(self._freq_history), dim=0)
        avg_freq = stacked.mean(dim=0)
        return bool(_entropy(avg_freq) < entropy_threshold)

def _entropy(probs: torch.Tensor) -> float:
    p = probs.float().clamp(min=1e-12)
    return float(-(p * p.log()).sum().item())
