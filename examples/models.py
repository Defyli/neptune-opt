"""Sample models used by the examples.

Each one contains a reduction pair that `neptune_opt` can recognise: a producer reduction
whose result is consumed inside a second reduction. They are written the way a user would
write them in eager PyTorch, i.e. the multi-pass "decomposed" form -- that is what a graph
optimiser actually receives.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StableL2Norm(nn.Module):
    """Row-wise L2 norm of a normalised vector: `|x|_2` computed without overflow.

    Two reductions that both need the row maximum:
        m = max|x|            (producer)
        sum (x/m)^2           (consumer)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        return (m * torch.sqrt(((x / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


class SoftmaxLogits(nn.Module):
    """The max + sum-of-exp fold that every softmax, log-softmax and cross-entropy contains."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.amax(dim=-1, keepdim=True)
        return (torch.log(torch.exp(x - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


class RMSNormMax(nn.Module):
    """The paper's own headline example: `max_j x_j / sqrt(mean(x^2) + eps)`.

    Note the two reducers differ: the producer `mean(x^2)` is ADDITIVE while the consumer is a
    maximum. That is why this operator needed the producer/consumer reducers to be independent
    axes.

    Written with `.mean()` on purpose -- the row length stays implicit, and the generated
    kernel reads it from its own runtime argument (`ncols`), so one kernel serves every width.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = (x * x).mean(dim=-1, keepdim=True)
        return (x / torch.sqrt(r + self.eps)).amax(dim=-1)


class TinyAttention(nn.Module):
    """A miniature attention head: softmax over scores, then a weighted sum.

    The softmax is the part `neptune_opt` can fuse; the matmuls are left to Inductor.
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.q(x) @ self.k(x).transpose(-1, -2)
        m = scores.amax(dim=-1, keepdim=True)
        weights = torch.exp(scores - m)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights @ self.v(x)
