"""
The reference implementations the kernels are checked against.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.skeleton import SKELETONS, param_defaults


def reference(name: str, x, **params):
    """Row-wise torch reference (the mathematically exact result)."""
    import torch

    p = {**param_defaults(name), **params}
    t = x.reshape(x.shape[0], -1).float()
    if name == "logsumexp":
        return torch.logsumexp(t, dim=-1)
    if name == "lse_biased":
        return torch.logsumexp(t + p["alpha"], dim=-1)
    if name == "softcap_lse":
        # logsumexp of the squashed logits; torch.logsumexp is the stable form
        return torch.logsumexp(torch.tanh(p["beta"] * t), dim=-1)
    if name == "rmsnorm_max":
        # divide by the DECLARED N, not by the row length: the kernel has N baked in as a
        # literal, so using .mean() here would silently test two different operators.
        r = (t * t).mean(dim=-1, keepdim=True)   # N is `ncols`, i.e. the row length
        return (t / torch.sqrt(r + p["eps"])).amax(dim=-1)
    # clamp_min is required: without it an all-zero row gives 0/0 = nan in the
    # REFERENCE, which the tests would then attribute to the kernel.
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    if name == "stable_l2":
        return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)
    if name == "sumsq_div":
        return ((t * t) / m).sum(dim=-1, keepdim=True).squeeze(-1)
    if name == "linear_attn":
        return (t / m).sum(dim=-1, keepdim=True).squeeze(-1)
    if name == "decay_renorm":
        return (t / (m + p["N"])).sum(dim=-1, keepdim=True).squeeze(-1)
    raise KeyError(name)


def graph_reference(name: str, x, **params):
    """The decomposed torch form, i.e. what Inductor is handed today."""
    import torch

    p = {**param_defaults(name), **params}
    t = x.reshape(x.shape[0], -1).float()

    def lse(z):
        m = z.amax(dim=-1, keepdim=True)
        return (torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)

    if name == "logsumexp":
        return lse(t)
    if name == "lse_biased":
        return lse(t + p["alpha"])
    if name == "softcap_lse":
        return lse(torch.tanh(p["beta"] * t))
    if name == "rmsnorm_max":
        r = (t * t).mean(dim=-1, keepdim=True)   # N is `ncols`, i.e. the row length
        return (t / torch.sqrt(r + p["eps"])).amax(dim=-1)
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    if name == "stable_l2":
        return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)
    if name == "sumsq_div":
        return ((t * t) / m).sum(dim=-1, keepdim=True).squeeze(-1)
    if name == "linear_attn":
        return (t / m).sum(dim=-1, keepdim=True).squeeze(-1)
    if name == "decay_renorm":
        return (t / (m + p["N"])).sum(dim=-1, keepdim=True).squeeze(-1)
    raise KeyError(name)
