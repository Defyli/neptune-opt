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


def rel_tolerance(dtype, ulps: float = 4.0) -> float:
    """A relative tolerance appropriate to `dtype`.

    Comparing a `bf16` kernel against an `fp32` reference with an `fp32` bound fails for the
    right answer: bf16 carries ~8 mantissa bits, so a relative error near 2**-8 is the FORMAT,
    not a bug. The tolerances are a few ulps of the dtype, which is tight enough to catch a
    wrong formula (those are off by O(1)) and loose enough not to fail on rounding.

    Verified in `experiments/gpu_m19_dtypes.py`, which reports the measured error per dtype so
    the bound can be seen to hold rather than assumed.
    """
    import torch

    return ulps * float(torch.finfo(dtype).eps)


def summation_tolerance(dtype, n, ulps: float = 4.0) -> float:
    """A relative tolerance for a SUM of `n` terms whose result is stored as `dtype`.

    Two effects, and only the first is what `rel_tolerance` covers:

      * the result is narrowed to `dtype`, costing up to `ulps` of ITS eps;
      * the sum itself is accumulated in fp32 over `n` terms, and a different summation order
        (the kernel reduces per block, torch reduces its own way) costs roughly `eps_f32 * sqrt(n)`
        for a random walk -- measured to matter: `linear_attn` over 2048 terms showed 6.2e-6, which
        is 13 ulps of fp32 and completely correct.

    The sqrt(n) growth is the standard random-walk model for accumulated rounding, not a fitted
    constant; the measured errors sit an order of magnitude inside it (see
    `experiments/gpu_m19_dtypes.py`, which prints both).
    """
    import torch

    n = max(int(n), 1)
    own = ulps * float(torch.finfo(dtype).eps)
    acc = 2 * ulps * float(torch.finfo(torch.float32).eps) * (n ** 0.5)
    return own + acc


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
    if name == "lse_scaled":
        return torch.logsumexp(t / p["T"], dim=-1)
    if name == "rmsnorm_max":
        # divide by the DECLARED N, not by the row length: the kernel has N baked in as a
        # literal, so using .mean() here would silently test two different operators.
        r = (t * t).mean(dim=-1, keepdim=True)   # N is `ncols`, i.e. the row length
        return (t / torch.sqrt(r + p["eps"])).amax(dim=-1)
    if name in ("variance", "std"):
        # PyTorch's OWN kernel, deliberately, rather than a hand-written mean/centre/square.
        # The design notes warn that a reference written from the same reading as the kernel
        # is a self-consistent error that no test can see; here an independent implementation
        # exists, so it is the better reference. `unbiased=False` is the N divisor the
        # skeleton declares.
        if name == "variance":
            return t.var(dim=-1, unbiased=False)
        return t.std(dim=-1, unbiased=False)
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
    if name == "lse_scaled":
        # the DECOMPOSED form Inductor is handed: subtract the scaled max, exponentiate, sum
        return lse(t / p["T"])
    if name == "rmsnorm_max":
        r = (t * t).mean(dim=-1, keepdim=True)   # N is `ncols`, i.e. the row length
        return (t / torch.sqrt(r + p["eps"])).amax(dim=-1)
    if name in ("variance", "std"):
        # What Inductor is handed today: the producer is a SEPARATE reduction, so the centred
        # pass is a second kernel and the mean is a materialised intermediate. The generated
        # two-pass kernel does the same arithmetic with one launch and no intermediate.
        centred = (t - t.mean(dim=-1, keepdim=True)) ** 2
        if name == "variance":
            return centred.mean(dim=-1)
        return centred.mean(dim=-1).sqrt()
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
