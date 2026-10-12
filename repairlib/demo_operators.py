"""
Numeric verification harness for derived repair functions.

The symbolic classifier in repairlib/__init__.py proves conditions (3) and (4). This
module turns a derived repair into an executable torch lambda and checks it against a
naive multi-pass reference on random data -- the same check as experiments 1, but driven
by the classifier instead of by hand-written formulas.

That matters because a hand-written formula and its symbolic derivation can drift apart;
having one source of truth catches exactly the class of bug hit while building the
classifier (three separate substitution-order traps, each of which produced an h that
looked plausible and silently failed condition (3)).

Usage:
    python repairlib/demo_operators.py            # CPU
    python repairlib/demo_operators.py --cuda     # GPU
"""
from __future__ import annotations

import argparse
import sys

import sympy as sp
import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from repairlib import (  # noqa: E402
    Verdict, corpus, derive_repair, s, s2, u, c,
)

DT = torch.float64


# --------------------------------------------------------------- repair -> torch
def make_repair(res, eps: float = 1e-6) -> object:
    """Compile the symbolic h into a torch callable  h(t, s, s2) -> tensor.

    h is an expression in (u, s, s2) built by Eq.5; `u` is the accumulator value we
    feed in. Any remaining free symbols (e.g. the `eps` in RMSNorm) are folded into
    concrete numbers first, otherwise lambdify leaves them as sympy objects and torch
    fails on them.
    """
    h = res.h
    assert h is not None
    free = h.free_symbols - {u, s, s2}
    subs_map = {}
    for sym in free:
        if str(sym) == "eps":
            subs_map[sym] = eps
        else:
            raise ValueError(f"unsupported free symbol in h: {sym}")
    h = h.subs(subs_map)
    lam = sp.lambdify((u, s, s2), h, modules="numpy")

    def repair(t, s_val, s2_val):
        out = lam(t, s_val, s2_val)
        return torch.as_tensor(out, dtype=t.dtype, device=t.device)

    return repair


# --------------------------------------------------------------- streaming kernels
def online_reduce(x, f_kind: str, g_kind: str, repair, dim=-1):
    """One streaming pass: maintain the producer value r, accumulate the consumer
    reduction with the repair applied at every step."""
    from repairlib import apply_f  # noqa: F401  (documentation of intent)

    n = x.shape[dim]
    shp = list(x.shape)
    shp[dim] = 1

    if f_kind == "add":
        acc = torch.zeros(shp, dtype=x.dtype, device=x.device)
        combine = lambda a, b: a + b            # noqa: E731
    elif f_kind == "max":
        acc = torch.full(shp, float("-inf"), dtype=x.dtype, device=x.device)
        combine = torch.maximum                    # noqa: E731
    else:
        raise ValueError(f_kind)

    # the producer reduction
    if f_kind in ("add", "max"):
        if g_kind in ("softmax", "logsumexp", "rmsnorm_max"):
            r = torch.full(shp, float("-inf"), dtype=x.dtype, device=x.device)
            prod = torch.maximum
            r_init = float("-inf")
        elif g_kind == "stable_l2":
            r = torch.zeros(shp, dtype=x.dtype, device=x.device)
            prod = torch.maximum
            r_init = 0.0
        else:
            raise ValueError(g_kind)
    else:
        raise ValueError(f_kind)

    for j in range(n):
        cj = x.narrow(dim, j, 1)
        r_prev = r.clone()
        # producer update
        if g_kind == "stable_l2":
            r = torch.maximum(r, cj.abs())
        elif g_kind in ("softmax", "logsumexp", "rmsnorm_max"):
            r = torch.maximum(r, cj)
        # consumer term at the *current* producer value
        if g_kind in ("softmax", "logsumexp"):
            term = torch.exp(cj - r)
        elif g_kind == "stable_l2":
            term = (cj / r) ** 2
        elif g_kind == "rmsnorm_max":
            term = cj * cj / (r * r)
        # repair the stale accumulator, then add the new term
        acc = combine(repair(acc, r_prev, r), term)
    return acc, r


def reference(x, g_kind: str, dim=-1):
    if g_kind in ("softmax", "logsumexp"):
        m = x.amax(dim=dim, keepdim=True)
        return torch.exp(x - m).sum(dim=dim, keepdim=True)
    if g_kind == "stable_l2":
        return (x * x).sum(dim=dim, keepdim=True)
    if g_kind == "rmsnorm_max":
        return (x * x).amax(dim=dim, keepdim=True)
    raise ValueError(g_kind)


# --------------------------------------------------------------- driver
# (classifier-name, f_kind, g_kind, reference) for the rows we can execute
EXECUTABLE = {
    "online_softmax": ("add", "softmax", "sum(exp(x-amax))"),
    "logsumexp": ("add", "logsumexp", "sum(exp(x-amax))"),
    "stable_l2": ("add", "stable_l2", "sum(x^2)"),
    "rmsnorm_max": ("max", "rmsnorm_max", "max(x^2)"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"device = {dev}   torch = {torch.__version__}")
    print("=" * 96)

    rc = 0
    for name, f, g, how in corpus():
        res = derive_repair(f, g)
        spec = EXECUTABLE.get(name)
        if spec is None:
            print(f"{name:20s} {res.verdict.value:12s} (no executable spec)")
            continue
        f_kind, g_kind, ref_desc = spec
        torch.manual_seed(0)
        x = torch.randn(4, 16, 64, dtype=DT, device=dev)
        if name == "stable_l2":
            x = x * 1e3
        if not res.verdict.fusable:
            # The classifier says this repair cannot be folded into an accumulator.
            # Running the stream anyway would be meaningless, so we report the verdict
            # instead of a number -- that is the honest outcome.
            print(f"{name:20s} {res.verdict.value:12s} "
                  f"h={str(res.h):32s} {'not-foldable: ' + '; '.join(res.notes)}")
            continue
        repair = make_repair(res)
        got, r = online_reduce(x, f_kind, g_kind, repair)
        # The framework's accumulator holds the DENORMALISED quantity: for g = (c/s)^2
        # it is the residual sum of squares scaled by 1/s^2, so the final conversion is
        # got * s^2. Feeding h an already-absolute-scale value would double-count the
        # rescaling -- a representation trap worth stating explicitly.
        if g_kind == "stable_l2":
            got = got * (r * r)
        ref = reference(x, g_kind)
        err = (got - ref).abs().max().item()
        ok = err < 1e-6
        rc |= 0 if ok else 1
        print(f"{name:20s} {res.verdict.value:12s} "
              f"h={str(res.h):32s} vs {ref_desc:16s} max_err={err:.3e} "
              f"{'PASS' if ok else 'FAIL'}")

    print("=" * 96)
    print("Numeric agreement between the CLASSIFIER-DERIVED repair and a naive "
          "multi-pass reference.")
    print("A hand-written kernel and its symbolic derivation can drift; this harness "
          "keeps one source of truth.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
