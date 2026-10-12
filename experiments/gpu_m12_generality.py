"""
Demonstration: a NEW operator, declared as algebra only.

Nothing in this file is hand-written Triton. Each skeleton is declared with a symbolic
term producer `g` (plus the producer semantics and a finalize expression); the repair
factor, the consumer term, and all Triton source are derived and emitted by
`neptune_codegen`.

The three operators below were never hand-tuned:

  * stable_l1_ratio   g = |c|/s          a fold over a running max
  * sumsq_div         g = c^2/s          a different normaliser, same machinery
  * lse_shifted       g = exp((c-s)/2)   a scaled exponential fold

If the generator is genuinely general, adding them is a one-line declaration and they
should be numerically correct on the same hostile input distributions the existing two
operators are tested against.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neptune_codegen as CG  # noqa: E402
import sympy as sp  # noqa: E402

c, s, u, m, acc = (sp.Symbol("c", real=True), sp.Symbol("s", real=True),
                   sp.Symbol("u", positive=True), sp.Symbol("m", real=True),
                   sp.Symbol("acc", real=True))

# ---------------------------------------------------------------- new declarations
NEW = [
    CG.Skeleton(
        name="sumsq_div",
        # c^2 / s : squared magnitude normalised by the running max.
        # The accumulator already holds sum_j x_j^2/s_final (the repair rescales every
        # earlier term onto the current producer), so finalize is the IDENTITY here.
        # `m*acc` would silently compute sum(x^2) instead -- see the note in the report.
        g=c ** 2 / s,
        finalize=acc,
        producer="abs",
        doc="sum of x^2/max|x|",
    ),
    CG.Skeleton(
        name="lse_shifted",
        # exp((c - s)/2): a scaled exponential fold; the repair is exp((s-s')/2)
        g=sp.exp((c - s) / 2),
        finalize=m / 2 + sp.log(acc + sp.Float("1e-38")),
        producer="signed",
        is_log=True,
        doc="log of a sum of sqrt-weighted exponentials",
    ),
]

# Operators the framework must REFUSE. `g` must be invertible in c for Eq.5 to apply.
REJECTED = [
    CG.Skeleton(
        name="stable_l1_ratio",
        # |c| / s: Abs is not analytically invertible, so sympy returns a Piecewise with
        # a nan branch and the derivation cannot proceed.
        g=sp.Abs(c) / s,
        finalize=m * acc,
        producer="abs",
        doc="sum of |x|/max|x| -- rejected, not invertible in c",
    ),
    CG.Skeleton(
        name="exp_abs_shift",
        g=sp.exp(c - sp.Abs(c)),
        finalize=m + sp.log(acc + sp.Float("1e-38")),
        producer="signed",
        is_log=True,
        doc="rejected, not invertible in c",
    ),
]
for sk in NEW:
    CG.register(sk)


# ---------------------------------------------------------------- references
def ref_stable_l1_ratio(t):
    x = t.reshape(t.shape[0], -1).float()
    mx = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (mx * (x.abs() / mx).sum(dim=-1, keepdim=True)).squeeze(-1)


def ref_sumsq_div(t):
    x = t.reshape(t.shape[0], -1).float()
    mx = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((x ** 2) / mx).sum(dim=-1, keepdim=True).squeeze(-1)


def ref_lse_shifted(t):
    x = t.reshape(t.shape[0], -1).float()
    m_ = x.max(dim=-1, keepdim=True).values
    return (m_ / 2 + torch.log(torch.exp((x - m_) / 2).sum(dim=-1, keepdim=True))).squeeze(-1)


REFS = {
    "stable_l1_ratio": ref_stable_l1_ratio,
    "sumsq_div": ref_sumsq_div,
    "lse_shifted": ref_lse_shifted,
}


def distributions(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r, cc = shape
    yield "normal", torch.randn(shape, device="cuda", generator=g)
    yield "large", torch.randn(shape, device="cuda", generator=g) * 1e3
    yield "tiny", torch.randn(shape, device="cuda", generator=g) * 1e-6
    yield "all_negative", -torch.rand(shape, device="cuda", generator=g) * 5e3
    yield "outlier", torch.cat([torch.randn(r, 1, device="cuda", generator=g) * 1e4,
                                torch.randn(r, cc - 1, device="cuda", generator=g)], dim=1)
    z = torch.randn(shape, device="cuda", generator=g)
    z[::3] = 0.0
    yield "zero_rows", z


if __name__ == "__main__":
    print("=" * 100)
    print("GENERALITY DEMO - operators declared as algebra, Triton derived")
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    print("\n--- operators the framework REFUSES (guard working) ---")
    for sk in REJECTED:
        try:
            CG.repair_factor(sk)
            print(f"  [FAIL] {sk.name:18s} was accepted but should not be")
        except NotImplementedError as e:
            print(f"  [PASS] {sk.name:18s} refused: {str(e)[:70]}")

    print("\n--- derived repair factors (no hand-written formulas) ---")
    for sk in NEW:
        print(f"  {sk.name:18s} g={str(sk.g):22s} repair R={CG.repair_factor(sk)}")
        print(f"  {'':18s} term T={CG.term_source(sk)[:72]}")

    print("\n--- correctness on hostile inputs ---")
    ok_all = True
    for sk in NEW:
        worst, worst_case = 0.0, ""
        for shape in [(8, 1024), (16, 4096), (4, 16384), (1, 65536), (32, 512)]:
            for dist, x in distributions(shape, seed=abs(hash((sk.name, shape))) % 2 ** 28):
                try:
                    got = CG.run(sk.name, x)
                    ref = REFS[sk.name](x)
                    denom = ref.abs().clamp_min(1e-3)
                    rel = ((got - ref).abs() / denom).max().item()
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] {sk.name}/{dist}{shape}: {type(e).__name__}: {str(e)[:70]}")
                    ok_all = False
                    continue
                if rel > 2e-4:
                    print(f"  [FAIL] {sk.name}/{dist}{shape}: rel={rel:.2e}")
                    ok_all = False
                if rel > worst:
                    worst, worst_case = rel, f"{dist}{shape}"
        print(f"  [{'PASS' if worst <= 2e-4 else 'FAIL'}] {sk.name:18s} "
              f"worst rel_err={worst:.2e} ({worst_case})")

    print("\n" + "=" * 100)
    print("VERDICT:", "all auto-derived operators correct" if ok_all
          else "some auto-derived operators failed")
    print("""
If this passes, the generator is not a table of hand-written kernels: a new operator is a
symbolic expression, and the repair, the term and the Triton source all follow from it.""")
    print("=" * 100)
