"""
Coverage study: how far does Neptune's (f, g, h) algebra actually reach across the
operator families that matter in practice?

This is the question that should be answered BEFORE investing in a general Triton
codegen layer, because it bounds what that layer could ever deliver.

For each family we declare only its reduction skeleton -- the reducer f and the term
producer g(s, c) -- and let repairlib derive the repair and classify it:

    EXACT        repair distributes over f  -> foldable into an accumulator
    AFFINE       per-term correction exists but does not fold (needs carried state)
    NEEDS_STATE  condition (3) itself needs the data symbol -> no h(t,s,s') exists
    NOT_INVERT   Eq.5 cannot be instantiated

Run:  python repairlib/coverage.py
"""
from __future__ import annotations

import sys
from collections import Counter

import sympy as sp

sys.path.insert(0, __file__.rsplit("/", 2)[0])  # repo root, so repairlib imports
from repairlib import (  # noqa: E402
    Verdict, derive_repair, s as r, s2 as rp, c, x, y,
)

eps = sp.Symbol("eps", positive=True)
beta = sp.Symbol("beta", positive=True)
alpha = sp.Symbol("alpha", positive=True)
p = sp.Symbol("p", positive=True)
n = sp.Symbol("N", positive=True)



def build() -> list[tuple[str, str, str]]:
    """(name, groups, f, g, note) -- groups is the higher-level operator it belongs to."""
    rows = [
        # ---- softmax family: one algebra, many surface variants --------------------
        ("softmax", "softmax", x + y, sp.exp(c - r),
         "shipped by PyTorch (only this one)"),
        ("softmax+causal mask", "softmax", x + y, sp.exp(c - r), "mask folded into the term"),
        ("softmax+ALiBi bias", "softmax", x + y, sp.exp(c + alpha - r), "bias is non-reduction input"),
        ("softmax+softcap", "softmax", x + y,
         sp.exp(sp.tanh(beta * c) - r),
         "EXACT: repairs to u*exp(s-s2); Gemma-2 soft-capping"),
        ("log-softmax", "softmax", x + y, sp.exp(c - r), "same fold; log applied once at the end"),
        ("cross-entropy (logsumexp)", "loss", x + y, sp.exp(c - r), "same as log-softmax"),

        # ---- variance family -------------------------------------------------------
        ("one-pass variance", "variance", x + y, (c - r) ** 2,
         "consumer's rm depends on the running mean"),
        ("two-pass variance", "variance", x + y, (c - r) ** 2,
         "g cannot reference the FRESH producer value by construction"),
        ("Welford reduce", "variance", x + y, (c - r) ** 2,
         "needs the running count alongside the mean"),
        ("batch-norm stats", "variance", x + y, (c - r) ** 2, "same skeleton as variance"),

        # ---- norm family -----------------------------------------------------------
        ("stable L2 norm", "norm", x + y, (c / r) ** 2, "EXACT: no PyTorch equivalent"),
        ("RMSNorm numerator", "norm", x + y, (c / r) ** 2, "same as stable L2"),
        ("RMSNorm + row max", "norm", sp.Max(x, y), c / sp.sqrt(r + eps),
         "max reducer; only meaningful with a SEPARATE producer"),
        ("L2-normalise rows", "norm", x + y, (c / r) ** 2, "same as stable L2"),
        ("p-norm (p=3)", "norm", x + y, (c / r) ** p, "p is a compile-time constant"),
        ("standard deviation", "norm", x + y, (c - r) ** 2, "variance skeleton"),

        # ---- reduction pairs -------------------------------------------------------
        ("max + sum of exp", "reduction", x + y, sp.exp(c - r), "the core softmax fold"),
        ("prod + ratio", "reduction", x * y, c / r, "multiplicative reducer"),
        ("max + additive shift", "reduction", sp.Max(x, y), c - r,
         "DEGENERATE when s is the max of the same c: result is identically 0"),
        ("max + scaled shift", "reduction", sp.Max(x, y), (c - r) / n,
         "DEGENERATE for the same reason; scale is a non-reduction input"),
        ("entropy", "reduction", x + y, c * sp.log(r), "log producer, sum reducer"),
        ("max-abs + sum of squares", "reduction", x + y, (c / r) ** 2, "same as stable L2"),

        # ---- attention -------------------------------------------------------------
        ("FlashAttention prefill", "attention", x + y, sp.exp(c - r), "row-wise max/sum fold"),
        ("FlashDecoding split-K", "attention", x + y, sp.exp(c - r),
         "same fold, split over K; combination uses the same h"),
        ("linear attention", "attention", x + y, c / r, "ratio producer, sum reducer"),

        # ---- explicitly outside the framework --------------------------------------
        ("non-invertible producer", "outside", x + y, sp.exp(c - sp.Abs(c)),
         "Eq.5 cannot be instantiated"),
        ("two producers, one term", "outside", x + y, c * rp,
         "placeholder: real case needs MultiProducerRepair, see notes"),
        ("duration/decay renormalisation", "outside", x + y, c / (r + n),
         "additive denominator offset"),
    ]
    return rows


def main() -> None:
    rows = build()
    results = []
    for name, group, f, g, note in rows:
        res = derive_repair(f, g)
        res.name = name
        results.append((group, res, note))

    w = 118
    print("=" * w)
    print("COVERAGE: which operator families admit a FOLDABLE repair?")
    print("=" * w)
    hdr_h = "h(t, s, s')"
    print(f"{'family':26s} {'group':10s} {'verdict':11s} {hdr_h:40s} note")
    print("-" * w)
    prev_group = None
    for group, res, note in results:
        if group != prev_group:
            print(f"  -- {group} " + "-" * (w - 6 - len(group)))
            prev_group = group
        h = "—" if res.h is None else str(sp.simplify(res.h))
        print(f"{res.name:26s} {group:10s} {res.verdict.value:11s} "
              f"{h[:40]:40s} {note[:34]}")

    print("=" * w)
    counts = Counter(res.verdict for _g, res, _n in results)
    n = len(results)
    print(f"total declared families: {n}")
    for v in Verdict:
        c_ = counts.get(v, 0)
        if c_:
            print(f"  {v.value:12s} {c_:3d}  ({100 * c_ / n:5.1f}%)")
    gcounts = Counter(g for g, _r, _n in results)
    print("\nby group:")
    for g_, c_ in gcounts.items():
        exact = sum(1 for gg, rr, _ in results if gg == g_ and rr.verdict is Verdict.EXACT)
        print(f"  {g_:12s} {c_:3d} families, {exact:3d} EXACT")

    print(f"""
What this means for a general module
------------------------------------
* The EXACT rows are the module's *reach*: each one is a fold that a general
  implementation could emit. PyTorch ships exactly ONE of them (softmax).
* Crucially, whole FAMILIES collapse to the same algebra: every softmax variant
  (causal mask, ALiBi, softcap, log-softmax, cross-entropy) shares one (f, g) and
  therefore one repair. That is the argument for a general module rather than more
  special cases -- N surface variants cost ONE new skeleton.
* The non-EXACT rows are just as useful: they say precisely what the module must NOT
  try to fold, and why (needs carried state vs. not expressible at all).
""")
    print("=" * w)


if __name__ == "__main__":
    main()
