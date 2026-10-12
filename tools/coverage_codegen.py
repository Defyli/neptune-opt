"""
Codegen coverage: of the families the ALGEBRA admits, how many can the GENERATOR
actually turn into a kernel?

There are three different "coverage" numbers in this project and they must not be
conflated:

  (A) ALGEBRA reach     -- `repairlib/coverage.py`: how many families admit a foldable
                           repair at all. This is a property of Neptune's conditions
                           (3)/(4), not of any implementation.
  (B) GENERATION reach  -- this file: how many of those the code generator can emit
                           Triton source for, as it exists today.
  (C) RECOGNITION reach -- how many the graph selector recognises. Not measured here.

(B) is measured per family with a deliberately WEAK finalize (`acc`, the identity):
if a family cannot even be derived and translated with the simplest possible readout,
no choice of `finalize` would save it. A pass therefore means "the hard part works";
the human still owes the correct `finalize`.

Refusals are classified by CAUSE, because "unsupported" is useless on its own:
    f-not-additive        the reduction template folds an additive reducer only
    not-invertible        Eq.5 cannot be instantiated (g_c^-1 does not exist)
    repair-not-multiplicative  h does not factor as u * R(s, s')
    unbound-symbol        a scalar parameter (p, N, alpha, eps) has no kernel argument
    translator-unsupported the algebra is fine but the sympy->Triton translator lacks a case

Run:  python coverage_codegen.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import neptune_codegen as CG  # noqa: E402
from neptune_codegen import c_sym, s2_sym, s_sym  # noqa: E402
from neptune_codegen.sym_to_triton import TranslationError, to_triton  # noqa: E402
from repairlib import Verdict, derive_repair  # noqa: E402
import repairlib.coverage as RC  # noqa: E402

# repairlib declares `s`, `s2`, `c` with these names; the generator uses its own
# symbols, so the probe maps by name.
SYMMAP = {RC.r: s_sym, RC.c: c_sym}   # coverage.py aliases s -> r

W = 108


def classify_failure(err: Exception) -> str:
    """Name the actual blocker. 'unsupported' alone is not actionable."""
    msg = str(err)
    if "mentions the FRESH producer" in msg or "mentions the term value" in msg:
        return "g-depends-on-fresh-value"
    if "does not depend on the producer" in msg:
        # separation succeeds trivially (R=1) but the operator is not a fold
        return "g-ignores-producer"
    if "does not separate" in msg:
        return "repair-not-multiplicative"
    if "separating g changed its value" in msg:
        return "separation-changed-value"
    if "is negative, so it does not distribute" in msg:
        return "repair-does-not-distribute"
    if isinstance(err, TranslationError):
        if "no Triton binding" in msg:
            return "needs-scalar-param-plumbing"
        return "translator-unsupported"
    return f"other({type(err).__name__})"


def probe(f, g):
    """Try to derive + translate a kernel for one family. Returns (ok, detail)."""

    g2 = g.xreplace(SYMMAP)
    # Any free symbol that is not one of the three algebra symbols is a scalar parameter
    # the kernel needs: declare it as an extra kernel argument, exactly as an integration
    # would. This is what closes the "no Triton binding for symbol eps/alpha/N" refusals.
    extra = sorted(g2.free_symbols - {s_sym, s2_sym, c_sym}, key=lambda z: z.name)
    params = tuple((z, z.name) for z in extra)
    # The two reducers are part of the declaration, exactly like `finalize`, so the probe
    # searches that space. It is not guessing on the generator's behalf: it is asking "does
    # ANY declaration of this family emit?", which is the coverage question.
    # `(max, sum)` first because it is the overwhelmingly common shape.
    last = None
    # The consumer reducer is DETERMINED by f -- it is not a free choice. Allowing the
    # probe to pick "sum" for a mul reducer (f = x*y) made it emit prod+ratio, which is
    # simply a false declaration: the emitted fold would not compute f.
    if f == RC.x + RC.y:
        cons_reducers = ("sum",)
    elif f == sp.Max(RC.x, RC.y):
        cons_reducers = ("max",)
    else:
        return False, "f-needs-carried-state", ""   # only additive and max are foldable
    for pr, cr in [(p_, c_) for c_ in cons_reducers
                   for p_ in ("max", "sum")]:
        for producer, is_log in (("signed", True), ("abs", False)):
            sk = CG.Skeleton(name="probe", g=g2, finalize=CG._FINAL_ACC,
                             producer=producer, is_log=is_log, params=params,
                             producer_reducer=pr, consumer_reducer=cr)
            try:
                # use the REAL emitters, not a reimplementation, so this measures the
                # generation path an integration would actually call
                vals = {arg: 1.0 for _sym, arg in params}   # literals; value irrelevant
                r_src, t_src = CG.scale_source(sk, vals), CG.term_source(sk, vals)
                # The fold itself, not just its two halves: an affine repair's bias lives
                # only in this line, so reporting scale/term alone would hide it and let a
                # bias-bearing kernel look like a bare rescale.
                fold = CG.consumer_update(sk, vals)
            except (NotImplementedError, TranslationError) as e:
                last = e
                continue
            return True, f"{pr}/{cr}", f"{fold}\n      scale={r_src}\n      term={t_src}"
    if last is None:
        return False, "f-not-foldable", ""
    return False, classify_failure(last), ""


def main() -> None:
    rows = RC.build()
    print("=" * W)
    print("CODEGEN COVERAGE -- can the GENERATOR emit a kernel, given that the algebra admits one?")
    print("=" * W)
    print(f"{'family':28s} {'group':10s} {'verdict':11s} {'emit':7s} detail")
    print("-" * W)

    emitted, refused = [], []
    reasons = Counter()
    prev = None
    for name, group, f, g, note in rows:
        res = derive_repair(f, g)
        ok, why, detail = probe(f, g)
        if ok:
            emitted.append((name, group, res.verdict.value, why, detail))
            mark, extra = "YES", detail[:44]
        else:
            refused.append((name, group, res.verdict.value, why))
            reasons[why] += 1
            mark, extra = "no", why
        if group != prev:
            print(f"  -- {group} " + "-" * (W - 6 - len(group)))
            prev = group
        print(f"{name:28s} {group:10s} {res.verdict.value:11s} {mark:7s} {extra}")

    n = len(rows)
    print("=" * W)
    print(f"total declared families          : {n}")
    print(f"  ALGEBRA admits a fold (EXACT)  : "
          f"{sum(1 for _n, _g, f, gg, _no in rows if derive_repair(f, gg).verdict is Verdict.EXACT):3d}")
    print(f"  CODEGEN emits a kernel         : {len(emitted):3d}  ({100 * len(emitted) / n:.1f}%)")
    print(f"  refused                        : {len(refused):3d}")
    print("\nrefusals by cause:")
    for why, cnt in reasons.most_common():
        print(f"  {why:28s} {cnt:3d}")

    print("\nemitted, by verdict:")
    for v, cnt in Counter(e[2] for e in emitted).most_common():
        print(f"  {v:28s} {cnt:3d}")

    print("\nrefused, by (verdict, cause) -- this is the work list:")
    for (v, why), cnt in Counter((r[2], r[3]) for r in refused).most_common():
        names = [r[0] for r in refused if r[2] == v and r[3] == why]
        print(f"  {v:12s} {why:28s} {cnt:3d}  {', '.join(names[:3])}"
              + (" ..." if len(names) > 3 else ""))
    print("=" * W)


if __name__ == "__main__":
    main()
