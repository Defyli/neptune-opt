"""
Recovering a skeleton's scalar parameters from the graph.
"""
from __future__ import annotations

import dataclasses
import random

import sympy as sp




def _respects(sym, val) -> bool:
    """Does an inferred value satisfy the symbol's DECLARED domain?

    Needed because two skeletons can coincide at a degenerate parameter value: with N = 0,
    `decay_renorm` (g = c/(N+s)) collapses onto `linear_attn` (g = c/s), and without this
    check the more specific skeleton wins and the recognised operator is simply wrong.
    N is declared positive, so 0 is not a value it may take.
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if sym.is_positive and not v > 0:
        return False
    if sym.is_nonnegative and not v >= 0:
        return False
    return True


def _infer_numerically(expr, sym, samples: int = 12):
    """Solve `expr == 0` for a single symbol by sampling the other free symbols.

    Needed because a parameter can have no symbolic inverse: solving
    `tanh(beta*c) == tanh(c/2)` returns
        beta = log(-sqrt(-(tanh(c/2) + 1)/(tanh(c/2) - 1))) / c
    which still contains `c` and is therefore not a value. Numerically, fixing `c` makes it
    a one-dimensional root find. The answer is then CHECKED on held-out samples of the
    other symbols, so a spurious root cannot be accepted.
    """
    import random

    free = sorted((f for f in expr.free_symbols if f != sym), key=lambda z: z.name)
    if not free:
        return None
    rng = random.Random(20251012)
    for _ in range(samples):
        point = {f: sp.Rational(rng.randint(1, 30), rng.randint(1, 11)) for f in free}
        one_d = expr.subs(point)
        for guess in (1.0, 0.3, 2.0, 0.05):
            try:
                root = sp.nsolve(one_d, sym, guess, tol=1e-13, maxsteps=80)
            except Exception:
                continue
            if not (root.is_number and root.is_real) or not _respects(sym, root):
                continue
            ok = True
            for _ in range(5):
                held = {f: sp.Rational(rng.randint(1, 30), rng.randint(1, 11)) for f in free}
                try:
                    val = abs(complex(sp.N(expr.subs({**held, sym: root}), 25)))
                except (TypeError, ValueError):
                    ok = False
                    break
                if val > 1e-9:
                    ok = False
                    break
            if ok:
                return sp.Float(root)
    return None


def _solve_for(expr, sym):
    """Solve `expr == 0` for one symbol: symbolic first, numeric as a fallback.

    The numeric path is what handles a parameter with no symbolic inverse. Solving
    `tanh(beta*c) == tanh(c/2)` returns an expression in `c`, not a value, so `beta` can only
    be found numerically; `N` in `c/(N+s)` and `tiny` in `log(a+t) == log(a)` are found
    symbolically. One helper, tried in that order, covers both.
    """
    try:
        sols = sp.solve(sp.Eq(expr, 0), sym, dict=True)
    except (NotImplementedError, ValueError, TypeError):
        sols = []
    for s_ in sols:
        if s_ and len(s_) == 1:
            v = next(iter(s_.values()))
            if v.is_number and v.is_real and _respects(sym, v):
                return v
    return _infer_numerically(expr, sym)
