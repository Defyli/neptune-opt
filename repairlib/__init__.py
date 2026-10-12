"""
General framework for Neptune-style break-and-repair fusion.

Operator-agnostic core. An operator family is described by a *reduction skeleton*:

    f    the associative reducer of the consumer reduction        (Add, Max, Mul, ...)
    g    the term producer:  g(s, c) -> term, where `s` is the value of the producing
         reduction and `c` collects everything else (program inputs)

and we answer mechanically:

    1. does a repair function h exist?                      derive_repair()
    2. is it USABLE for fusion?                             classify()
       i.e. does h distribute over f, so it can correct an ACCUMULATOR rather than
       having to rebuild the result from individual terms?

Conditions (Neptune Eq. 3/4):

    (3) h(g(s, c), s, s') = g(s', c)              replace a stale producer value
    (4) h(x f y, s, s') = h(x,s,s') f h(y,s,s')   distribute over the reducer

Condition (4) is the one that matters. Without it a "repair" cannot be applied to a fold
and there is no fusion.

IMPLEMENTATION NOTE (this is a real trap, hit twice while writing this):
    s and s' must be TWO DISTINCT sympy symbols from the very beginning. If you write g
    with a single symbol `r` and then substitute r -> r_prime to form the fresh version,
    the substitution also rewrites the `r` that came out of the inverse, and h collapses
    to the identity. Every case then looks like "NO_REPAIR with h = t", which is exactly
    the wrong answer this module exists to avoid.
"""
from __future__ import annotations

import dataclasses
import enum
import random
from collections.abc import Callable
from typing import Any

import sympy as sp

# ---------------------------------------------------------------- symbols
# The stale producer value is `s`; the fresh one is `s2`. They are never the same symbol.
x, y = sp.symbols("x y", real=True)             # accumulator operands for condition (4)
s, s2 = sp.symbols("s s2", real=True)           # stale / fresh producer value
c = sp.Symbol("c", real=True)                   # everything not produced by a reduction
t = sp.Symbol("t", positive=True)               # accumulator value fed to h
u = sp.Symbol("u", positive=True)               # result of g, used while inverting


class Verdict(enum.Enum):
    EXACT = "EXACT"
    AFFINE = "AFFINE"
    NEEDS_STATE = "NEEDS_STATE"
    NO_REPAIR = "NO_REPAIR"
    NOT_INVERT = "NOT_INVERT"

    @property
    def fusable(self) -> bool:
        return self is Verdict.EXACT


@dataclasses.dataclass
class RepairResult:
    name: str
    f: Any
    g: Any
    verdict: Verdict
    inverse: Any | None = None
    h: Any | None = None
    cond3_residual: Any | None = None
    cond4_residual: Any | None = None
    notes: list[str] = dataclasses.field(default_factory=list)
    # how the repair has to be applied when it is not a pure accumulator homomorphism
    needs_state: bool = False

    def row(self) -> str:
        h = "—" if self.h is None else str(self.h)
        return (f"{self.name:24s} {self.verdict.value:11s} {h[:46]:46s} "
                f"{'; '.join(self.notes)[:46]}")


# ---------------------------------------------------------------- core
def apply_f(f: Any, a: Any, b: Any) -> Any:
    """f is a sympy expression in (x, y)."""
    return f.subs({x: a, y: b})


def invert_g(g: Any) -> tuple[Any, Any] | None:
    """Solve g(s, c) == u for c. Returns (c_expr, u) or None."""
    sols = sp.solve(sp.Eq(g, u), c)
    if not sols:
        return None
    return sols[0], u


def derive_repair(f: Any, g: Any, *, h_override: Any | None = None) -> RepairResult:
    """Derive h by Neptune Eq.5 and evaluate conditions (3) and (4).

    Eq.5:  h(t, s, s') = g(s', g_c^{-1}(s, t))
    """
    res = RepairResult(name="", f=f, g=g, verdict=Verdict.NO_REPAIR)
    inv = invert_g(g)
    if inv is None:
        res.verdict = Verdict.NOT_INVERT
        res.notes.append("g not invertible in c")
        if h_override is not None:
            return check_conditions(res, h_override)
        return res
    c_inv, _ = inv
    res.inverse = simplify(c_inv)
    if not c_inv.has(s):
        # the inverse does not reference the producer value, so g does not really
        # depend on it through c -- nothing to repair by this construction
        res.verdict = Verdict.NOT_INVERT
        res.notes.append("g_c^{-1} independent of the producer value")
        if h_override is not None:
            return check_conditions(res, h_override)
        return res

    if h_override is not None:
        res.notes.append("h supplied")
        return check_conditions(res, h_override)

    # --- Eq.5, constructed symbolically -----------------------------------------
    # h takes the ACCUMULATOR value t (built under the stale producer value s) and
    # returns the value it would have under the fresh value s'. To build it:
    #     (1) recover c from t under s:   c = g_c^{-1}(s, t)
    #     (2) re-apply g with the FRESH value as first argument:  g(s', c)
    # Traps encountered while writing this (all of them produce an h that silently
    # fails condition (3), so they are worth encoding):
    #   a. using ONE symbol for stale and fresh makes the substitution ill-defined;
    #   b. `g.subs({c: inverse})` can cancel the stale value inside the term
    #      (g = exp(c-s), inverse = s+log(t)) leaving a bare `t`;
    #   c. substituting the stale value right after (b) rewrites the wrong
    #      occurrence and yields h(t,s,s') = g(s, c(t)) = t, i.e. the IDENTITY.
    # The safe construction is to substitute the inverse for c with the stale value
    # renamed to a placeholder, then feed the fresh value in as g's first argument.
    a = sp.Symbol("a", real=True)
    c_recovered = c_inv.xreplace({s: a})        # c = g_c^{-1}(a, t), a = stale value
    h = res.g.xreplace({s: s2})                 # g(s', c) -- fresh value in place
    h = h.subs({c: c_recovered})                # c recovered from the STALE value
    h = h.xreplace({a: s})                      # keep the stale value symbolic for (3)
    h = simplify(h)
    assert h.has(s2), f"repair must reference the fresh value; got {h}"
    assert h.has(s), (
        f"repair must still reference the stale value for (3) to be checkable; got {h}"
    )
    res.notes.append("Eq.5")
    return check_conditions(res, h)


def bind(h: Any, value: Any) -> Any:
    """Apply h to an accumulator value.

    h as constructed by Eq.5 is an expression in `u` (the value produced by g) and the
    producer pair (s, s2). To apply it to a value we substitute u := that value.
    """
    return h.subs({u: value})


def simplify(e: Any) -> Any:
    """sp.simplify, but rewrite hyperbolics through exp first.

    `sp.simplify` leaves `tanh(log(z))` unevaluated, so the softcap repair came out as
        exp(-s2 + tanh(log(-sqrt(-(s + log(u) + 1)/(s + log(u) - 1)))))
    instead of `u*exp(s - s2)`: the same function, but unreadable and untranslatable
    (the code generator has no tanh-of-log case, so a perfectly good operator was being
    reported as unsupported). Rewriting in terms of exp lets the exponentials cancel.
    """
    try:
        rewritten = e.rewrite(sp.exp)
    except (AttributeError, NotImplementedError):
        rewritten = e
    return sp.simplify(sp.simplify(rewritten))


# functions whose "zero on the principal branch" does not mean "identically zero".
# The numeric fallback below must not be trusted in their presence.
_BRANCH_CUT = (sp.log, sp.atanh, sp.acosh, sp.atan, sp.acos, sp.asin)


def _has_branch_cut(expr: Any) -> bool:
    return expr.has(sp.I) or any(expr.has(f) for f in _BRANCH_CUT)


def vanishes(expr: Any, samples: int = 9, tol: float = 1e-9) -> bool:
    """Is `expr` identically zero? Symbolic first, numeric fallback second.

    The fallback is needed because `sp.simplify` cannot prove identities involving Max:
    even for a CONCRETE k,
        sp.simplify(Max(x,y)*k - Max(x*k, y*k))  ->  k*Max(x,y) - Max(k*x, k*y)  != 0
    although both sides are equal for k >= 0. That made three families look like they
    fail condition (4) when in fact they satisfy it, i.e. the classifier reported AFFINE
    where EXACT was correct.

    Soundness: a filter, not a proof. A nonzero residual has to vanish at all sampled
    points to slip through, so several points are used and the symbolic attempt is always
    made first. The verdicts are additionally backed by numeric operator tests
    (repairlib/demo_operators.py, gpu_m10/m12).
    """
    if expr == 0:
        return True
    if sp.simplify(expr) == 0:
        return True
    if _has_branch_cut(expr):
        # e.g. log/atanh introduced by inverting a transcendental g. The residual can be
        # zero on sympy's chosen branch and nonzero on another, and sampling cannot tell
        # the difference (measured: the softcap residual evaluated to ~1e-152 at every
        # sampled point while the two sides differ off-branch). Require a symbolic zero.
        return False
    free = sorted(expr.free_symbols, key=lambda z: z.name)
    if not free:
        return False
    rng = random.Random(20251012)   # fixed seed: a checker must not change between runs
    for _ in range(samples):
        point = {z: sp.Rational(rng.randint(1, 40), rng.randint(1, 17)) for z in free}
        try:
            val = abs(complex(sp.N(expr.subs(point), 30)))
        except (TypeError, ValueError):
            return False
        if val > tol * max(1.0, val):
            return False
    return True


def check_conditions(res: RepairResult, h: Any) -> RepairResult:
    """Evaluate (3) and (4)."""
    # (3): h(g(s,c), s, s') == g(s', c)
    lhs3 = bind(h, res.g)
    rhs3 = res.g.xreplace({s: s2})
    res.cond3_residual = simplify(lhs3 - rhs3)
    ok3 = vanishes(res.cond3_residual)

    # (4): h(x f y, s, s') == h(x,s,s') f h(y,s,s')
    lhs4 = bind(h, apply_f(res.f, x, y))
    rhs4 = apply_f(res.f, bind(h, x), bind(h, y))
    res.cond4_residual = simplify(lhs4 - rhs4)
    ok4 = vanishes(res.cond4_residual)

    res.h = h
    if ok3 and ok4:
        res.verdict = Verdict.EXACT
    elif ok3:
        res.verdict = Verdict.AFFINE
        res.needs_state = True
        res.notes.append("(4) fails -> needs extra carried state")
    else:
        # Distinguish the two ways (3) can fail. If the residual still contains the
        # data symbol c, then no repair taking only (t, s, s') can work: something about
        # c has to be carried across iterations. That is the Welford situation -- the
        # running mean must be maintained alongside the accumulator. If the residual is
        # free of c, the construction simply does not repair this g at all.
        if res.cond3_residual.has(c):
            res.verdict = Verdict.NEEDS_STATE
            res.needs_state = True
            res.notes.append("(3) needs c -> carry extra state (Welford-like)")
        else:
            res.verdict = Verdict.NO_REPAIR
            res.notes.append("(3) fails, c-free -> g cannot be repaired")
    return res


# ---------------------------------------------------------------- corpus
def corpus() -> list[tuple[str, Any, Any, str]]:
    """(name, f, g, how-the-repair-is-applied) for the families we care about."""
    eps = sp.Symbol("eps", positive=True)
    return [
        # --- the one PyTorch ships
        ("online_softmax", x + y, sp.exp(c - s), "accumulator"),
        # --- same algebra, different consumer: log-sum-exp / cross-entropy
        ("logsumexp", x + y, sp.exp(c - s), "accumulator"),
        # --- numerically stable L2 / RMS norm numerator
        ("stable_l2", x + y, (c / s) ** 2, "accumulator"),
        # --- the paper's RMSNorm example: max of a normalised quantity
        ("rmsnorm_max", sp.Max(x, y), c / sp.sqrt(s + eps), "term"),
        # --- variance / Welford (the paper reports failure here)
        ("variance_shift", x + y, (s - c) ** 2, "term"),
        # --- multiplicative reducer
        ("prod_ratio", x * y, c / s, "accumulator"),
        # --- g not invertible in c
        ("noninvertible", x + y, sp.exp(c - sp.Abs(c)), "accumulator"),
        # --- additive-in-log producer with a max reducer
        ("max_shift", sp.Max(x, y), c - s, "accumulator"),
    ]


def analyse(name: str, f: Any, g: Any, how: str = "") -> RepairResult:
    res = derive_repair(f, g)
    res.name = name
    if how:
        res.notes.append(f"apply:{how}")
    return res


def main() -> None:
    w = 108
    print("=" * w)
    print("GENERAL REPAIR CLASSIFIER — which (f, g) pairs admit SAFE fusion?")
    print("=" * w)
    print(f"{'operator':24s} {'verdict':11s} {'h(t, s, s2)':46s} note")
    print("-" * w)
    results = []
    for name, f, g, how in corpus():
        r = analyse(name, f, g, how)
        results.append(r)
        print(r.row())
    print("-" * w)
    n_exact = sum(r.verdict is Verdict.EXACT for r in results)
    print(f"{n_exact}/{len(results)} pairs are EXACT (safe to fold into an accumulator)")
    print("""
Reading:
  EXACT      repair (4) holds -> h corrects an ACCUMULATOR, which is what fusion needs.
  AFFINE     (3) holds, (4) fails -> a per-term correction exists but cannot be
             propagated through the fold without carrying extra state.
  NEEDS_STATE (3) itself needs c -> no repair of the form h(t,s,s') can work; extra
             state must be carried across iterations (Welford carries the mean).
  NO_REPAIR  (3) fails and is c-free -> this construction cannot repair g at all.
  NOT_INVERT Eq.5 cannot be instantiated at all.

The shipped PyTorch implementation covers exactly ONE of these rows.""")
    print("=" * w)


if __name__ == "__main__":
    main()
